from contextlib import asynccontextmanager
from functools import partial
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import RedirectResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import csv
import io
import os
import json
import time
import uuid
from pathlib import Path
from urllib.parse import quote_plus
from urllib import request as urllib_request

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware

from api.auth import (
    ALLOW_OPEN_REGISTRATION,
    AdminUser,
    CurrentUser,
    DEFAULT_ORG_ID,
    TokenUser,
    create_access_token,
    create_user,
    decode_token,
    get_user_by_username,
    init_user_db,
    user_can_access_encounter,
    verify_password,
)

from google import genai
from google.genai import types
import re
import requests

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from src.snowflake_client import get_connection
from src.clinical_workflow import (
    EncounterInput,
    answer_and_update_tx,
    add_answer,
    add_question,
    append_audit_row,
    create_encounter,
    create_encounter_tx,
    ensure_clinical_tables,
    fetch_audit_action_summary,
    fetch_audit_rows_for_export,
    ingest_followup_answer_tokens,
    fetch_idempotent_response,
    get_encounter,
    get_encounter_kg_preview,
    normalize_encounter_dict,
    get_latest_kg_build_meta,
    hash_payload,
    merge_candidate_rankings,
    next_turn_no,
    purge_encounter_data,
    rank_diseases_from_context_tokens,
    rank_diseases_from_symptom_map,
    rank_diseases_from_graph,
    save_differential,
    seed_graph_from_symptom_map,
    store_idempotent_response,
)
from src.followup_policy import MAX_TURNS_DEFAULT, next_question as policy_next_question, safety_rails
from src.grounding import extract_urls_from_context, validate_grounding
from src.response_contract import contract_to_markdown, normalize_contract, validate_contract
from src.mlflow_telemetry import log_assessment_metrics
from src.agentic.assessment_graph import build_assess_graph
from src.report_pdf import build_encounter_pdf
from src.retrieval.rerank import maybe_rerank_literature

_log = logging.getLogger("medassist.api")
_DEBUG_LOG_PATH = "/Users/atharvakurlekar/Library/CloudStorage/OneDrive-NortheasternUniversity/Data Engineering/med/MedAssist.AI/.cursor/debug-742e45.log"
_DEBUG_ENDPOINT = "http://127.0.0.1:7299/ingest/6c1651b6-79fe-48a7-a0a7-0d0f9a35fdde"


def _api_debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    if os.environ.get("DEBUG", "").strip().lower() not in ("1", "true", "yes"):
        return
    _log.debug("%s %s %s %s", hypothesis_id, location, message, json.dumps(data, default=str))


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": "742e45",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        req = urllib_request.Request(
            _DEBUG_ENDPOINT,
            data=json.dumps(payload, default=str).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Debug-Session-Id": "742e45",
            },
            method="POST",
        )
        urllib_request.urlopen(req, timeout=0.7).read()
    except Exception:
        return


# region agent log
_PERF_DEBUG_LOG_064 = Path(__file__).resolve().parent.parent / ".cursor" / "debug-064a4f.log"
_PERF064_SESSION = "064a4f"


def _perf_debug_064(
    hypothesis_id: str,
    location: str,
    message: str,
    *,
    pipeline: str,
    encounter_id: str,
    phase: str,
    elapsed_ms: float,
    extra: dict | None = None,
) -> None:
    """NDJSON timing for debug session 064a4f (no secrets / no PII)."""
    payload = {
        "sessionId": _PERF064_SESSION,
        "timestamp": int(time.time() * 1000),
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": {
            "pipeline": pipeline,
            "encounter_id": encounter_id,
            "phase": phase,
            "elapsed_ms": round(elapsed_ms, 2),
            **(extra or {}),
        },
    }
    try:
        _PERF_DEBUG_LOG_064.parent.mkdir(parents=True, exist_ok=True)
        with open(_PERF_DEBUG_LOG_064, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass
    try:
        req = urllib_request.Request(
            _DEBUG_ENDPOINT,
            data=json.dumps(payload, default=str).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Debug-Session-Id": _PERF064_SESSION,
            },
            method="POST",
        )
        urllib_request.urlopen(req, timeout=0.7).read()
    except Exception:
        pass


# endregion

# RAG (ChromaDB) - optional; falls back to ILIKE if unavailable
try:
    from src.indexing.rag_index import query_rag
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False
    query_rag = None

# Local embedding model for Cortex vector search (same as build_rag_index).
# This is optional so the app can still load (and use Cortex COMPLETE / Gemini fallback)
# even when torch/sentence-transformers aren’t fully functional in the active venv.
_rag_embed_model = None


def _get_rag_embed_model():
    global _rag_embed_model
    if _rag_embed_model is not None:
        return _rag_embed_model
    if SentenceTransformer is None:
        return None
    try:
        _rag_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _rag_embed_model = None
    return _rag_embed_model


# Configure Vertex AI client (uses gcloud ADC, no API key)
client = genai.Client(
    vertexai=True,
    project="medassistai-488422",
    location="global",  # Gemini 3 preview models are served from the global location
)

MODEL_ID = "gemini-3-flash-preview"

# Option B: Vertex AI Search data store path for GCS grounding (set via env)
VERTEX_AI_DATASTORE_PATH = os.environ.get("VERTEX_AI_DATASTORE_PATH")
if not VERTEX_AI_DATASTORE_PATH and os.environ.get("VERTEX_AI_DATASTORE_ID"):
    VERTEX_AI_DATASTORE_PATH = (
        "projects/medassistai-488422/locations/global/"
        "collections/default_collection/dataStores/"
        + os.environ["VERTEX_AI_DATASTORE_ID"]
    )

CLINICAL_USE_CORTEX = os.environ.get("CLINICAL_USE_CORTEX", "0") == "1"
CLINICAL_MAX_TURNS = int(os.environ.get("CLINICAL_MAX_TURNS", str(MAX_TURNS_DEFAULT)))
FOLLOWUP_SELF_CRITIQUE = os.environ.get("FOLLOWUP_SELF_CRITIQUE", "").strip().lower() in ("1", "true", "yes")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    init_user_db()
    yield


app = FastAPI(title="MedAssist.AI API", lifespan=_lifespan)

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)
except Exception:
    pass


class _TraceIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        tid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.trace_id = tid
        response = await call_next(request)
        response.headers["X-Request-ID"] = tid
        return response


app.add_middleware(_TraceIdMiddleware)

_cors_origins = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS",
        "http://127.0.0.1:8501,http://localhost:8501,http://127.0.0.1:3000,http://localhost:3000",
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app)
except Exception:
    pass

try:
    from prometheus_client import Counter, Histogram

    _PROM_RAG = Counter("medassist_rag_fetch_total", "RAG retrieval outcomes", ["result"])
    _PROM_EVIDENCE_MODE = Counter("medassist_evidence_mode_total", "Journal-first evidence path", ["mode"])
    _PROM_ASSESS_FAST = Histogram(
        "medassist_assess_fast_seconds",
        "Assess-fast pipeline wall time",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 60.0, 120.0),
    )
except Exception:
    _PROM_RAG = None
    _PROM_EVIDENCE_MODE = None
    _PROM_ASSESS_FAST = None


def _prom_rag_result(hit: bool) -> None:
    if _PROM_RAG:
        _PROM_RAG.labels(result="hit" if hit else "miss").inc()


def _prom_evidence_mode(mode: str) -> None:
    if _PROM_EVIDENCE_MODE:
        _PROM_EVIDENCE_MODE.labels(mode=mode).inc()


# Primary UX is Streamlit (full console). Override with STREAMLIT_PUBLIC_URL if needed.
STREAMLIT_PUBLIC_URL = os.environ.get("STREAMLIT_PUBLIC_URL", "http://127.0.0.1:8501")

_bearer_optional = HTTPBearer(auto_error=False)


def _assert_encounter_access(user: TokenUser, encounter_id: str) -> dict:
    encounter = get_encounter(encounter_id)
    if not encounter:
        raise HTTPException(status_code=404, detail="Encounter not found")
    if not user_can_access_encounter(user, encounter):
        raise HTTPException(status_code=403, detail="Not allowed to access this encounter")
    return encounter


class LoginBody(BaseModel):
    username: str
    password: str


class RegisterBody(BaseModel):
    username: str
    password: str
    role: str = "doctor"
    org_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/login")
def auth_login(body: LoginBody):
    user = get_user_by_username(body.username)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(
        sub=user["id"],
        username=user["username"],
        role=user["role"],
        org_id=user["org_id"],
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "org_id": user["org_id"],
        "user_id": user["id"],
    }


@app.post("/auth/register")
def auth_register(
    body: RegisterBody,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_optional)],
):
    """Create a user. Open registration (doctor-only) when ALLOW_OPEN_REGISTRATION=1; otherwise requires admin Bearer token."""
    org = (body.org_id or "").strip() or DEFAULT_ORG_ID
    role = (body.role or "doctor").lower()
    if ALLOW_OPEN_REGISTRATION:
        if role != "doctor":
            raise HTTPException(status_code=400, detail="Open registration only supports role=doctor")
        try:
            create_user(body.username, body.password, "doctor", org)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"ok": True}
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Admin Bearer token required when open registration is disabled")
    try:
        claims = decode_token(creds.credentials)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc
    if str(claims.get("role", "")).lower() != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    if role not in ("doctor", "admin"):
        raise HTTPException(status_code=400, detail="role must be doctor or admin")
    try:
        create_user(body.username, body.password, role, org)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True}


@app.get("/auth/me")
def auth_me(user: CurrentUser):
    return {"user_id": user.sub, "username": user.username, "role": user.role, "org_id": user.org_id}


@app.get("/admin/metrics/summary")
def admin_metrics_summary(
    _admin: AdminUser,
    since: str | None = Query(None, description="ISO timestamp lower bound (Snowflake)"),
    until: str | None = Query(None, description="ISO timestamp upper bound"),
):
    """Aggregate audit counts by action (admin only). Requires Snowflake."""
    ensure_clinical_tables()
    actions = fetch_audit_action_summary(since_iso=since, until_iso=until)
    return {"actions": actions, "window": {"since": since, "until": until}}


@app.get("/admin/audit/export.csv")
def admin_audit_export_csv(
    _admin: AdminUser,
    since: str | None = Query(None),
    until: str | None = Query(None),
    limit: int = Query(5000, ge=1, le=20000),
):
    """Download bounded audit rows as CSV (admin only)."""
    ensure_clinical_tables()
    rows = fetch_audit_rows_for_export(since_iso=since, until_iso=until, limit=limit)
    buf = io.StringIO()
    if not rows:
        return Response(content="", media_type="text/csv")
    fieldnames = list(rows[0].keys())
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        flat = dict(r)
        pm = flat.get("pipeline_metrics")
        if pm is not None and not isinstance(pm, str):
            flat["pipeline_metrics"] = json.dumps(pm, default=str)
        w.writerow(flat)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="medassist_encounter_audit.csv"'},
    )


@app.post("/admin/encounters/{encounter_id}/purge")
def admin_purge_encounter(encounter_id: str, _admin: AdminUser):
    """Delete encounter and dependent rows from Snowflake (admin only; irreversible)."""
    ensure_clinical_tables()
    purge_encounter_data(encounter_id)
    return {"ok": True, "encounter_id": encounter_id}


@app.post("/analytics/event")
def analytics_event(
    user: CurrentUser,
    tab: str = Query(..., description="UI tab or feature id"),
    event: str = Query("open", description="e.g. open, click"),
):
    """Privacy-preserving product analytics hook (no PHI); optional Snowflake sink can be added later."""
    _log.info("analytics_event tab=%s event=%s org=%s user=%s", tab, event, user.org_id, user.sub)
    return {"ok": True}


@app.get("/")
def index():
    """Redirect to the Streamlit doctor console (single UI)."""
    return RedirectResponse(url=STREAMLIT_PUBLIC_URL, status_code=302)


class Question(BaseModel):
    question: str
    brief: bool = False  # if True, use short answer + references only


class SymptomInput(BaseModel):
    symptom: str
    onset: str | None = None
    severity: str | None = None
    duration: str | None = None


class EncounterStartRequest(BaseModel):
    age: int | None = None
    sex: str | None = None
    known_conditions: list[str] = []
    medications: list[str] = []
    allergies: list[str] = []
    history_summary: str = ""
    symptoms: list[SymptomInput] = []


class EncounterAnswerRequest(BaseModel):
    turn_no: int
    answer: str


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "may",
    "might",
    "of",
    "on",
    "or",
    "our",
    "should",
    "that",
    "the",
    "their",
    "then",
    "these",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "your",
    # Common query scaffolding words that are not clinical terms
    "condition",
    "conditions",
    "present",
    "presenting",
    "differential",
    "diagnosis",
    "diagnoses",
    "red",
    "flag",
    "flags",
    "consider",
    "associated",
    "symptom",
    "symptoms",
    "treatment",
    "treatments",
    "next",
    "steps",
    "suspected",
    "cause",
    "causes",
}


def _extract_query_terms(question: str, *, max_terms: int = 8) -> list[str]:
    """
    Extract useful query terms from a natural-language question for SQL ILIKE matching.
    Keeps short list of non-stopword tokens to avoid pulling irrelevant matches.
    """
    text = (question or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    terms: list[str] = []
    for t in tokens:
        if len(t) <= 2:
            continue
        if t in _STOPWORDS:
            continue
        if t not in terms:
            terms.append(t)
        if len(terms) >= max_terms:
            break
    # Adjacent token pairs improve ILIKE recall for short clinical phrases.
    base = list(terms)
    for i in range(len(base) - 1):
        if len(terms) >= max_terms:
            break
        pair = f"{base[i]} {base[i + 1]}"
        if pair not in terms:
            terms.append(pair)
    return terms


def fetch_all_literature_context(question: str, limit: int = 30) -> str:
    """
    Search PubMed, PMC, NCBI Bookshelf, and OpenStax in one go (all sources at once).
    Returns combined context for the LLM, including article URLs for references.
    """
    conn = get_connection()
    cur = conn.cursor()
    like = f"%{question}%"
    try:
        # Union all four sources; include URL/link for each row so the model can cite them
        sql = """
            SELECT source, title, abstract, url
            FROM (
                SELECT 'pubmed' AS source, title, abstract,
                    'https://pubmed.ncbi.nlm.nih.gov/' || CAST(pmid AS VARCHAR) || '/' AS url
                FROM RAW.PUBMED_ARTICLES
                UNION ALL
                SELECT 'pmc', title, abstract,
                    CASE WHEN pmcid LIKE 'PMC%%' THEN 'https://www.ncbi.nlm.nih.gov/pmc/articles/' || pmcid || '/'
                         ELSE 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC' || pmcid || '/' END AS url
                FROM RAW.PMC_ARTICLES
                UNION ALL
                SELECT 'ncbi_bookshelf', title, abstract, COALESCE(url, '') AS url
                FROM RAW.NCBI_BOOKSHELF
                UNION ALL
                SELECT 'openstax', title, content AS abstract, COALESCE(source_url, '') AS url
                FROM RAW.OPENSTAX_BOOKS
            ) unified
            WHERE (title ILIKE %s OR abstract ILIKE %s)
            LIMIT %s
        """
        cur.execute(sql, (like, like, limit))
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    if not rows:
        return "No matching articles were found across PubMed, PMC, NCBI Bookshelf, or OpenStax."

    parts: list[str] = []
    for idx, (source, title, abstract, url) in enumerate(rows, start=1):
        text = (abstract or "")[:8000]  # cap length per doc to avoid token overflow
        url_str = (url or "").strip()
        if url_str:
            parts.append(
                f"[{idx}] Source: {source}\nTitle: {title}\nURL: {url_str}\nText: {text}\n"
            )
        else:
            parts.append(
                f"[{idx}] Source: {source}\nTitle: {title}\nText: {text}\n"
            )
    return "\n---\n\n".join(parts)


def _fetch_rag_entries(question: str, limit: int = 15) -> list[dict]:
    """
    Semantic search over local ChromaDB RAG index.
    Returns list of {source, title, url, text}. Empty list if RAG unavailable or no results.
    """
    if not _RAG_AVAILABLE or query_rag is None:
        return []
    try:
        results = query_rag(question, n_results=limit)
    except Exception:
        return []
    if not results:
        return []

    entries = []
    for r in results:
        doc = r.get("document", "")
        meta = r.get("metadata", {})
        source = meta.get("source", "unknown")
        title = (meta.get("title") or meta.get("url") or "")[:500]
        url = meta.get("url", "")
        if source == "pubmed" and meta.get("pmid"):
            url = f"https://pubmed.ncbi.nlm.nih.gov/{meta['pmid']}/"
        elif source == "pmc" and meta.get("pmcid"):
            pmcid = str(meta["pmcid"])
            url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid.startswith("PMC") else f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"
        text = (doc or "")[:8000]
        entries.append({"source": source, "title": title, "url": url, "text": text})
    return entries


def _fetch_ilike_entries(question: str, limit: int = 30) -> list[dict]:
    """
    Keyword search over Snowflake (PubMed, PMC, NCBI Bookshelf, OpenStax).
    Returns list of {source, title, url, text}.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        terms = _extract_query_terms(question)
        if not terms:
            terms = [question.strip()] if (question or "").strip() else []

        # Build a loose OR clause over tokens, so natural-language questions still match content.
        # Example: "fever vomiting differential" becomes (title ILIKE %fever% OR abstract ILIKE %fever% OR ...)
        token_clause = " OR ".join(["title ILIKE %s OR abstract ILIKE %s"] * len(terms)) if terms else "FALSE"
        params: list[str | int] = []
        for t in terms:
            like = f"%{t}%"
            params.extend([like, like])

        sql = """
            SELECT source, title, abstract, url
            FROM (
                SELECT 'pubmed' AS source, title, abstract,
                    'https://pubmed.ncbi.nlm.nih.gov/' || CAST(pmid AS VARCHAR) || '/' AS url
                FROM RAW.PUBMED_ARTICLES
                UNION ALL
                SELECT 'pmc', title, abstract,
                    CASE WHEN pmcid LIKE 'PMC%%' THEN 'https://www.ncbi.nlm.nih.gov/pmc/articles/' || pmcid || '/'
                         ELSE 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC' || pmcid || '/' END AS url
                FROM RAW.PMC_ARTICLES
                UNION ALL
                SELECT 'ncbi_bookshelf', title, abstract, COALESCE(url, '') AS url
                FROM RAW.NCBI_BOOKSHELF
                UNION ALL
                SELECT 'openstax', title, content AS abstract, COALESCE(source_url, '') AS url
                FROM RAW.OPENSTAX_BOOKS
            ) unified
            WHERE ({token_clause})
            LIMIT %s
        """
        sql = sql.format(token_clause=token_clause)
        cur.execute(sql, tuple(params + [limit]))
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    entries = []
    for source, title, abstract, url in (rows or []):
        text = (abstract or "")[:8000]
        url_str = (url or "").strip()
        entries.append({"source": source, "title": title, "url": url_str, "text": text})
    return entries


def _merge_and_dedupe_literature(rag_entries: list[dict], ilike_entries: list[dict]) -> list[dict]:
    """
    Merge RAG (semantic) and ILIKE (keyword) results, deduplicating by URL or (source, title).
    RAG results come first; ILIKE fills in gaps.
    """
    seen: set[tuple[str, str]] = set()
    merged: list[dict] = []

    def key(e: dict) -> tuple:
        url = (e.get("url") or "").strip()
        if url:
            return ("url", url)
        return ("title", (e.get("source") or ""), (e.get("title") or "")[:200])

    for e in rag_entries:
        k = key(e)
        if k not in seen:
            seen.add(k)
            merged.append(e)

    for e in ilike_entries:
        k = key(e)
        if k not in seen:
            seen.add(k)
            merged.append(e)

    return merged


def _format_literature_context(entries: list[dict]) -> str:
    """Format merged entries into context string for the LLM."""
    if not entries:
        return "No matching articles were found across PubMed, PMC, NCBI Bookshelf, or OpenStax."
    parts = []
    for idx, e in enumerate(entries, start=1):
        source = e.get("source", "unknown")
        title = e.get("title", "")
        url = (e.get("url") or "").strip()
        text = (e.get("text") or "")[:8000]
        if url:
            parts.append(f"[{idx}] Source: {source}\nTitle: {title}\nURL: {url}\nText: {text}\n")
        else:
            parts.append(f"[{idx}] Source: {source}\nTitle: {title}\nText: {text}\n")
    return "\n---\n\n".join(parts)


def _normalize_evidence_entries(entries: list[dict], *, mode: str) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        out.append(
            {
                "source": (e.get("source") or "unknown").strip(),
                "title": (e.get("title") or "").strip()[:500],
                "url": (e.get("url") or "").strip(),
                "snippet": (e.get("text") or "").strip()[:500],
                "mode": mode,
            }
        )
    return out


def _is_evidence_sufficient(entries: list[dict], *, min_items: int = 3, min_with_url: int = 2) -> bool:
    if len(entries) < min_items:
        return False
    with_url = sum(1 for e in entries if (e.get("url") or "").strip())
    return with_url >= min_with_url


def _fetch_web_entries(question: str, limit: int = 8) -> list[dict]:
    """
    Lightweight web-assisted fallback using DuckDuckGo HTML endpoint.
    This is used only when journal-first evidence is insufficient.
    """
    q = (question or "").strip()
    if not q:
        return []
    try:
        # Bias towards medical/journal content with explicit query hint.
        url = f"https://duckduckgo.com/html/?q={quote_plus(q + ' site:ncbi.nlm.nih.gov OR site:who.int OR site:nih.gov')}"
        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "MedAssistAI/1.0 (clinical-evidence-retrieval)"},
        )
        resp.raise_for_status()
        html = resp.text
    except Exception:
        return []

    # Extract coarse result links and titles from HTML blocks.
    matches = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    out: list[dict] = []
    for href, title_html in matches[: limit * 3]:
        title = re.sub(r"<[^>]+>", "", title_html or "").strip()
        if not href or not title:
            continue
        source = "web"
        lhref = href.lower()
        if "pubmed.ncbi.nlm.nih.gov" in lhref:
            source = "pubmed_web"
        elif "ncbi.nlm.nih.gov/pmc" in lhref:
            source = "pmc_web"
        elif "who.int" in lhref:
            source = "who_web"
        elif "nih.gov" in lhref:
            source = "nih_web"
        out.append({"source": source, "title": title[:500], "url": href, "text": ""})
        if len(out) >= limit:
            break
    return out


def _fetch_live_pubmed_entries(question: str, limit: int = 8) -> list[dict]:
    """Live PubMed retrieval via NCBI E-utilities (JSON endpoints)."""
    q = (question or "").strip()
    if not q:
        return []
    try:
        esearch = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "retmode": "json", "retmax": str(limit), "sort": "relevance", "term": q},
            timeout=10,
        )
        esearch.raise_for_status()
        ids = (esearch.json().get("esearchresult") or {}).get("idlist") or []
        if not ids:
            return []
        esummary = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "retmode": "json", "id": ",".join(ids[:limit])},
            timeout=10,
        )
        esummary.raise_for_status()
        result = (esummary.json() or {}).get("result") or {}
    except Exception:
        return []

    rows: list[dict] = []
    for pmid in ids[:limit]:
        node = result.get(str(pmid)) or {}
        title = (node.get("title") or "").strip()
        if not title:
            continue
        rows.append(
            {
                "source": "pubmed_live",
                "title": title[:500],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "text": (node.get("source") or "")[:250],
            }
        )
    return rows


def _fetch_europe_pmc_entries(question: str, limit: int = 8) -> list[dict]:
    """Live medical literature fallback from Europe PMC."""
    q = (question or "").strip()
    if not q:
        return []
    try:
        resp = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": q, "format": "json", "pageSize": str(limit)},
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json() or {}
    except Exception:
        return []
    out: list[dict] = []
    items = ((data.get("resultList") or {}).get("result")) or []
    for it in items[:limit]:
        title = (it.get("title") or "").strip()
        if not title:
            continue
        pmid = (it.get("pmid") or "").strip()
        pmcid = (it.get("pmcid") or "").strip()
        if pmid:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        elif pmcid:
            url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
        else:
            url = (it.get("doi") and f"https://doi.org/{it.get('doi')}") or ""
        out.append(
            {
                "source": "europe_pmc_live",
                "title": title[:500],
                "url": url,
                "text": (it.get("journalTitle") or "")[:250],
            }
        )
    return out


_SOURCE_QUALITY_WEIGHTS = {
    "pubmed": 1.0,
    "pubmed_live": 1.0,
    "pmc": 0.95,
    "pmc_web": 0.95,
    "europe_pmc_live": 0.9,
    "orphanet": 0.95,
    "ncbi_bookshelf": 0.85,
    "nih_web": 0.85,
    "who_web": 0.85,
    "openstax": 0.6,
    "web": 0.35,
}


def _evaluate_evidence_quality(entries: list[dict]) -> dict:
    if not entries:
        return {"score": 0.0, "trusted_count": 0, "count": 0, "sufficient": False}
    score = 0.0
    trusted_count = 0
    for e in entries:
        src = (e.get("source") or "web").strip().lower()
        w = _SOURCE_QUALITY_WEIGHTS.get(src, 0.35)
        if w >= 0.85:
            trusted_count += 1
        score += w
    count = len(entries)
    avg = score / max(1, count)
    # Require enough items and enough trusted medical sources.
    sufficient = count >= 3 and trusted_count >= 2 and avg >= 0.7
    return {
        "score": round(avg, 3),
        "trusted_count": trusted_count,
        "count": count,
        "sufficient": sufficient,
    }


def _audit_pipeline_metrics_assess(
    *,
    candidates: list[dict],
    errors: list[str],
    evidence_entries: list[dict],
    fallback_mode: str,
    t0: float,
) -> dict:
    eq = _evaluate_evidence_quality(evidence_entries)
    return {
        "evidence_fallback_mode": fallback_mode,
        "evidence_quality_score": float(eq["score"]),
        "evidence_sufficient": bool(eq["sufficient"]),
        "evidence_trusted_count": int(eq["trusted_count"]),
        "evidence_total_count": int(eq["count"]),
        "n_candidates": len(candidates),
        "n_top_candidates": min(8, len(candidates)),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "error_flags_count": len(errors),
    }


def retrieve_evidence_journal_first(question: str, *, limit: int = 30) -> tuple[list[dict], str]:
    """
    Returns (entries, fallback_mode).
    fallback_mode:
      - journal_first: literature evidence was sufficient
      - web_assisted: journal evidence insufficient, web fallback merged in
      - insufficient: still weak after fallback
    """
    rag_entries = _fetch_rag_entries(question, limit=15)
    _prom_rag_result(bool(rag_entries))
    ilike_entries: list[dict] = []
    if len(rag_entries) < 5:
        ilike_entries = _fetch_ilike_entries(question, limit=limit)
    journal_entries = _merge_and_dedupe_literature(rag_entries, ilike_entries)
    journal_entries = maybe_rerank_literature(question, journal_entries, top_k=max(limit, 40))
    if _is_evidence_sufficient(journal_entries):
        _prom_evidence_mode("journal_first")
        return journal_entries[:limit], "journal_first"

    live_pubmed = _fetch_live_pubmed_entries(question, limit=min(10, limit))
    combined = _merge_and_dedupe_literature(journal_entries, live_pubmed)
    if _is_evidence_sufficient(combined, min_items=3, min_with_url=2):
        _prom_evidence_mode("web_assisted")
        return combined[:limit], "web_assisted"

    europe_pmc = _fetch_europe_pmc_entries(question, limit=min(10, limit))
    combined = _merge_and_dedupe_literature(combined, europe_pmc)
    if _is_evidence_sufficient(combined, min_items=3, min_with_url=2):
        _prom_evidence_mode("web_assisted")
        return combined[:limit], "web_assisted"

    web_entries = _fetch_web_entries(question, limit=min(8, limit))
    combined = _merge_and_dedupe_literature(combined, web_entries)
    if _is_evidence_sufficient(combined, min_items=3, min_with_url=2):
        _prom_evidence_mode("web_assisted")
        return combined[:limit], "web_assisted"
    out = combined[:limit]
    _prom_evidence_mode("insufficient")
    return out, "insufficient"


def fetch_all_literature_context_hybrid(question: str, limit: int = 30) -> str:
    """
    Hybrid retrieval: try RAG (ChromaDB) first, fall back to ILIKE (Snowflake) if few results.
    Merge and deduplicate, then return formatted context for the LLM.
    """
    merged, _mode = retrieve_evidence_journal_first(question, limit=limit)
    return _format_literature_context(merged)


def fetch_symptom_disease_context(question: str, limit: int = 50) -> str:
    """
    Look up symptom → disease from NORMALIZED.SYMPTOM_DISEASE_MAP (Orphanet).
    Helps with differential diagnosis (e.g. 'fever vomiting' → rare diseases).
    """
    conn = get_connection()
    cur = conn.cursor()
    # Symptoms are usually content words; strip punctuation + common stopwords to avoid noisy matches.
    words = _extract_query_terms(question, max_terms=6)
    if not words:
        return ""

    try:
        patterns = [f"%{w}%" for w in words]
        or_clause = " OR ".join(["symptom ILIKE %s"] * len(patterns))
        sql = f"SELECT symptom, orpha_code, disease_name, frequency FROM NORMALIZED.SYMPTOM_DISEASE_MAP WHERE ({or_clause}) LIMIT %s"
        cur.execute(sql, patterns + [limit])
        rows = cur.fetchall()
    except Exception:
        rows = []
    finally:
        cur.close()
        conn.close()

    if not rows:
        return ""

    lines = ["Symptom → Rare diseases (Orphanet):"]
    seen = set()
    for symptom, orpha_code, disease_name, freq in rows:
        key = (orpha_code, disease_name)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  - {disease_name} (ORPHA:{orpha_code}); symptom: {symptom}; frequency: {freq or 'N/A'}")
    return "\n".join(lines)


def fetch_pubmed_context(question: str, limit: int = 20) -> str:
    """
    Fetch a few relevant PubMed articles from Snowflake and format as plain-text context.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        sql = """
            SELECT title, abstract
            FROM RAW.PUBMED_ARTICLES
            WHERE title ILIKE %s OR abstract ILIKE %s
            ORDER BY pub_date DESC
            LIMIT %s
        """
        like = f"%{question}%"
        cur.execute(sql, (like, like, limit))
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    if not rows:
        return "No matching PubMed articles were found for this question."

    parts: list[str] = []
    for idx, (title, abstract) in enumerate(rows, start=1):
        parts.append(
            f"Article {idx}:\n"
            f"Title: {title}\n"
            f"Abstract: {abstract or ''}\n"
        )
    return "\n\n---\n\n".join(parts)


# Snowflake Cortex model for COMPLETE (override via CORTEX_MODEL env; see SHOW MODELS IN SNOWFLAKE.CORTEX)
_CORTEX_DEFAULT = "claude-sonnet-4-6"
CORTEX_MODEL = (os.environ.get("CORTEX_MODEL") or _CORTEX_DEFAULT).strip() or _CORTEX_DEFAULT


def cortex_complete(prompt: str) -> str:
    """Call Snowflake CORTEX.COMPLETE with the given prompt. Returns the model response text."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS response",
            (CORTEX_MODEL, prompt),
        )
        row = cur.fetchone()
        return (row[0] or "").strip() if row else ""
    except Exception as e:
        return f"[Cortex error: {e}]"
    finally:
        cur.close()
        conn.close()


def cortex_diagnostic_answer(question: str) -> str:
    """
    End-to-end MedRAG-style reasoning in Snowflake Cortex.

    - Embeds the question locally with sentence-transformers (same model as RAG index).
    - Retrieves top-k semantic matches from VECTORS.RAG_CHUNKS.
    - Retrieves candidate rare diseases from NORMALIZED.SYMPTOM_DISEASE_MAP.
    - Builds a structured prompt that asks for diagnoses, treatments, and follow-up questions.
    - Calls SNOWFLAKE.CORTEX.COMPLETE and returns the answer text.
    """
    embed_model = _get_rag_embed_model()
    if embed_model is None:
        return "[Cortex diagnostic error: embedding model unavailable in this environment]"

    # 1) Embed question locally to a 384-dim vector (all-MiniLM-L6-v2)
    q_vec = embed_model.encode([question])[0]
    q_vec_str = "[" + ",".join(str(float(x)) for x in q_vec) + "]"

    conn = get_connection()
    cur = conn.cursor()
    try:
        sql = """
            WITH
            q AS (
                SELECT PARSE_JSON(%s)::VECTOR(FLOAT, 384) AS q_vec
            ),
            rag AS (
                SELECT
                    r.chunk_id,
                    r.source,
                    r.document_text,
                    r.metadata,
                    VECTOR_COSINE_SIMILARITY(q.q_vec, r.embedding) AS sim
                FROM VECTORS.RAG_CHUNKS AS r,
                     q
                ORDER BY sim DESC
                LIMIT 12
            ),
            rag_ranked AS (
                SELECT
                    chunk_id,
                    source,
                    document_text,
                    metadata,
                    sim,
                    ROW_NUMBER() OVER (ORDER BY sim DESC) AS rn
                FROM rag
            ),
            rag_text AS (
                SELECT
                    LISTAGG(
                        CONCAT(
                            '[',
                            rn,
                            '] Source: ',
                            source,
                            '; Snippet: ',
                            LEFT(document_text, 600)
                        ),
                        '\n\n'
                    ) WITHIN GROUP (ORDER BY rn) AS text
                FROM rag_ranked
            ),
            tokens AS (
                SELECT DISTINCT LOWER(TRIM(value::string)) AS token
                FROM TABLE(SPLIT_TO_TABLE(%s, ' '))
                WHERE LENGTH(token) > 3
            ),
            orpha_candidates AS (
                SELECT
                    d.orpha_code,
                    d.disease_name,
                    ARRAY_AGG(DISTINCT s.symptom) AS symptom_list
                FROM NORMALIZED.SYMPTOM_DISEASE_MAP AS s
                JOIN NORMALIZED.DISEASES AS d
                  ON d.orpha_code = s.orpha_code
                WHERE EXISTS (
                    SELECT 1
                    FROM tokens t
                    WHERE s.symptom ILIKE '%%' || t.token || '%%'
                )
                GROUP BY d.orpha_code, d.disease_name
                LIMIT 25
            ),
            orpha_text AS (
                SELECT
                    LISTAGG(
                        CONCAT(
                            '- ',
                            disease_name,
                            ' (ORPHA:',
                            orpha_code,
                            '); symptoms: ',
                            ARRAY_TO_STRING(symptom_list, ', ')
                        ),
                        '\n'
                    ) WITHIN GROUP (ORDER BY disease_name) AS text
                FROM orpha_candidates
            ),
            prompt AS (
                SELECT
                    CONCAT(
                        'You are MedAssist.AI, a medical assistant for clinicians.\n\n',
                        'Use ONLY the provided context and rare-disease candidates as your primary evidence. ',
                        'If the context is incomplete or multiple diagnoses are plausible, you MUST:\n',
                        '1) Explicitly say that information is limited or ambiguous, and\n',
                        '2) Propose 3–6 very specific follow-up questions that would help distinguish between diseases.\n\n',
                        'Respond in Markdown with the following sections:\n',
                        '1. **Summary** – 2–3 sentences.\n',
                        '2. **Likely Diagnoses** – bullet list, split into common vs rare (mark rare items and include ORPHA codes when given).\n',
                        '3. **Reasoning** – short explanation of why these diagnoses fit (refer to key manifestations and context).\n',
                        '4. **Treatment & Medication Suggestions** – bullet list; be conservative and mention when specialist input is needed.\n',
                        '5. **Follow-up Questions** – 3–6 concrete questions you would ask the clinician or patient.\n',
                        '6. **References** – 3–5 references as clickable markdown links in the format [Title](URL), using URLs from the retrieved context when available.\n\n',
                        'Question:\n',
                        %s,
                        '\n\n',
                        'Retrieved medical context (RAG over literature and guidelines):\n',
                        COALESCE((SELECT text FROM rag_text), 'None.\n'),
                        '\n\n',
                        'Candidate rare diseases from Orphanet (symptom→disease index):\n',
                        COALESCE((SELECT text FROM orpha_text), 'None.\n')
                    ) AS full_prompt
            )
            SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, full_prompt) AS answer
            FROM prompt
        """
        # Parameters: embedded query vector JSON, question for token splitting,
        # question for prompt body, model id for COMPLETE.
        cur.execute(sql, (q_vec_str, question, question, CORTEX_MODEL))
        row = cur.fetchone()
        return (row[0] or "").strip() if row else ""
    except Exception as e:
        return f"[Cortex diagnostic error: {e}]"
    finally:
        cur.close()
        conn.close()


def _build_medassist_prompt(question: str, context: str) -> str:
    """Shared prompt for Gemini and Cortex (same context, same structure)."""
    return (
        "You are MedAssist.AI, a medical assistant for clinicians.\n\n"
        "Use the context below (from PubMed, PMC, NCBI Bookshelf, OpenStax, and Orphanet symptom–disease data) "
        "as your PRIMARY evidence to answer the question. If the context is incomplete, you may carefully supplement "
        "with general medical knowledge, and you MUST say when you are going beyond the provided context.\n\n"
        "Keep each section concise; avoid long paragraphs where bullets suffice.\n\n"
        "IMPORTANT: You MUST write a COMPLETE answer. Do not stop mid-sentence. Write every section fully. "
        "If no rare diseases appear in the context, say 'None identified in the provided context' in that section.\n\n"
        "Respond in **Markdown** using EXACTLY these sections:\n"
        "1. **Summary** – 2–3 sentence overview.\n"
        "2. **Common Differential Diagnosis** – bullet list of common conditions (at least 5 items).\n"
        "3. **Rare Diseases (from Orphanet)** – bullet list; mark items that came from the Orphanet data (at least 3 if present in context).\n"
        "4. **Red-Flag Features** – bullet list of findings that require urgent action (at least 4 items).\n"
        "5. **Suggested Next Steps** – bullet list of investigations / management steps (at least 4 items).\n"
        "6. **References** – list 3–5 key sources as clickable markdown hyperlinks. "
        "Each reference MUST use the format `- [Article Title](https://full-url-from-context)`. "
        "Use ONLY the exact URLs provided in the context (e.g. https://pubmed.ncbi.nlm.nih.gov/..., "
        "https://www.ncbi.nlm.nih.gov/pmc/articles/...). Never invent URLs.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n"
    )


def _build_medassist_prompt_brief(question: str, context: str) -> str:
    """Shorter prompt for brief mode: bullets + references only."""
    return (
        "You are MedAssist.AI, a medical assistant for clinicians.\n\n"
        "Use the context below as your PRIMARY evidence. Give a very concise answer.\n\n"
        "Respond in **Markdown** with:\n"
        "1. **Summary** – 3–5 bullet points (one line each). No long paragraphs.\n"
        "2. **References** – list 3–5 key sources as clickable markdown hyperlinks. "
        "Each reference MUST use the format `- [Article Title](https://full-url-from-context)`. "
        "Use ONLY the exact URLs provided in the context. Never invent URLs.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n"
    )


def _generate_with_gemini(prompt: str, *, max_output_tokens: int = 4096) -> str:
    """Generate answer with Gemini and return normalized text."""
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]

    config = types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.9,
        max_output_tokens=max_output_tokens,
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=contents,
        config=config,
    )

    answer = ""
    try:
        if response.candidates:
            c = response.candidates[0]
            if getattr(c, "content", None) and getattr(c.content, "parts", None):
                for p in c.content.parts:
                    if hasattr(p, "text") and p.text:
                        answer += p.text
    except Exception:
        pass
    if not answer:
        answer = getattr(response, "text", None) or ""
    return answer.strip()


def _is_valid_cortex_answer(answer: str) -> bool:
    """Treat explicit Cortex error wrappers as failures."""
    text = (answer or "").strip()
    if not text:
        return False
    lowered = text.lower()
    return not (
        lowered.startswith("[cortex error:")
        or lowered.startswith("[cortex diagnostic error:")
    )


def _short_evidence_summary(entries: list[dict], *, fallback_mode: str) -> str:
    if not entries:
        return f"No grounded evidence found (mode: {fallback_mode})."
    sources = sorted({(e.get("source") or "unknown") for e in entries})
    return (
        f"Evidence mode: {fallback_mode}. "
        f"Retrieved {len(entries)} items across {len(sources)} source type(s): {', '.join(sources[:5])}."
    )


def _build_differential_evidence_query(encounter: dict, candidates: list[dict]) -> str:
    symptoms = [str(s.get("symptom", "")).strip() for s in (encounter.get("symptoms") or []) if s.get("symptom")]
    top_diseases = [str(c.get("disease_name", "")).strip() for c in (candidates or [])[:5] if c.get("disease_name")]
    history = (encounter.get("history_summary") or "").strip()
    known_conditions = [str(x).strip() for x in (encounter.get("known_conditions") or []) if str(x).strip()]
    medications = [str(x).strip() for x in (encounter.get("medications") or []) if str(x).strip()]
    allergies = [str(x).strip() for x in (encounter.get("allergies") or []) if str(x).strip()]
    parts = []
    if symptoms:
        parts.append("symptoms: " + ", ".join(symptoms[:8]))
    if top_diseases:
        parts.append("top differential: " + ", ".join(top_diseases))
    if history:
        parts.append("history: " + history[:180])
    if known_conditions:
        parts.append("known conditions: " + ", ".join(known_conditions[:8]))
    if medications:
        parts.append("medications: " + ", ".join(medications[:8]))
    if allergies:
        parts.append("allergies: " + ", ".join(allergies[:8]))
    return "; ".join(parts) or "clinical differential evidence"


@app.post("/ask")
def ask(question: Question, user: CurrentUser):
    """
    Answer a clinical question using all literature sources (PubMed, PMC, NCBI Bookshelf, OpenStax)
    and the symptom→disease index (Orphanet). Data is searched in Snowflake in one place.
    """
    t_ask0 = time.perf_counter()
    evidence_entries, fallback_mode = retrieve_evidence_journal_first(question.question, limit=30)
    literature = _format_literature_context(evidence_entries)
    # 2) Add symptom → rare-disease matches when the question looks like symptoms
    symptom_block = fetch_symptom_disease_context(question.question, limit=50)

    context = literature
    if symptom_block:
        context = context + "\n\n" + symptom_block

    prompt = (
        _build_medassist_prompt_brief(question.question, context)
        if question.brief
        else _build_medassist_prompt(question.question, context)
    )

    # Default provider chain: Cortex first, Gemini fallback.
    evidence_sources = _normalize_evidence_entries(evidence_entries[:6], mode=fallback_mode)
    evidence_summary = _short_evidence_summary(evidence_entries, fallback_mode=fallback_mode)
    evidence_quality = _evaluate_evidence_quality(evidence_entries)
    if (fallback_mode == "insufficient") or (not evidence_quality.get("sufficient", False)):
        followups = [
            "Can you add key exam findings and symptom timeline?",
            "Any high-risk comorbidities, active medications, or contraindications?",
            "What immediate red-flag vitals/labs/imaging are available?",
        ]
        log_assessment_metrics(
            run_name=f"ask_{hash_payload({'q': question.question})[:16]}",
            metrics={
                "evidence_quality_score": float(evidence_quality["score"]),
                "evidence_trusted_count": int(evidence_quality["trusted_count"]),
                "evidence_total_count": int(evidence_quality["count"]),
                "evidence_sufficient": bool(evidence_quality["sufficient"]),
                "llm_latency_ms": 0.0,
                "total_latency_ms": round((time.perf_counter() - t_ask0) * 1000, 2),
            },
            tags={"endpoint": "ask", "provider": "none", "fallback_mode": "insufficient"},
        )
        return {
            "answer": (
                "Insufficient high-quality evidence for a reliable answer right now. "
                "Please provide additional clinical details; re-querying will improve grounded output."
            ),
            "provider": "none",
            "fallback_mode": "insufficient",
            "evidence_summary": evidence_summary,
            "evidence_sources": evidence_sources,
            "evidence_quality": evidence_quality,
            "insufficient_evidence": True,
            "follow_up_questions": followups,
        }

    t_llm = time.perf_counter()
    answer_cortex = cortex_complete(prompt)
    llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
    if _is_valid_cortex_answer(answer_cortex):
        log_assessment_metrics(
            run_name=f"ask_{hash_payload({'q': question.question})[:16]}",
            metrics={
                "evidence_quality_score": float(evidence_quality["score"]),
                "evidence_trusted_count": int(evidence_quality["trusted_count"]),
                "evidence_total_count": int(evidence_quality["count"]),
                "llm_latency_ms": float(llm_ms),
                "total_latency_ms": round((time.perf_counter() - t_ask0) * 1000, 2),
            },
            tags={"endpoint": "ask", "provider": "cortex", "fallback_mode": fallback_mode},
        )
        return {
            "answer": answer_cortex,
            "provider": "cortex",
            "fallback_mode": fallback_mode,
            "evidence_summary": evidence_summary,
            "evidence_sources": evidence_sources,
            "evidence_quality": evidence_quality,
            "insufficient_evidence": False,
        }

    t_llm2 = time.perf_counter()
    answer_gemini = _generate_with_gemini(prompt)
    llm_ms_g = round((time.perf_counter() - t_llm2) * 1000, 2)
    log_assessment_metrics(
        run_name=f"ask_{hash_payload({'q': question.question})[:16]}",
        metrics={
            "evidence_quality_score": float(evidence_quality["score"]),
            "evidence_trusted_count": int(evidence_quality["trusted_count"]),
            "evidence_total_count": int(evidence_quality["count"]),
            "llm_latency_ms": float(llm_ms + llm_ms_g),
            "total_latency_ms": round((time.perf_counter() - t_ask0) * 1000, 2),
        },
        tags={"endpoint": "ask", "provider": "gemini", "fallback_mode": fallback_mode},
    )
    return {
        "answer": answer_gemini,
        "provider": "gemini",
        "fallback_from": "cortex",
        "cortex_error": answer_cortex,
        "fallback_mode": fallback_mode,
        "evidence_summary": evidence_summary,
        "evidence_sources": evidence_sources,
        "evidence_quality": evidence_quality,
        "insufficient_evidence": False,
    }


@app.post("/ask-both")
def ask_both(question: Question, user: CurrentUser):
    """
    Return two answers for the same question: one from Gemini (Vertex AI) and one from
    Snowflake Cortex (claude-sonnet-4-6), both using the same hybrid context (RAG + ILIKE).
    """
    evidence_entries, fallback_mode = retrieve_evidence_journal_first(question.question, limit=30)
    literature = _format_literature_context(evidence_entries)
    symptom_block = fetch_symptom_disease_context(question.question, limit=50)
    context = literature + ("\n\n" + symptom_block if symptom_block else "")

    prompt = (
        _build_medassist_prompt_brief(question.question, context)
        if question.brief
        else _build_medassist_prompt(question.question, context)
    )

    # 1) Gemini
    answer_gemini = _generate_with_gemini(prompt)

    # 2) Cortex
    answer_cortex = cortex_complete(prompt)

    return {
        "answer_gemini": answer_gemini,
        "answer_cortex": answer_cortex,
        "fallback_mode": fallback_mode,
        "evidence_summary": _short_evidence_summary(evidence_entries, fallback_mode=fallback_mode),
        "evidence_sources": _normalize_evidence_entries(evidence_entries[:6], mode=fallback_mode),
        "evidence_quality": _evaluate_evidence_quality(evidence_entries),
    }


@app.post("/ask-cortex")
def ask_cortex(question: Question, user: CurrentUser):
    """
    MedRAG-style endpoint backed entirely by Snowflake Cortex.

    Retrieval and reasoning (including follow-up questions) happen in Snowflake;
    this endpoint only returns the final Cortex answer text.
    """
    answer = cortex_diagnostic_answer(question.question)
    return {"answer": answer}


def _summarize_encounter_context(encounter: dict) -> str:
    encounter = normalize_encounter_dict(encounter)
    symptoms = encounter.get("symptoms", [])
    symptom_lines = []
    for s in symptoms:
        symptom_lines.append(
            f"- {s.get('symptom')} (onset={s.get('onset') or 'unknown'}, severity={s.get('severity') or 'unknown'}, duration={s.get('duration') or 'unknown'})"
        )
    qa = encounter.get("qa_history", [])
    qa_lines = [f"- Q{item['turn_no']}: {item['question']} | A: {item.get('answer') or 'pending'}" for item in qa]

    return (
        f"Patient profile:\n"
        f"- Age: {encounter.get('age')}\n"
        f"- Sex: {encounter.get('sex')}\n"
        f"- Known conditions: {', '.join(encounter.get('known_conditions') or []) or 'None'}\n"
        f"- Medications: {', '.join(encounter.get('medications') or []) or 'None'}\n"
        f"- Allergies: {', '.join(encounter.get('allergies') or []) or 'None'}\n"
        f"- History summary: {encounter.get('history_summary') or 'None'}\n\n"
        f"Symptoms:\n{chr(10).join(symptom_lines) if symptom_lines else '- None'}\n\n"
        f"Prior follow-up QA:\n{chr(10).join(qa_lines) if qa_lines else '- None'}"
    )


def _initial_assessment_prompt(encounter: dict, candidates: list[dict]) -> str:
    context = _summarize_encounter_context(encounter)
    cand_lines = [
        f"- {c['disease_name']} ({c.get('disease_code') or 'no-code'}): {c.get('rationale')}"
        for c in candidates
    ]
    return (
        "You are MedAssist.AI supporting a clinician.\n\n"
        "Use the patient context and knowledge-graph ranked candidates. Be precise and conservative.\n\n"
        "Return Markdown sections:\n"
        "1. Summary\n"
        "2. Prioritized Differential (top 5)\n"
        "3. Why these diagnoses fit\n"
        "4. Red Flags requiring urgent action\n"
        "5. Next tests / management steps\n"
        "6. Follow-up Questions (3-5)\n\n"
        f"{context}\n\n"
        "Knowledge graph ranked candidates:\n"
        f"{chr(10).join(cand_lines) if cand_lines else '- None'}"
    )


def _encounter_narrative_for_contract(encounter: dict, *, latest_answer: str | None = None) -> str:
    """Full intake narrative for structured assessment (not symptoms-only)."""
    sy_lines = []
    for s in encounter.get("symptoms") or []:
        sy_lines.append(
            f"{s.get('symptom') or '?'} "
            f"(onset={s.get('onset') or '—'}, severity={s.get('severity') or '—'}, duration={s.get('duration') or '—'})"
        )
    symptoms_block = "; ".join(sy_lines) if sy_lines else "None recorded"
    kc = ", ".join(encounter.get("known_conditions") or []) or "None"
    meds = ", ".join(encounter.get("medications") or []) or "None"
    alle = ", ".join(encounter.get("allergies") or []) or "None"
    parts = [
        f"Age {encounter.get('age')}, {encounter.get('sex') or 'sex unknown'}.",
        f"Known conditions: {kc}.",
        f"Medications: {meds}.",
        f"Allergies: {alle}.",
        f"History / exam summary: {encounter.get('history_summary') or 'None'}.",
        f"Symptoms (with attributes): {symptoms_block}.",
    ]
    if latest_answer:
        parts.append(f"Latest clinician follow-up response: {latest_answer}")
    return " ".join(parts)


def _chart_context_line(encounter: dict, max_len: int = 220) -> str:
    """Short line for follow-up questions: meds + conditions + history snippet."""
    bits: list[str] = []
    m = encounter.get("medications") or []
    if m:
        bits.append("Meds: " + ", ".join(str(x) for x in m[:6]))
    k = encounter.get("known_conditions") or []
    if k:
        bits.append("Conditions: " + ", ".join(str(x) for x in k[:5]))
    a = encounter.get("allergies") or []
    if a:
        bits.append("Allergies: " + ", ".join(str(x) for x in a[:4]))
    h = (encounter.get("history_summary") or "").strip()
    if h:
        bits.append("History: " + h[: max(0, max_len - sum(len(x) for x in bits) - 20)])
    s = " | ".join(bits)
    return s[:max_len]


def _build_contract_from_candidates(encounter: dict, candidates: list[dict], *, latest_answer: str | None = None) -> dict:
    summary = _encounter_narrative_for_contract(encounter, latest_answer=latest_answer)
    rails = safety_rails(encounter, confidence=0.45 if candidates else 0.2)
    payload = {
        "summary": summary,
        "differential": candidates[:8],
        "red_flags": rails["red_flags"],
        "next_steps": [
            "Cross-check current medications and contraceptives against the differential (e.g. secondary causes, drug effects).",
            "Reassess vitals and red flags after each new symptom update.",
            "Order targeted tests based on top candidates, full history, and medication context.",
            "Continue structured follow-up; incorporate conditions, allergies, and social history into decisions.",
        ],
        "follow_up_questions": [],
        "confidence": 0.65 if candidates else 0.25,
        "insufficient_evidence": len(candidates) == 0,
        "uncertainty_note": rails["uncertainty_note"],
        "contraindications": rails["contraindications"],
    }
    return normalize_contract(payload)


def _attempt_rank_with_degradation(
    encounter_id: str,
    encounter: dict | None = None,
    limit: int = 12,
) -> tuple[list[dict], str, list[str]]:
    """Combine KG ranking, symptom-map matching, and history/medication token hits.

    KG stays primary when available; fallback/context results are additive.
    """
    errors: list[str] = []
    enc = encounter or get_encounter(encounter_id)
    ranked: list[dict] = []
    kg_ranked: list[dict] = []
    degraded = "none"
    # region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H5",
        location="api/main.py:_attempt_rank_with_degradation:entry",
        message="rank_attempt_start",
        data={
            "encounter_id": encounter_id,
            "limit": int(limit),
            "encounter_present": bool(enc),
        },
    )
    # endregion
    try:
        kg_ranked = rank_diseases_from_graph(encounter_id, limit=limit)
        ranked = list(kg_ranked)
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H5",
            location="api/main.py:_attempt_rank_with_degradation:kg_success",
            message="kg_rank_success",
            data={"encounter_id": encounter_id, "kg_count": len(kg_ranked)},
        )
        # endregion
    except Exception as exc:
        errors.append(f"kg_unavailable:{exc}")
        degraded = "symptom_map_only"
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H5",
            location="api/main.py:_attempt_rank_with_degradation:kg_error",
            message="kg_rank_failed",
            data={"encounter_id": encounter_id, "error": str(exc)},
        )
        # endregion
    if not ranked:
        try:
            ranked = rank_diseases_from_symptom_map(encounter_id, limit=limit)
            # region agent log
            _debug_log(
                run_id="pre-fix",
                hypothesis_id="H5",
                location="api/main.py:_attempt_rank_with_degradation:symptom_map_success",
                message="symptom_map_rank_success",
                data={"encounter_id": encounter_id, "count": len(ranked)},
            )
            # endregion
        except Exception as exc2:
            errors.append(f"symptom_map_unavailable:{exc2}")
            # region agent log
            _debug_log(
                run_id="pre-fix",
                hypothesis_id="H5",
                location="api/main.py:_attempt_rank_with_degradation:symptom_map_error",
                message="symptom_map_rank_failed",
                data={"encounter_id": encounter_id, "error": str(exc2)},
            )
            # endregion
    if enc:
        try:
            ctx_rank = rank_diseases_from_context_tokens(enc, limit=min(12, limit))
            ranked = merge_candidate_rankings(ranked, ctx_rank, limit=limit)
        except Exception as exc3:
            errors.append(f"context_rank_unavailable:{exc3}")
    # Keep KG-origin candidates first so the displayed differential remains KG-led.
    if kg_ranked:
        kg_keys = {
            (
                str(x.get("disease_name") or "").strip().lower(),
                str(x.get("disease_code") or "").strip().lower(),
            )
            for x in kg_ranked
        }
        kg_first = list(kg_ranked)
        for row in ranked:
            key = (
                str(row.get("disease_name") or "").strip().lower(),
                str(row.get("disease_code") or "").strip().lower(),
            )
            if key not in kg_keys:
                kg_first.append(row)
        ranked = kg_first[:limit]
    if not ranked:
        degraded = "no_ranker"
    elif degraded == "none" and errors:
        pass
    # Order by estimated clinical probability so follow-up answers can move diagnoses up/down.
    ranked.sort(
        key=lambda x: (
            -float(x.get("confidence_score") if x.get("confidence_score") is not None else 0.0),
            -float(x.get("score") or 0.0),
        )
    )
    # region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H5",
        location="api/main.py:_attempt_rank_with_degradation:exit",
        message="rank_attempt_end",
        data={
            "encounter_id": encounter_id,
            "degraded_mode": degraded,
            "errors_count": len(errors),
            "ranked_count": len(ranked),
            "top3": [
                {
                    "disease_name": c.get("disease_name"),
                    "score": c.get("score"),
                    "confidence_score": c.get("confidence_score"),
                    "source": c.get("source"),
                }
                for c in ranked[:3]
            ],
        },
    )
    # endregion
    return ranked, degraded, errors


def _render_data_only_fallback(encounter: dict, candidates: list[dict], latest_answer: str | None = None) -> str:
    contract = _build_contract_from_candidates(encounter, candidates, latest_answer=latest_answer)
    contract["summary"] = (
        contract["summary"]
        + " Generated in data-only mode because deep reasoning provider was unavailable."
    )
    return contract_to_markdown(contract)


def _enforce_response_contract_markdown(answer: str, encounter: dict, candidates: list[dict]) -> tuple[str, float, bool]:
    """Enforce minimal response contract for deep path textual responses."""
    lowered = (answer or "").lower()
    required_markers = ["summary", "differential", "red", "next", "follow-up"]
    ok = all(m in lowered for m in required_markers)
    insufficient = "insufficient" in lowered or "limited" in lowered
    confidence = 0.6 if ok else 0.35
    if ok:
        return answer, confidence, insufficient
    fallback = _render_data_only_fallback(encounter, candidates)
    return fallback, 0.3, True


def _next_question_prompt(encounter: dict) -> str:
    context = _summarize_encounter_context(encounter)
    return (
        "You are generating exactly ONE next-best clinical follow-up question to narrow differential diagnosis.\n"
        "Prioritize high information gain and safety.\n\n"
        "Respond as plain text with a single question only.\n\n"
        f"{context}"
    )


def _heuristic_next_question(encounter: dict) -> str:
    """Fast deterministic fallback to avoid long-latency model calls."""
    symptoms = [str(s.get("symptom", "")).lower() for s in encounter.get("symptoms", [])]
    asked = " ".join((q.get("question") or "").lower() for q in encounter.get("qa_history", []))
    candidates = [
        ("fever", "What is the maximum recorded temperature, and has there been any rigors/chills?"),
        ("headache", "Is there neck stiffness, photophobia, or confusion with the headache?"),
        ("vomiting", "Is vomiting projectile or bilious, and can the patient keep fluids down?"),
        ("chest pain", "Is the chest pain exertional, pleuritic, or reproducible on palpation?"),
        ("shortness of breath", "What are the oxygen saturation and respiratory rate at rest?"),
    ]
    for trigger, question in candidates:
        if trigger in symptoms and question.lower() not in asked:
            return question
    return "What symptom started first, and what has changed in severity over the last 24 hours?"


@app.post("/encounters/start")
def start_encounter(
    payload: EncounterStartRequest,
    user: CurrentUser,
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
        description="Optional. Reuse with identical JSON body to replay the same response; different body returns 409.",
    ),
):
    ensure_clinical_tables()
    seed_graph_from_symptom_map()
    req_payload = payload.model_dump(mode="json")
    req_hash = hash_payload(req_payload)
    if idempotency_key:
        previous, conflict = fetch_idempotent_response("/encounters/start", idempotency_key, req_hash)
        if conflict:
            raise HTTPException(status_code=409, detail="Idempotency key reused with different payload")
        if previous:
            return previous
    inp = EncounterInput(
        age=payload.age,
        sex=payload.sex,
        known_conditions=payload.known_conditions,
        medications=payload.medications,
        allergies=payload.allergies,
        history_summary=payload.history_summary,
        symptoms=[s.model_dump() for s in payload.symptoms],
        org_id=user.org_id,
        created_by_user_id=user.sub,
    )
    encounter_id = create_encounter_tx(inp)
    response = {"encounter_id": encounter_id}
    if idempotency_key:
        store_idempotent_response("/encounters/start", idempotency_key, req_hash, response)
    return response


def _execute_assess_fast_body(encounter_id: str, *, actor_user_id: str | None = None) -> dict | None:
    """Core assess-fast pipeline; returns None if encounter missing."""
    ensure_clinical_tables()
    t0 = time.perf_counter()
    encounter = get_encounter(encounter_id)
    if not encounter:
        return None

    candidates, degraded_mode, errors = _attempt_rank_with_degradation(encounter_id, encounter=encounter, limit=12)
    # region agent log
    _perf_debug_064(
        "H2",
        "api/main.py:_execute_assess_fast_body",
        "phase_timing",
        pipeline="assess_fast",
        encounter_id=encounter_id,
        phase="after_rank",
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        extra={"n_candidates": len(candidates)},
    )
    # endregion
    kg_version, kg_build_id, snapshot_ts = get_latest_kg_build_meta()
    save_differential(
        encounter_id,
        iteration=1,
        rows=candidates,
        kg_version=kg_version,
        kg_build_id=kg_build_id,
        source_snapshot_ts=snapshot_ts,
    )

    contract = _build_contract_from_candidates(encounter, candidates)
    valid, missing = validate_contract(contract)
    if not valid:
        errors.extend([f"contract_missing:{x}" for x in missing])
        contract = normalize_contract(contract)

    answer = contract_to_markdown(contract)
    evidence_query = _build_differential_evidence_query(encounter, candidates)
    evidence_entries, fallback_mode = retrieve_evidence_journal_first(evidence_query, limit=12)
    # region agent log
    _perf_debug_064(
        "H1",
        "api/main.py:_execute_assess_fast_body",
        "phase_timing",
        pipeline="assess_fast",
        encounter_id=encounter_id,
        phase="after_evidence_retrieval",
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        extra={"fallback_mode": fallback_mode, "n_evidence": len(evidence_entries or [])},
    )
    # endregion
    evidence_summary = _short_evidence_summary(evidence_entries, fallback_mode=fallback_mode)
    evidence_sources = _normalize_evidence_entries(evidence_entries[:5], mode=fallback_mode)
    pipeline_metrics = _audit_pipeline_metrics_assess(
        candidates=candidates,
        errors=errors,
        evidence_entries=evidence_entries,
        fallback_mode=fallback_mode,
        t0=t0,
    )
    append_audit_row(
        encounter_id=encounter_id,
        turn_no=0,
        action="assess_fast",
        provider="knowledge_graph_context",
        model_name="none",
        degraded_mode=degraded_mode,
        context_text=_summarize_encounter_context(encounter),
        output_text=answer,
        kg_version=kg_version or "v1",
        kg_build_id=kg_build_id or "seed_symptom_map",
        error_flags=errors,
        pipeline_metrics=pipeline_metrics,
        actor_user_id=actor_user_id,
    )
    log_assessment_metrics(
        run_name=f"assess_fast_{encounter_id}",
        metrics=pipeline_metrics,
        tags={"endpoint": "assess_fast", "degraded_mode": degraded_mode or "none"},
    )
    _api_debug_log(
        "H3",
        "api:assess_fast",
        "rank_result",
        {
            "encounter_id": encounter_id,
            "n_encounter_symptoms": len(encounter.get("symptoms") or []),
            "n_candidates": len(candidates),
            "degraded_mode": degraded_mode,
        },
    )

    if _PROM_ASSESS_FAST:
        _PROM_ASSESS_FAST.observe(time.perf_counter() - t0)
    # region agent log
    _perf_debug_064(
        "H1",
        "api/main.py:_execute_assess_fast_body",
        "phase_timing",
        pipeline="assess_fast",
        encounter_id=encounter_id,
        phase="total",
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        extra={"degraded_mode": degraded_mode},
    )
    # endregion

    return {
        "encounter_id": encounter_id,
        "provider_used": "knowledge_graph_context",
        "degraded_mode": degraded_mode,
        "errors": errors,
        "contract": contract,
        "top_candidates": candidates[:8],
        "assessment": answer,
        "kg_version": kg_version or "v1",
        "kg_build_id": kg_build_id or "seed_symptom_map",
        "evidence_summary": evidence_summary,
        "evidence_sources": evidence_sources,
        "fallback_mode": fallback_mode,
    }


@app.post("/encounters/{encounter_id}/assess-fast")
def assess_fast(encounter_id: str, user: CurrentUser):
    _assert_encounter_access(user, encounter_id)
    # region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H6",
        location="api/main.py:assess_fast:entry",
        message="assess_fast_called",
        data={"encounter_id": encounter_id, "actor_user_id_present": bool(user.sub)},
    )
    # endregion
    out = _execute_assess_fast_body(encounter_id, actor_user_id=user.sub)
    if out is None:
        raise HTTPException(status_code=404, detail="Encounter not found")
    return out


@app.post("/encounters/{encounter_id}/assess-agentic")
def assess_agentic(encounter_id: str, user: CurrentUser):
    """Same clinical output as assess-fast, routed through a LangGraph agent wrapper for comparison."""
    _assert_encounter_access(user, encounter_id)
    try:
        exec_fn = partial(_execute_assess_fast_body, actor_user_id=user.sub)
        graph = build_assess_graph(exec_fn)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="LangGraph not available. Install langgraph and langchain-core.",
        ) from None
    final = graph.invoke({"encounter_id": encounter_id})
    payload = final.get("result")
    if payload is None:
        raise HTTPException(status_code=404, detail="Encounter not found")
    trace = final.get("agent_trace") or []
    return {**payload, "agent_trace": trace, "agent_provider": "langgraph"}


@app.post("/encounters/{encounter_id}/assess-deep")
def assess_deep(encounter_id: str, user: CurrentUser):
    _assert_encounter_access(user, encounter_id)
    ensure_clinical_tables()
    t0 = time.perf_counter()
    encounter = get_encounter(encounter_id)
    if not encounter:
        raise HTTPException(status_code=404, detail="Encounter not found")

    candidates, degraded_mode, errors = _attempt_rank_with_degradation(encounter_id, encounter=encounter, limit=12)
    kg_version, kg_build_id, snapshot_ts = get_latest_kg_build_meta()
    save_differential(
        encounter_id,
        iteration=1,
        rows=candidates,
        kg_version=kg_version,
        kg_build_id=kg_build_id,
        source_snapshot_ts=snapshot_ts,
    )

    prompt = _initial_assessment_prompt(encounter, candidates)
    answer = cortex_complete(prompt)
    provider = "cortex"
    model_name = CORTEX_MODEL
    if not _is_valid_cortex_answer(answer):
        errors.append("cortex_unavailable")
        degraded_mode = "no_llm"
        answer = _render_data_only_fallback(encounter, candidates)
        provider = "knowledge_graph_context"
        model_name = "none"
    else:
        allowed = set()
        literature = fetch_all_literature_context_hybrid(prompt, limit=20)
        allowed |= set(re.findall(r"https?://[^\s\)]+", literature))
        validated_answer, violations = validate_grounding(answer, allowed)
        if violations:
            errors.append(f"grounding_violations:{violations}")
        answer, confidence, insufficient = _enforce_response_contract_markdown(validated_answer, encounter, candidates)
        contract = _build_contract_from_candidates(encounter, candidates)
        contract["summary"] = validated_answer.splitlines()[0] if validated_answer else contract["summary"]
        contract["confidence"] = confidence
        contract["insufficient_evidence"] = insufficient
        contract = normalize_contract(contract)
        valid, missing = validate_contract(contract)
        if not valid:
            errors.extend([f"contract_missing:{x}" for x in missing])
    evidence_query = _build_differential_evidence_query(encounter, candidates)
    evidence_entries_pre, fallback_mode_pre = retrieve_evidence_journal_first(evidence_query, limit=12)
    pipeline_metrics = _audit_pipeline_metrics_assess(
        candidates=candidates,
        errors=errors,
        evidence_entries=evidence_entries_pre,
        fallback_mode=fallback_mode_pre,
        t0=t0,
    )
    append_audit_row(
        encounter_id=encounter_id,
        turn_no=0,
        action="assess_deep",
        provider=provider,
        model_name=model_name,
        degraded_mode=degraded_mode,
        context_text=_summarize_encounter_context(encounter),
        output_text=answer,
        kg_version=kg_version or "v1",
        kg_build_id=kg_build_id or "seed_symptom_map",
        error_flags=errors,
        pipeline_metrics=pipeline_metrics,
        actor_user_id=user.sub,
    )
    log_assessment_metrics(
        run_name=f"assess_deep_{encounter_id}",
        metrics=pipeline_metrics,
        tags={"endpoint": "assess_deep", "degraded_mode": degraded_mode or "none", "provider": provider},
    )
    evidence_summary = _short_evidence_summary(evidence_entries_pre, fallback_mode=fallback_mode_pre)
    evidence_sources = _normalize_evidence_entries(evidence_entries_pre[:5], mode=fallback_mode_pre)
    return {
        "encounter_id": encounter_id,
        "provider_used": provider,
        "degraded_mode": degraded_mode,
        "errors": errors,
        "top_candidates": candidates[:8],
        "assessment": answer,
        "kg_version": kg_version or "v1",
        "kg_build_id": kg_build_id or "seed_symptom_map",
        "evidence_summary": evidence_summary,
        "evidence_sources": evidence_sources,
        "fallback_mode": fallback_mode_pre,
    }


@app.post("/encounters/{encounter_id}/initial-assessment")
def initial_assessment(encounter_id: str, user: CurrentUser):
    # Compatibility wrapper
    if CLINICAL_USE_CORTEX:
        return assess_deep(encounter_id, user)
    return assess_fast(encounter_id, user)


def _top_differential_snapshot(encounter: dict, limit: int = 3) -> list[dict]:
    diff = encounter.get("differential") or []
    if not diff:
        return []
    iterations = [int(d.get("iteration") or 0) for d in diff]
    mx = max(iterations)
    rows = [d for d in diff if int(d.get("iteration") or 0) == mx]
    rows.sort(key=lambda x: int(x.get("rank_no") or 999))
    out: list[dict] = []
    for d in rows[:limit]:
        out.append(
            {
                "disease_name": d.get("disease_name"),
                "disease_code": d.get("disease_code"),
                "score": d.get("score"),
            }
        )
    return out


def _followup_question_discriminative(encounter: dict, proposed_question: str) -> bool:
    """Return True if the proposed follow-up is judged discriminative (fail-open on LLM errors)."""
    if not (proposed_question or "").strip():
        return True
    top = _top_differential_snapshot(encounter, 3)
    payload = json.dumps({"top_differential": top, "proposed_follow_up": proposed_question}, ensure_ascii=False)
    prompt = (
        "You are a clinical informatics reviewer. Given the top differential (JSON) and one proposed "
        "follow-up question for the clinician, decide if the question is clinically discriminative "
        "(helps narrow causes or distinguishes between listed differentials).\n"
        "Reply with exactly one line starting with YES or NO, then a brief reason.\n\n"
        f"{payload}"
    )
    try:
        text = cortex_complete(prompt)
        if not _is_valid_cortex_answer(text):
            text = _generate_with_gemini(prompt, max_output_tokens=128)
        line = (text or "").strip().splitlines()[0].upper() if (text or "").strip() else ""
        return bool(line.startswith("YES"))
    except Exception:
        return True


@app.post("/encounters/{encounter_id}/next-question")
def next_question(encounter_id: str, user: CurrentUser):
    _assert_encounter_access(user, encounter_id)
    ensure_clinical_tables()
    encounter = get_encounter(encounter_id)
    if not encounter:
        raise HTTPException(status_code=404, detail="Encounter not found")

    turn_no = next_turn_no(encounter_id)
    question_text, policy_reason, answer_choices = policy_next_question(encounter, max_turns=CLINICAL_MAX_TURNS)
    if not question_text:
        question_text = "What is the symptom timeline and which symptom started first?"
        answer_choices = None
    if FOLLOWUP_SELF_CRITIQUE and not _followup_question_discriminative(encounter, question_text):
        question_text, policy_reason, answer_choices = policy_next_question(
            encounter,
            max_turns=CLINICAL_MAX_TURNS,
            skip_questions=(question_text,),
        )
    add_question(encounter_id, turn_no, question_text)
    return {
        "encounter_id": encounter_id,
        "turn_no": turn_no,
        "question": question_text,
        "policy_reason": policy_reason,
        "answer_choices": list(answer_choices) if answer_choices else None,
    }


@app.post("/encounters/{encounter_id}/answer")
def answer_question(
    encounter_id: str,
    payload: EncounterAnswerRequest,
    user: CurrentUser,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    _assert_encounter_access(user, encounter_id)
    # region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H1",
        location="api/main.py:answer_question:entry",
        message="answer_request_received",
        data={
            "encounter_id": encounter_id,
            "turn_no": int(payload.turn_no),
            "answer_len": len((payload.answer or "").strip()),
        },
    )
    # endregion
    ensure_clinical_tables()
    t0 = time.perf_counter()
    encounter = get_encounter(encounter_id)
    if not encounter:
        raise HTTPException(status_code=404, detail="Encounter not found")

    req_payload = {"encounter_id": encounter_id, **payload.model_dump(mode="json")}
    req_hash = hash_payload(req_payload)
    if idempotency_key:
        previous, conflict = fetch_idempotent_response("/encounters/answer", idempotency_key, req_hash)
        if conflict:
            raise HTTPException(status_code=409, detail="Idempotency key reused with different payload")
        if previous:
            return previous

    # Persist answer first so get_encounter includes it in qa_history for token extraction.
    add_answer(encounter_id, payload.turn_no, payload.answer)
    ingest_followup_answer_tokens(encounter_id, payload.answer)
    encounter = get_encounter(encounter_id)
    if not encounter:
        raise HTTPException(status_code=404, detail="Encounter not found")

    previous_candidates = encounter.get("differential", [])[:12]
    # region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H3",
        location="api/main.py:answer_question:before_rank",
        message="previous_candidates_snapshot",
        data={
            "encounter_id": encounter_id,
            "previous_count": len(previous_candidates),
            "previous_top3": [
                {
                    "disease_name": c.get("disease_name"),
                    "score": c.get("score"),
                    "confidence_score": c.get("confidence_score"),
                }
                for c in previous_candidates[:3]
            ],
        },
    )
    # endregion
    candidates, degraded_mode, errors = _attempt_rank_with_degradation(
        encounter_id, encounter=encounter, limit=12
    )
    # region agent log
    _perf_debug_064(
        "H2",
        "api/main.py:answer_question",
        "phase_timing",
        pipeline="answer",
        encounter_id=encounter_id,
        phase="after_rank",
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        extra={"turn_no": int(payload.turn_no), "n_candidates": len(candidates)},
    )
    # endregion
    # region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H3",
        location="api/main.py:answer_question:after_rank",
        message="new_candidates_snapshot",
        data={
            "encounter_id": encounter_id,
            "new_count": len(candidates),
            "degraded_mode": degraded_mode,
            "errors_count": len(errors or []),
            "new_top3": [
                {
                    "disease_name": c.get("disease_name"),
                    "score": c.get("score"),
                    "confidence_score": c.get("confidence_score"),
                }
                for c in candidates[:3]
            ],
        },
    )
    # endregion
    t_kg = time.perf_counter()
    kg_version, kg_build_id, snapshot_ts = get_latest_kg_build_meta()
    # region agent log
    _perf_debug_064(
        "H4",
        "api/main.py:answer_question",
        "phase_timing",
        pipeline="answer",
        encounter_id=encounter_id,
        phase="kg_meta_only",
        elapsed_ms=(time.perf_counter() - t_kg) * 1000,
        extra={"turn_no": int(payload.turn_no)},
    )
    # endregion
    t_persist = time.perf_counter()
    answer_and_update_tx(
        encounter_id=encounter_id,
        turn_no=payload.turn_no,
        answer_text=payload.answer,
        candidates=candidates,
        kg_version=kg_version,
        kg_build_id=kg_build_id,
        source_snapshot_ts=snapshot_ts,
        diff_previous=previous_candidates,
    )
    # region agent log
    _perf_debug_064(
        "H3",
        "api/main.py:answer_question",
        "phase_timing",
        pipeline="answer",
        encounter_id=encounter_id,
        phase="persist_block_tx_and_delta",
        elapsed_ms=(time.perf_counter() - t_persist) * 1000,
        extra={"turn_no": int(payload.turn_no)},
    )
    # endregion
    contract = _build_contract_from_candidates(encounter, candidates, latest_answer=payload.answer)
    answer = contract_to_markdown(contract)
    provider = "knowledge_graph_context"
    # region agent log
    _perf_debug_064(
        "H3",
        "api/main.py:answer_question",
        "phase_timing",
        pipeline="answer",
        encounter_id=encounter_id,
        phase="after_persist_and_contract",
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        extra={"turn_no": int(payload.turn_no)},
    )
    # endregion
    evidence_query = _build_differential_evidence_query(encounter, candidates)
    evidence_entries_ans, fallback_mode_ans = retrieve_evidence_journal_first(evidence_query, limit=12)
    # region agent log
    _perf_debug_064(
        "H1",
        "api/main.py:answer_question",
        "phase_timing",
        pipeline="answer",
        encounter_id=encounter_id,
        phase="after_evidence_retrieval",
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        extra={
            "turn_no": int(payload.turn_no),
            "fallback_mode": fallback_mode_ans,
            "n_evidence": len(evidence_entries_ans or []),
        },
    )
    # endregion
    pipeline_metrics = _audit_pipeline_metrics_assess(
        candidates=candidates,
        errors=errors,
        evidence_entries=evidence_entries_ans,
        fallback_mode=fallback_mode_ans,
        t0=t0,
    )
    pipeline_metrics["turn_no"] = int(payload.turn_no)
    append_audit_row(
        encounter_id=encounter_id,
        turn_no=payload.turn_no,
        action="answer",
        provider=provider,
        model_name="none",
        degraded_mode=degraded_mode,
        context_text=_summarize_encounter_context(encounter),
        output_text=answer,
        kg_version=kg_version or "v1",
        kg_build_id=kg_build_id or "seed_symptom_map",
        error_flags=errors,
        pipeline_metrics=pipeline_metrics,
        actor_user_id=user.sub,
    )
    log_assessment_metrics(
        run_name=f"answer_{encounter_id}_t{payload.turn_no}",
        metrics=pipeline_metrics,
        tags={"endpoint": "answer", "degraded_mode": degraded_mode or "none"},
    )

    response = {
        "encounter_id": encounter_id,
        "provider_used": provider,
        "degraded_mode": degraded_mode,
        "errors": errors,
        "contract": contract,
        "top_candidates": candidates[:8],
        "assessment": answer,
        "kg_version": kg_version or "v1",
        "kg_build_id": kg_build_id or "seed_symptom_map",
    }
    response["evidence_summary"] = _short_evidence_summary(evidence_entries_ans, fallback_mode=fallback_mode_ans)
    response["evidence_sources"] = _normalize_evidence_entries(evidence_entries_ans[:5], mode=fallback_mode_ans)
    response["fallback_mode"] = fallback_mode_ans
    # region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H4",
        location="api/main.py:answer_question:response",
        message="answer_response_payload",
        data={
            "encounter_id": encounter_id,
            "response_top_count": len(response.get("top_candidates") or []),
            "response_top3": [
                {
                    "disease_name": c.get("disease_name"),
                    "score": c.get("score"),
                    "confidence_score": c.get("confidence_score"),
                }
                for c in (response.get("top_candidates") or [])[:3]
            ],
        },
    )
    # endregion
    if idempotency_key:
        store_idempotent_response("/encounters/answer", idempotency_key, req_hash, response)
    # region agent log
    _perf_debug_064(
        "H1",
        "api/main.py:answer_question",
        "phase_timing",
        pipeline="answer",
        encounter_id=encounter_id,
        phase="total",
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        extra={"turn_no": int(payload.turn_no), "degraded_mode": degraded_mode},
    )
    # endregion
    return response


@app.get("/encounters/{encounter_id}/context")
def encounter_context(encounter_id: str, user: CurrentUser):
    _assert_encounter_access(user, encounter_id)
    ensure_clinical_tables()
    encounter = get_encounter(encounter_id)
    if not encounter:
        raise HTTPException(status_code=404, detail="Encounter not found")
    # region agent log
    _api_debug_log(
        "H2",
        "api:encounter_context",
        "structured_shape",
        {
            "encounter_id": encounter_id,
            "n_symptoms": len(encounter.get("symptoms") or []),
            "n_conditions": len(encounter.get("known_conditions") or []),
        },
    )
    # endregion
    return {
        "encounter_id": encounter_id,
        "context_text": _summarize_encounter_context(encounter),
        "structured_context": encounter,
    }


@app.get("/encounters/{encounter_id}/kg-preview")
def encounter_kg_preview(
    encounter_id: str,
    user: CurrentUser,
    max_edges: int = Query(2500, ge=1, le=5000, description="Safety cap on rows; all matching diseases included up to this limit"),
):
    """All HAS_SYMPTOM disease–symptom pairs for this encounter’s symptoms (bounded by max_edges)."""
    _assert_encounter_access(user, encounter_id)
    ensure_clinical_tables()
    encounter = get_encounter(encounter_id)
    if not encounter:
        raise HTTPException(status_code=404, detail="Encounter not found")
    # region agent log
    _api_debug_log(
        "H4",
        "api:kg_preview",
        "before_kg_query",
        {
            "encounter_id": encounter_id,
            "n_encounter_symptoms": len(encounter.get("symptoms") or []),
        },
    )
    # endregion
    return get_encounter_kg_preview(encounter_id, max_edges=max_edges)


def _latest_differential_as_candidates(encounter: dict) -> list[dict]:
    diff = encounter.get("differential") or []
    if not diff:
        return []
    iterations = [int(d.get("iteration") or 0) for d in diff]
    max_it = max(iterations)
    rows = [d for d in diff if int(d.get("iteration") or 0) == max_it]
    rows.sort(key=lambda x: int(x.get("rank_no") or 999))
    out: list[dict] = []
    for d in rows:
        out.append(
            {
                "disease_name": d.get("disease_name"),
                "disease_code": d.get("disease_code"),
                "score": d.get("score"),
                "rationale": d.get("rationale"),
                "source": d.get("source"),
            }
        )
    return out


@app.get("/encounters/{encounter_id}/report.pdf")
def encounter_report_pdf(encounter_id: str, user: CurrentUser):
    """Download a PDF summary for the encounter (doctor: own; admin: org)."""
    enc = _assert_encounter_access(user, encounter_id)
    cands = _latest_differential_as_candidates(enc)
    kg_version, kg_build_id, _snap = get_latest_kg_build_meta()
    evidence_summary = ""
    fallback_mode: str | None = None
    if cands:
        eq = _build_differential_evidence_query(enc, cands)
        ev, fallback_mode = retrieve_evidence_journal_first(eq, limit=8)
        evidence_summary = _short_evidence_summary(ev, fallback_mode=fallback_mode or "journal_first")
    contract = _build_contract_from_candidates(enc, cands[:12] if cands else [])
    assessment_markdown = contract_to_markdown(contract)
    pdf_bytes = build_encounter_pdf(
        encounter_id=encounter_id,
        encounter=enc,
        assessment_markdown=assessment_markdown,
        top_candidates=cands[:8] if cands else [],
        evidence_summary=evidence_summary or None,
        kg_version=kg_version,
        kg_build_id=kg_build_id,
        fallback_mode=fallback_mode,
        org_id=enc.get("org_id"),
    )
    append_audit_row(
        encounter_id=encounter_id,
        turn_no=0,
        action="export_pdf",
        provider="pdf",
        model_name="none",
        degraded_mode="none",
        context_text="",
        output_text="pdf_generated",
        kg_version=kg_version or "v1",
        kg_build_id=kg_build_id or "seed_symptom_map",
        error_flags=[],
        pipeline_metrics={"bytes": len(pdf_bytes)},
        actor_user_id=user.sub,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="medassist-encounter-{encounter_id}.pdf"'},
    )


# ---- Option B: Answer from GCS documents via Vertex AI Search ----
# Requires a Vertex AI Search data store created from gs://medassist-data-gcs/medassist/
# See OPTION_B_SETUP.md. Set VERTEX_AI_DATASTORE_PATH or VERTEX_AI_DATASTORE_ID.


@app.post("/ask-gcs")
def ask_gcs(question: Question, user: CurrentUser):
    """
    Answer a clinical question using only documents in your GCS bucket, via Vertex AI Search
    grounding. No Snowflake. Requires a data store created in AI Applications (see OPTION_B_SETUP.md).
    """
    if not VERTEX_AI_DATASTORE_PATH:
        raise HTTPException(
            status_code=503,
            detail=(
                "Vertex AI Search data store not configured. Create a data store from "
                "gs://medassist-data-gcs/medassist/ in AI Applications, then set "
                "VERTEX_AI_DATASTORE_PATH or VERTEX_AI_DATASTORE_ID. See OPTION_B_SETUP.md."
            ),
        )

    # Use a model that supports grounding with Vertex AI Search (e.g. gemini-2.5-flash or gemini-3-flash)
    model_for_grounding = "gemini-2.5-flash"
    try:
        vertex_ai_search = getattr(types, "VertexAISearch", None)
        retrieval_cls = getattr(types, "Retrieval", None)
        tool_cls = getattr(types, "Tool", None)
        if not all([vertex_ai_search, retrieval_cls, tool_cls]):
            raise HTTPException(
                status_code=501,
                detail="This endpoint requires google-genai with Vertex AI Search support (Retrieval, VertexAISearch, Tool). Upgrade: pip install -U google-genai",
            )
        tool = tool_cls(
            retrieval=retrieval_cls(
                vertex_ai_search=vertex_ai_search(datastore=VERTEX_AI_DATASTORE_PATH)
            )
        )
    except TypeError as e:
        raise HTTPException(
            status_code=501,
            detail=f"Vertex AI Search tool construction failed: {e}. Ensure google-genai is up to date.",
        ) from e

    prompt = (
        "You are MedAssist.AI, a medical assistant for clinicians. "
        "Answer the following question using ONLY the provided document context from the retrieval. "
        "If the context is insufficient, say so and do not invent facts. "
        "Respond in clear Markdown with a brief summary, key points, and any relevant caveats or next steps.\n\n"
        f"Question: {question.question}"
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]

    config = types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=4096,
        tools=[tool],
    )

    response = client.models.generate_content(
        model=model_for_grounding,
        contents=contents,
        config=config,
    )

    answer = ""
    try:
        if response.candidates:
            c = response.candidates[0]
            if getattr(c, "content", None) and getattr(c.content, "parts", None):
                for p in c.content.parts:
                    if hasattr(p, "text") and p.text:
                        answer += p.text
    except Exception:
        pass
    if not answer:
        answer = getattr(response, "text", None) or ""
    answer = answer.strip()
    return {"answer": answer}

