from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from pathlib import Path
import json

from google import genai
from google.genai import types
import re

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
    fetch_idempotent_response,
    get_encounter,
    get_latest_kg_build_meta,
    hash_payload,
    next_turn_no,
    rank_diseases_from_symptom_map,
    rank_diseases_from_graph,
    save_differential,
    save_diff_delta,
    seed_graph_from_symptom_map,
    store_idempotent_response,
)
from src.followup_policy import MAX_TURNS_DEFAULT, next_question as policy_next_question, safety_rails
from src.grounding import extract_urls_from_context, validate_grounding
from src.response_contract import contract_to_markdown, normalize_contract, validate_contract

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

app = FastAPI(title="MedAssist.AI API")

# Frontend: serve static UI at /
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.get("/")
def index():
    """Serve the MedAssist.AI web UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Static frontend not found")
    return FileResponse(index_path)


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


def fetch_all_literature_context_hybrid(question: str, limit: int = 30) -> str:
    """
    Hybrid retrieval: try RAG (ChromaDB) first, fall back to ILIKE (Snowflake) if few results.
    Merge and deduplicate, then return formatted context for the LLM.
    """
    RAG_MIN_RESULTS = 5  # If RAG returns fewer, use ILIKE too

    rag_entries = _fetch_rag_entries(question, limit=15)
    ilike_entries: list[dict] = []
    if len(rag_entries) < RAG_MIN_RESULTS:
        ilike_entries = _fetch_ilike_entries(question, limit=limit)

    merged = _merge_and_dedupe_literature(rag_entries, ilike_entries)
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


# Snowflake Cortex model for /ask-both
CORTEX_MODEL = "claude-sonnet-4-6"


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


def _generate_with_gemini(prompt: str) -> str:
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
        max_output_tokens=4096,
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


@app.post("/ask")
def ask(question: Question):
    """
    Answer a clinical question using all literature sources (PubMed, PMC, NCBI Bookshelf, OpenStax)
    and the symptom→disease index (Orphanet). Data is searched in Snowflake in one place.
    """
    # 1) Hybrid: RAG (ChromaDB) first, ILIKE (Snowflake) fallback when few results; merge & dedupe
    literature = fetch_all_literature_context_hybrid(question.question, limit=30)
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
    answer_cortex = cortex_complete(prompt)
    if _is_valid_cortex_answer(answer_cortex):
        return {"answer": answer_cortex, "provider": "cortex"}

    answer_gemini = _generate_with_gemini(prompt)
    return {
        "answer": answer_gemini,
        "provider": "gemini",
        "fallback_from": "cortex",
        "cortex_error": answer_cortex,
    }


@app.post("/ask-both")
def ask_both(question: Question):
    """
    Return two answers for the same question: one from Gemini (Vertex AI) and one from
    Snowflake Cortex (claude-sonnet-4-6), both using the same hybrid context (RAG + ILIKE).
    """
    literature = fetch_all_literature_context_hybrid(question.question, limit=30)
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
    }


@app.post("/ask-cortex")
def ask_cortex(question: Question):
    """
    MedRAG-style endpoint backed entirely by Snowflake Cortex.

    Retrieval and reasoning (including follow-up questions) happen in Snowflake;
    this endpoint only returns the final Cortex answer text.
    """
    answer = cortex_diagnostic_answer(question.question)
    return {"answer": answer}


def _summarize_encounter_context(encounter: dict) -> str:
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


def _build_contract_from_candidates(encounter: dict, candidates: list[dict], *, latest_answer: str | None = None) -> dict:
    symptoms = ", ".join(s.get("symptom", "") for s in encounter.get("symptoms", [])) or "none"
    profile = f"Symptoms currently tracked: {symptoms}"
    if latest_answer:
        profile += f". Latest clinician input: {latest_answer}"
    rails = safety_rails(encounter, confidence=0.45 if candidates else 0.2)
    payload = {
        "summary": profile,
        "differential": candidates[:8],
        "red_flags": rails["red_flags"],
        "next_steps": [
            "Reassess vitals and red flags after each new symptom update.",
            "Order targeted tests based on top candidates and symptom chronology.",
            "Continue structured follow-up questions to reduce uncertainty.",
        ],
        "follow_up_questions": [],
        "confidence": 0.65 if candidates else 0.25,
        "insufficient_evidence": len(candidates) == 0,
        "uncertainty_note": rails["uncertainty_note"],
        "contraindications": rails["contraindications"],
    }
    return normalize_contract(payload)


def _attempt_rank_with_degradation(encounter_id: str, limit: int = 12) -> tuple[list[dict], str, list[str]]:
    errors: list[str] = []
    try:
        ranked = rank_diseases_from_graph(encounter_id, limit=limit)
        return ranked, "none", errors
    except Exception as exc:
        errors.append(f"kg_unavailable:{exc}")
        try:
            ranked = rank_diseases_from_symptom_map(encounter_id, limit=limit)
            return ranked, "symptom_map_only", errors
        except Exception as exc2:
            errors.append(f"symptom_map_unavailable:{exc2}")
            return [], "no_ranker", errors


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
def start_encounter(payload: EncounterStartRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
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
    )
    encounter_id = create_encounter_tx(inp)
    response = {"encounter_id": encounter_id}
    if idempotency_key:
        store_idempotent_response("/encounters/start", idempotency_key, req_hash, response)
    return response


@app.post("/encounters/{encounter_id}/assess-fast")
def assess_fast(encounter_id: str):
    ensure_clinical_tables()
    encounter = get_encounter(encounter_id)
    if not encounter:
        raise HTTPException(status_code=404, detail="Encounter not found")

    candidates, degraded_mode, errors = _attempt_rank_with_degradation(encounter_id, limit=12)
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
    append_audit_row(
        encounter_id=encounter_id,
        turn_no=0,
        action="assess_fast",
        provider="knowledge_graph_context",
        model_name="none",
        degraded_mode=degraded_mode,
        context_text=_summarize_encounter_context(encounter),
        output_text=answer,
        kg_version=kg_version,
        kg_build_id=kg_build_id,
        error_flags=errors,
    )

    return {
        "encounter_id": encounter_id,
        "provider_used": "knowledge_graph_context",
        "degraded_mode": degraded_mode,
        "errors": errors,
        "contract": contract,
        "top_candidates": candidates[:8],
        "assessment": answer,
        "kg_version": kg_version,
        "kg_build_id": kg_build_id,
    }


@app.post("/encounters/{encounter_id}/assess-deep")
def assess_deep(encounter_id: str):
    ensure_clinical_tables()
    encounter = get_encounter(encounter_id)
    if not encounter:
        raise HTTPException(status_code=404, detail="Encounter not found")

    candidates, degraded_mode, errors = _attempt_rank_with_degradation(encounter_id, limit=12)
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
    append_audit_row(
        encounter_id=encounter_id,
        turn_no=0,
        action="assess_deep",
        provider=provider,
        model_name=model_name,
        degraded_mode=degraded_mode,
        context_text=_summarize_encounter_context(encounter),
        output_text=answer,
        kg_version=kg_version,
        kg_build_id=kg_build_id,
        error_flags=errors,
    )
    return {
        "encounter_id": encounter_id,
        "provider_used": provider,
        "degraded_mode": degraded_mode,
        "errors": errors,
        "top_candidates": candidates[:8],
        "assessment": answer,
        "kg_version": kg_version,
        "kg_build_id": kg_build_id,
    }


@app.post("/encounters/{encounter_id}/initial-assessment")
def initial_assessment(encounter_id: str):
    # Compatibility wrapper
    if CLINICAL_USE_CORTEX:
        return assess_deep(encounter_id)
    return assess_fast(encounter_id)


@app.post("/encounters/{encounter_id}/next-question")
def next_question(encounter_id: str):
    ensure_clinical_tables()
    encounter = get_encounter(encounter_id)
    if not encounter:
        raise HTTPException(status_code=404, detail="Encounter not found")

    turn_no = next_turn_no(encounter_id)
    question_text, policy_reason = policy_next_question(encounter, max_turns=CLINICAL_MAX_TURNS)
    if not question_text:
        question_text = "What is the symptom timeline and which symptom started first?"

    add_question(encounter_id, turn_no, question_text)
    return {"encounter_id": encounter_id, "turn_no": turn_no, "question": question_text, "policy_reason": policy_reason}


@app.post("/encounters/{encounter_id}/answer")
def answer_question(
    encounter_id: str,
    payload: EncounterAnswerRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    ensure_clinical_tables()
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

    previous_candidates = encounter.get("differential", [])[:12]
    updated = get_encounter(encounter_id)
    candidates, degraded_mode, errors = _attempt_rank_with_degradation(encounter_id, limit=12)
    kg_version, kg_build_id, snapshot_ts = get_latest_kg_build_meta()
    answer_and_update_tx(
        encounter_id=encounter_id,
        turn_no=payload.turn_no,
        answer_text=payload.answer,
        candidates=candidates,
        kg_version=kg_version,
        kg_build_id=kg_build_id,
        source_snapshot_ts=snapshot_ts,
    )
    save_diff_delta(encounter_id, payload.turn_no, previous_candidates, candidates)
    contract = _build_contract_from_candidates(updated or encounter, candidates, latest_answer=payload.answer)
    answer = contract_to_markdown(contract)
    provider = "knowledge_graph_context"
    append_audit_row(
        encounter_id=encounter_id,
        turn_no=payload.turn_no,
        action="answer",
        provider=provider,
        model_name="none",
        degraded_mode=degraded_mode,
        context_text=_summarize_encounter_context(updated or encounter),
        output_text=answer,
        kg_version=kg_version,
        kg_build_id=kg_build_id,
        error_flags=errors,
    )

    response = {
        "encounter_id": encounter_id,
        "provider_used": provider,
        "degraded_mode": degraded_mode,
        "errors": errors,
        "contract": contract,
        "top_candidates": candidates[:8],
        "assessment": answer,
        "kg_version": kg_version,
        "kg_build_id": kg_build_id,
    }
    if idempotency_key:
        store_idempotent_response("/encounters/answer", idempotency_key, req_hash, response)
    return response


@app.get("/encounters/{encounter_id}/context")
def encounter_context(encounter_id: str):
    ensure_clinical_tables()
    encounter = get_encounter(encounter_id)
    if not encounter:
        raise HTTPException(status_code=404, detail="Encounter not found")
    return {
        "encounter_id": encounter_id,
        "context_text": _summarize_encounter_context(encounter),
        "structured_context": encounter,
    }


# ---- Option B: Answer from GCS documents via Vertex AI Search ----
# Requires a Vertex AI Search data store created from gs://medassist-data-gcs/medassist/
# See OPTION_B_SETUP.md. Set VERTEX_AI_DATASTORE_PATH or VERTEX_AI_DATASTORE_ID.


@app.post("/ask-gcs")
def ask_gcs(question: Question):
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

