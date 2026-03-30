from fastapi import FastAPI, HTTPException
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
if SentenceTransformer is not None:
    try:
        _rag_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _rag_embed_model = None


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
    if _rag_embed_model is None:
        return "[Cortex diagnostic error: embedding model unavailable in this environment]"

    # 1) Embed question locally to a 384-dim vector (all-MiniLM-L6-v2)
    q_vec = _rag_embed_model.encode([question])[0]
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

