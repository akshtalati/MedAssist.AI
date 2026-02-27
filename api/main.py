from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from google import genai
from google.genai import types

from src.snowflake_client import get_connection


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


class Question(BaseModel):
    question: str


def fetch_all_literature_context(question: str, limit: int = 30) -> str:
    """
    Search PubMed, PMC, NCBI Bookshelf, and OpenStax in one go (all sources at once).
    Returns combined context for the LLM.
    """
    conn = get_connection()
    cur = conn.cursor()
    like = f"%{question}%"
    try:
        # Single query: union all four literature sources, then filter by search term
        sql = """
            SELECT source, title, abstract
            FROM (
                SELECT 'pubmed' AS source, title, abstract FROM RAW.PUBMED_ARTICLES
                UNION ALL
                SELECT 'pmc', title, abstract FROM RAW.PMC_ARTICLES
                UNION ALL
                SELECT 'ncbi_bookshelf', title, abstract FROM RAW.NCBI_BOOKSHELF
                UNION ALL
                SELECT 'openstax', title, content AS abstract FROM RAW.OPENSTAX_BOOKS
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
    for idx, (source, title, abstract) in enumerate(rows, start=1):
        text = (abstract or "")[:8000]  # cap length per doc to avoid token overflow
        parts.append(
            f"[{idx}] Source: {source}\nTitle: {title}\nText: {text}\n"
        )
    return "\n---\n\n".join(parts)


def fetch_symptom_disease_context(question: str, limit: int = 50) -> str:
    """
    Look up symptom → disease from NORMALIZED.SYMPTOM_DISEASE_MAP (Orphanet).
    Helps with differential diagnosis (e.g. 'fever vomiting' → rare diseases).
    """
    conn = get_connection()
    cur = conn.cursor()
    words = [w.strip() for w in question.lower().split() if len(w.strip()) > 2]
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


@app.post("/ask")
def ask(question: Question):
    """
    Answer a clinical question using all literature sources (PubMed, PMC, NCBI Bookshelf, OpenStax)
    and the symptom→disease index (Orphanet). Data is searched in Snowflake in one place.
    """
    # 1) Search all literature at once (no dbt required; we union raw tables in SQL)
    literature = fetch_all_literature_context(question.question, limit=30)
    # 2) Add symptom → rare-disease matches when the question looks like symptoms
    symptom_block = fetch_symptom_disease_context(question.question, limit=50)

    context = literature
    if symptom_block:
        context = context + "\n\n" + symptom_block

    prompt = (
        "You are MedAssist.AI, a medical assistant for clinicians.\n\n"
        "Use the context below (from PubMed, PMC, NCBI Bookshelf, OpenStax, and Orphanet symptom–disease data) "
        "as your PRIMARY evidence to answer the question. If the context is incomplete, you may carefully supplement "
        "with general medical knowledge, and you MUST say when you are going beyond the provided context.\n\n"
        "IMPORTANT: You MUST write a COMPLETE answer. Do not stop mid-sentence. Write every section fully.\n\n"
        "Respond in **Markdown** using EXACTLY these sections:\n"
        "1. **Summary** – 2–3 sentence overview.\n"
        "2. **Common Differential Diagnosis** – bullet list of common conditions (at least 5 items).\n"
        "3. **Rare Diseases (from Orphanet)** – bullet list; mark items that came from the Orphanet data (at least 3 if present in context).\n"
        "4. **Red-Flag Features** – bullet list of findings that require urgent action (at least 4 items).\n"
        "5. **Suggested Next Steps** – bullet list of investigations / management steps (at least 4 items).\n\n"
        f"Question:\n{question.question}\n\n"
        f"Context:\n{context}\n"
    )

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

    # Use non-streaming call to ensure we capture the full answer
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=contents,
        config=config,
    )

    # Extract full text: join all parts from the first candidate (avoids truncation from .text)
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

