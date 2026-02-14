"""
RAG (Retrieval Augmented Generation) index builder.

Chunks text from PubMed, OpenStax, NCBI Bookshelf and stores embeddings in ChromaDB
for semantic search. Enables "fever and vomiting" → retrieve relevant medical context.
"""

import json
from pathlib import Path
from typing import Iterator

from ..config import get_data_paths, PROJECT_ROOT


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    if not text or len(text) < 100:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def _load_pubmed_chunks(paths: dict) -> Iterator[tuple[str, dict]]:
    """Yield (text, metadata) from PubMed raw JSON."""
    raw_dir = paths["raw"] / "pubmed"
    if not raw_dir.exists():
        return
    for f in sorted(raw_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
        try:
            with open(f) as fp:
                data = json.load(fp)
            articles = data.get("data", {}).get("articles", [])
            for a in articles:
                title = a.get("title", "")
                abstract = a.get("abstract", "")
                pmid = a.get("pmid", "")
                text = f"{title}\n\n{abstract}".strip()
                for chunk in _chunk_text(text):
                    yield chunk, {"source": "pubmed", "pmid": str(pmid), "title": title[:200]}
        except (json.JSONDecodeError, KeyError):
            continue


def _load_openstax_chunks(paths: dict) -> Iterator[tuple[str, dict]]:
    """Yield (text, metadata) from OpenStax extracted JSON."""
    extracted_dir = paths["raw"] / "openstax" / "extracted"
    if not extracted_dir.exists():
        return
    for f in list(extracted_dir.glob("*.json"))[:10]:
        try:
            with open(f) as fp:
                data = json.load(fp)
            inner = data.get("data", {})
            meta = inner.get("metadata", {})
            book = meta.get("book_slug", "unknown")
            title = meta.get("title", "")
            for ch in inner.get("chapters", []):
                content = ch.get("content", "")
                if len(content) < 50:
                    continue
                for chunk in _chunk_text(content):
                    yield chunk, {"source": "openstax", "book": book, "title": title, "page": ch.get("page", 0)}
        except (json.JSONDecodeError, KeyError):
            continue


def _load_pmc_chunks(paths: dict) -> Iterator[tuple[str, dict]]:
    """Yield (text, metadata) from PMC raw JSON."""
    raw_dir = paths["raw"] / "pmc"
    if not raw_dir.exists():
        return
    for f in sorted(raw_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
        try:
            with open(f) as fp:
                data = json.load(fp)
            articles = data.get("data", {}).get("articles", [])
            for a in articles:
                title = a.get("title", "")
                abstract = a.get("abstract", "")
                pmcid = a.get("pmcid", "")
                text = f"{title}\n\n{abstract}".strip()
                for chunk in _chunk_text(text):
                    yield chunk, {"source": "pmc", "pmcid": str(pmcid), "title": title[:200]}
        except (json.JSONDecodeError, KeyError):
            continue


def _load_ncbi_bookshelf_chunks(paths: dict) -> Iterator[tuple[str, dict]]:
    """Yield (text, metadata) from NCBI Bookshelf JSON."""
    sections_dir = paths["raw"] / "ncbi_bookshelf" / "sections"
    if not sections_dir.exists():
        return
    for f in list(sections_dir.glob("*.json"))[:10]:
        try:
            with open(f) as fp:
                data = json.load(fp)
            books = data.get("data", {}).get("books", [])
            for b in books:
                title = b.get("title", "")
                abstract = b.get("abstract", "")
                nbk = b.get("nbk_id", "")
                url = b.get("url", "")
                text = f"{title}\n\n{abstract}".strip()
                if text:
                    yield text, {"source": "ncbi_bookshelf", "nbk_id": nbk, "title": title[:200], "url": url}
        except (json.JSONDecodeError, KeyError):
            continue


def build_rag_index(
    persist_dir: Path | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    max_chunks: int = 5000,
) -> None:
    """
    Build RAG vector index from raw data sources.

    Args:
        persist_dir: ChromaDB persist directory. Default: data/vectors/
        model_name: Sentence transformer model name
        max_chunks: Max chunks to index (for quick builds)
    """
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
    except ImportError as e:
        raise ImportError("Install: pip install sentence-transformers chromadb") from e

    paths = get_data_paths()
    if persist_dir is None:
        persist_dir = PROJECT_ROOT / "data" / "vectors"
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Collect chunks
    chunks = []
    metadatas = []
    for text, meta in _load_pubmed_chunks(paths):
        chunks.append(text)
        metadatas.append(meta)
        if len(chunks) >= max_chunks:
            break
    for text, meta in _load_pmc_chunks(paths):
        chunks.append(text)
        metadatas.append(meta)
        if len(chunks) >= max_chunks:
            break
    for text, meta in _load_openstax_chunks(paths):
        chunks.append(text)
        metadatas.append(meta)
        if len(chunks) >= max_chunks:
            break
    for text, meta in _load_ncbi_bookshelf_chunks(paths):
        chunks.append(text)
        metadatas.append(meta)
        if len(chunks) >= max_chunks:
            break

    if not chunks:
        print("No chunks to index. Run fetch scripts first.")
        return

    print(f"Embedding {len(chunks)} chunks with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection("medassist_rag", metadata={"hnsw:space": "cosine"})
    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=[{k: str(v)[:500] for k, v in m.items()} for m in metadatas],
    )
    print(f"Indexed {len(chunks)} chunks to {persist_dir}")


def query_rag(
    query: str,
    n_results: int = 5,
    persist_dir: Path | None = None,
) -> list[dict]:
    """
    Semantic search over RAG index.

    Returns list of {document, metadata, distance}.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
    except ImportError as e:
        raise ImportError("Install: pip install sentence-transformers chromadb") from e

    if persist_dir is None:
        persist_dir = PROJECT_ROOT / "data" / "vectors"
    persist_dir = Path(persist_dir)
    client = chromadb.PersistentClient(path=str(persist_dir))

    try:
        collection = client.get_collection("medassist_rag")
    except Exception:
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([query])
    results = collection.query(query_embeddings=q_emb.tolist(), n_results=n_results)

    out = []
    if results and results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            out.append({
                "document": doc,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
    return out
