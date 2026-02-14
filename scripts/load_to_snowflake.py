#!/usr/bin/env python3
"""
Load MedAssist.AI data into Snowflake.

Run snowflake_setup.sql first to create warehouse, database, and tables.
Set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD in .env.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_data_paths, PROJECT_ROOT
from src.snowflake_client import get_connection


def load_manifests(cursor) -> int:
    """Load fetch manifests from data/metadata/."""
    meta_dir = PROJECT_ROOT / "data" / "metadata"
    if not meta_dir.exists():
        return 0
    count = 0
    for f in meta_dir.glob("*_manifest.json"):
        try:
            with open(f) as fp:
                m = json.load(fp)
            cursor.execute(
                """
                INSERT INTO RAW.FETCH_MANIFESTS
                (source, fetch_id, fetched_at, api_endpoint, query_params, record_count,
                 total_available, file_path, checksum_sha256, status, error)
                SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                WHERE NOT EXISTS (SELECT 1 FROM RAW.FETCH_MANIFESTS WHERE source=%s AND fetch_id=%s)
                """,
                (
                    m.get("source", ""),
                    m.get("fetch_id", ""),
                    m.get("fetched_at"),
                    m.get("api_endpoint", ""),
                    json.dumps(m.get("query_params", {})),
                    m.get("record_count", 0),
                    m.get("total_available"),
                    m.get("file_path", ""),
                    m.get("checksum_sha256"),
                    m.get("status", "success"),
                    m.get("error"),
                    m.get("source", ""),
                    m.get("fetch_id", ""),
                ),
            )
            count += 1
        except Exception as e:
            print(f"  Skip {f.name}: {e}")
    return count


def load_symptom_index(cursor) -> int:
    """Load symptom→disease map from data/normalized/symptom_index.json."""
    path = PROJECT_ROOT / "data" / "normalized" / "symptom_index.json"
    if not path.exists():
        return 0
    with open(path) as f:
        data = json.load(f)
    stod = data.get("symptom_to_diseases", {})
    count = 0
    for symptom, entries in stod.items():
        for entry in entries:
            if isinstance(entry, (list, tuple)):
                orpha = str(entry[0]) if len(entry) > 0 else ""
                name = str(entry[1]) if len(entry) > 1 else ""
                freq = str(entry[2]) if len(entry) > 2 else ""
                hpo = str(entry[3]) if len(entry) > 3 else ""
            else:
                orpha, name, freq, hpo = "", "", "", ""
            cursor.execute(
                """
                INSERT INTO NORMALIZED.SYMPTOM_DISEASE_MAP
                (symptom, orpha_code, disease_name, frequency, hpo_id)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (symptom[:500], orpha[:20], name[:500], freq[:100], hpo[:50]),
            )
            count += 1
    return count


def load_pubmed(cursor) -> int:
    """Load PubMed articles from data/raw/pubmed/."""
    raw_dir = PROJECT_ROOT / "data" / "raw" / "pubmed"
    if not raw_dir.exists():
        return 0
    count = 0
    for f in raw_dir.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            articles = data.get("data", {}).get("articles", [])
            for a in articles:
                pmid = a.get("pmid")
                if pmid is None:
                    continue
                cursor.execute(
                    """
                    INSERT INTO RAW.PUBMED_ARTICLES (pmid, title, abstract, journal, pub_date, source_file)
                    SELECT %s, %s, %s, %s, %s, %s
                    WHERE NOT EXISTS (SELECT 1 FROM RAW.PUBMED_ARTICLES WHERE pmid=%s)
                    """,
                    (
                        pmid,
                        str(a.get("title", ""))[:1000],
                        str(a.get("abstract", ""))[:100000],
                        str(a.get("journal", ""))[:200],
                        str(a.get("pub_date", ""))[:50],
                        f.name,
                        pmid,
                    ),
                )
                count += 1
        except Exception as e:
            print(f"  Skip {f.name}: {e}")
    return count


def load_pmc(cursor) -> int:
    """Load PMC articles from data/raw/pmc/."""
    raw_dir = PROJECT_ROOT / "data" / "raw" / "pmc"
    if not raw_dir.exists():
        return 0
    count = 0
    for f in raw_dir.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            articles = data.get("data", {}).get("articles", [])
            for a in articles:
                pmcid = a.get("pmcid")
                if not pmcid:
                    continue
                cursor.execute(
                    """
                    INSERT INTO RAW.PMC_ARTICLES (pmcid, title, abstract, journal, pub_date, source_file)
                    SELECT %s, %s, %s, %s, %s, %s
                    WHERE NOT EXISTS (SELECT 1 FROM RAW.PMC_ARTICLES WHERE pmcid=%s)
                    """,
                    (
                        str(pmcid)[:50],
                        str(a.get("title", ""))[:1000],
                        str(a.get("abstract", ""))[:100000],
                        str(a.get("journal", ""))[:200],
                        str(a.get("pub_date", ""))[:50],
                        f.name,
                        str(pmcid)[:50],
                    ),
                )
                count += 1
        except Exception as e:
            print(f"  Skip {f.name}: {e}")
    return count


def load_rag_chunks(cursor) -> int:
    """Load RAG chunks from ChromaDB into Snowflake VECTORS.RAG_CHUNKS."""
    try:
        import chromadb
    except ImportError:
        print("  chromadb not installed, skipping RAG chunks")
        return 0

    persist_dir = PROJECT_ROOT / "data" / "vectors"
    if not persist_dir.exists():
        return 0
    try:
        client = chromadb.PersistentClient(path=str(persist_dir))
        coll = client.get_collection("medassist_rag")
    except Exception:
        return 0

    # Fetch all with embeddings
    data = coll.get(include=["documents", "metadatas", "embeddings"])
    if not data or not data["ids"]:
        return 0

    count = 0
    for i, cid in enumerate(data["ids"]):
        doc = data["documents"][i] if data["documents"] else ""
        meta = data["metadatas"][i] if data["metadatas"] else {}
        emb = data["embeddings"][i] if data["embeddings"] else []
        if len(emb) != 384:
            continue  # all-MiniLM-L6-v2 produces 384-dim vectors
        try:
            meta_json = json.dumps(meta) if meta else "{}"
            emb_str = "[" + ",".join(str(float(x)) for x in emb) + "]"
            cursor.execute(
                """
                INSERT INTO VECTORS.RAG_CHUNKS (chunk_id, source, document_text, metadata, embedding)
                SELECT %s, %s, %s, PARSE_JSON(%s), PARSE_JSON(%s)::VECTOR(FLOAT, 384)
                WHERE NOT EXISTS (SELECT 1 FROM VECTORS.RAG_CHUNKS WHERE chunk_id=%s)
                """,
                (cid, meta.get("source", "unknown"), doc[:100000], meta_json, emb_str, cid),
            )
            count += 1
        except Exception as e:
            print(f"  Skip chunk {cid}: {e}")
    return count


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Load MedAssist data into Snowflake")
    parser.add_argument("--manifests", action="store_true", help="Load fetch manifests")
    parser.add_argument("--symptoms", action="store_true", help="Load symptom index")
    parser.add_argument("--pubmed", action="store_true", help="Load PubMed articles")
    parser.add_argument("--pmc", action="store_true", help="Load PMC articles")
    parser.add_argument("--rag", action="store_true", help="Load RAG chunks from ChromaDB")
    parser.add_argument("--all", action="store_true", help="Load everything")
    args = parser.parse_args()

    load_all = args.all or not any([args.manifests, args.symptoms, args.pubmed, args.pmc, args.rag])

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("USE WAREHOUSE MEDASSIST_WH")
        cursor.execute("USE DATABASE MEDASSIST_DB")

        if load_all or args.manifests:
            n = load_manifests(cursor)
            print(f"Loaded {n} manifests")
        if load_all or args.symptoms:
            n = load_symptom_index(cursor)
            print(f"Loaded {n} symptom-disease rows")
        if load_all or args.pubmed:
            n = load_pubmed(cursor)
            print(f"Loaded {n} PubMed articles")
        if load_all or args.pmc:
            n = load_pmc(cursor)
            print(f"Loaded {n} PMC articles")
        if load_all or args.rag:
            n = load_rag_chunks(cursor)
            print(f"Loaded {n} RAG chunks")

        conn.commit()
        print("Done.")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
