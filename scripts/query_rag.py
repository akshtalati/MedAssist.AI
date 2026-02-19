#!/usr/bin/env python3
"""Semantic search over RAG index (PubMed, OpenStax, NCBI Bookshelf)."""

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexing.rag_index import query_rag


def main():
    parser = argparse.ArgumentParser(description="Query RAG index with semantic search")
    parser.add_argument("query", nargs="+", help="Search query (joined with spaces)")
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of results")
    parser.add_argument("--persist-dir", type=Path, help="ChromaDB persist directory")
    args = parser.parse_args()

    query = " ".join(args.query)
    results = query_rag(query, n_results=args.num, persist_dir=args.persist_dir)

    if not results:
        print("No results. Build the index first: python scripts/build_rag_index.py")
        return

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        source = meta.get("source", "?")
        title = meta.get("title", "")[:60]
        print(f"\n--- Result {i} [{source}] {title} ---")
        doc = r.get("document", "")
        print(doc[:400] + ("..." if len(doc) > 400 else ""))
        if r.get("distance") is not None:
            print(f"(distance: {r['distance']:.4f})")


if __name__ == "__main__":
    main()
