#!/usr/bin/env python3
"""Build RAG vector index from raw data (PubMed, PMC, OpenStax, NCBI Bookshelf)."""

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexing.rag_index import build_rag_index


def main():
    parser = argparse.ArgumentParser(description="Build RAG index for semantic search")
    parser.add_argument("--max-chunks", type=int, default=5000, help="Max chunks to index")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--persist-dir", type=Path, help="ChromaDB persist directory")
    args = parser.parse_args()

    build_rag_index(
        persist_dir=args.persist_dir,
        model_name=args.model,
        max_chunks=args.max_chunks,
    )


if __name__ == "__main__":
    main()
