#!/usr/bin/env python3
"""
Offline retrieval smoke: hybrid merge + optional rerank (no LLM).

Usage:
  PYTHONPATH=. python evals/run_retrieval_eval.py --queries "fever vomiting" "chest pain dyspnea"
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", nargs="+", default=["fever and vomiting in adult", "acute chest pain shortness of breath"])
    args = p.parse_args()

    from api.main import _fetch_ilike_entries, _fetch_rag_entries, _merge_and_dedupe_literature
    from src.retrieval.rerank import maybe_rerank_literature

    for q in args.queries:
        rag = _fetch_rag_entries(q, limit=15)
        ilike = _fetch_ilike_entries(q, limit=30) if len(rag) < 5 else []
        merged = _merge_and_dedupe_literature(rag, ilike)
        reranked = maybe_rerank_literature(q, merged, top_k=20)
        print("---")
        print("Query:", q)
        print("  RAG hits:", len(rag), " ILIKE hits:", len(ilike), " merged:", len(merged), " after_rerank:", len(reranked))
        for i, e in enumerate(reranked[:5], start=1):
            title = (e.get("title") or "")[:80]
            print(f"  {i}. {e.get('source')} | {title}")


if __name__ == "__main__":
    main()
