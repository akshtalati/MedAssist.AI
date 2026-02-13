#!/usr/bin/env python3
"""Fetch data from a single MedAssist.AI source.

Usage:
  python scripts/fetch_source.py pubmed
  python scripts/fetch_source.py pubmed --term "acute porphyria" --max_records 200
  python scripts/fetch_source.py openfda --endpoint label --max_records 1000
  python scripts/fetch_source.py orphanet --dataset phenotypes
  python scripts/fetch_source.py rxnorm --query "ibuprofen"
  python scripts/fetch_source.py pmc --term "rare disease" --max_records 500
  python scripts/fetch_source.py who --endpoint documents --limit 50
  python scripts/fetch_source.py ncbi_bookshelf --term "pharmacology" --max_records 50
  python scripts/fetch_source.py openstax --book pharmacology
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fetchers import (
    PubMedFetcher,
    PMCFetcher,
    OpenFDAFetcher,
    OrphanetFetcher,
    RxNormFetcher,
    WHOFetcher,
    NCBIBookshelfFetcher,
    OpenStaxFetcher,
)

FETCHERS = {
    "pubmed": (PubMedFetcher, {"term": "rare disease", "max_records": 500}),
    "pmc": (PMCFetcher, {"term": "rare disease", "max_records": 500}),
    "openfda": (OpenFDAFetcher, {"endpoint": "label", "max_records": 1000}),
    "orphanet": (OrphanetFetcher, {"dataset": "phenotypes"}),
    "rxnorm": (RxNormFetcher, {"query": "aspirin", "max_records": 100}),
    "who": (WHOFetcher, {"endpoint": "documents", "limit": 100}),
    "ncbi_bookshelf": (NCBIBookshelfFetcher, {"term": "pharmacology", "max_records": 50}),
    "openstax": (OpenStaxFetcher, {"book": "pharmacology"}),
}


def main():
    parser = argparse.ArgumentParser(description="Fetch data from a single MedAssist source")
    parser.add_argument("source", choices=list(FETCHERS), help="Data source to fetch")
    parser.add_argument("--term", help="Search term (pubmed, pmc)")
    parser.add_argument("--max_records", type=int, help="Max records to fetch")
    parser.add_argument("--query", help="Query string (rxnorm)")
    parser.add_argument("--endpoint", help="Endpoint (openfda: label/event/ndc; who: documents/guidelines)")
    parser.add_argument("--dataset", help="Dataset (orphanet: phenotypes/diseases/genes)")
    parser.add_argument("--limit", type=int, help="Limit (who)")
    parser.add_argument("--search", help="OpenFDA search query")
    parser.add_argument("--book", help="Book slug (openstax: pharmacology, anatomy-physiology-2e, etc.)")

    args = parser.parse_args()
    fetcher_cls, defaults = FETCHERS[args.source]
    fetcher = fetcher_cls()

    kwargs = dict(defaults)
    if args.term is not None:
        kwargs["term"] = args.term
    if args.max_records is not None:
        kwargs["max_records"] = args.max_records
    if args.query is not None:
        kwargs["query"] = args.query
    if args.endpoint is not None:
        kwargs["endpoint"] = args.endpoint
    if args.dataset is not None:
        kwargs["dataset"] = args.dataset
    if args.limit is not None:
        kwargs["limit"] = args.limit
    if args.search is not None:
        kwargs["search"] = args.search
    if args.book is not None:
        kwargs["book"] = args.book

    path = fetcher.fetch(**kwargs)
    print(f"Fetched to: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
