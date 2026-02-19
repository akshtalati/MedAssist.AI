#!/usr/bin/env python3
"""Fetch data from all MedAssist.AI sources."""

import sys
from pathlib import Path

# Add project root to path
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


def main():
    fetchers = [
        ("Orphanet", OrphanetFetcher(), {"dataset": "phenotypes"}),
        ("PubMed", PubMedFetcher(), {"term": "rare disease", "max_records": 100}),
        ("PMC", PMCFetcher(), {"term": "rare disease", "max_records": 100}),
        ("OpenFDA (labels)", OpenFDAFetcher(), {"endpoint": "label", "max_records": 2000}),
        ("OpenFDA (events)", OpenFDAFetcher(), {"endpoint": "event", "max_records": 2000}),
        ("RxNorm", RxNormFetcher(), {"query": "aspirin", "max_records": 50}),
        ("WHO", WHOFetcher(), {"endpoint": "documents", "limit": 50}),
        ("NCBI Bookshelf", NCBIBookshelfFetcher(), {"term": "pharmacology", "max_records": 30}),
        ("OpenStax (pharmacology)", OpenStaxFetcher(), {"book": "pharmacology"}),
    ]

    results = []
    for name, fetcher, kwargs in fetchers:
        try:
            path = fetcher.fetch(**kwargs)
            results.append((name, str(path), "OK"))
            print(f"[OK] {name}: {path}")
        except Exception as e:
            results.append((name, "", str(e)))
            print(f"[FAIL] {name}: {e}")

    print("\nSummary:")
    for name, path, status in results:
        print(f"  {name}: {status}")
    return 0 if all(r[2] == "OK" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
