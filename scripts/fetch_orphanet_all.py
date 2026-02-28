#!/usr/bin/env python3
"""Fetch all Orphadata products locally (no Snowflake/GCP).

Iterates over every product in the Orphanet product registry, downloads XML,
converts to JSON, and writes to data/raw/orphanet/<product_id>/.

Usage:
  python scripts/fetch_orphanet_all.py
  python scripts/fetch_orphanet_all.py --language en
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fetchers.orphanet import OrphanetFetcher


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch all Orphadata products to local data/raw/orphanet/")
    parser.add_argument("--language", default="en", help="Language code (en, fr, de, es, it, nl, pt)")
    args = parser.parse_args()

    fetcher = OrphanetFetcher()
    registry = fetcher.get_product_registry()
    print(f"Fetching {len(registry)} Orphadata products (language={args.language})...", flush=True)

    ok = 0
    failed = 0
    for product_id, path in fetcher.fetch_all_products(language=args.language):
        if path is not None:
            print(f"  {product_id}: {path}", flush=True)
            ok += 1
        else:
            print(f"  {product_id}: failed (see metadata for details)", flush=True)
            failed += 1

    print(f"Done: {ok} succeeded, {failed} failed.", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
