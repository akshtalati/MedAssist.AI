#!/usr/bin/env python3
"""
Fetch from all sources with checkpoint. Uses higher limits for more data.
On timeout or rate limit: saves checkpoint and exits. Run again to resume.
Checkpoint: data/.fetch_checkpoint.json
"""

import json
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_FILE = PROJECT_ROOT / "data" / ".fetch_checkpoint.json"

# Higher limits for more data (adjust if you hit timeouts/rate limits)
CONFIG_DEFAULT = [
    ("orphanet", "Orphanet", {"dataset": "phenotypes"}),
    ("pubmed", "PubMed", {"term": "rare disease", "max_records": 5000}),
    ("pmc", "PMC", {"term": "rare disease", "max_records": 2000}),
    ("openfda_label", "OpenFDA (labels)", {"endpoint": "label", "max_records": 25000}),
    ("openfda_event", "OpenFDA (events)", {"endpoint": "event", "max_records": 25000}),
    ("rxnorm", "RxNorm", {"query": "aspirin", "max_records": 100}),
    ("who", "WHO", {"endpoint": "documents", "limit": 200}),
    ("ncbi_bookshelf", "NCBI Bookshelf", {"term": "pharmacology", "max_records": 100}),
    ("openstax", "OpenStax (pharmacology)", {"book": "pharmacology"}),
]

CONFIG_FULL = [
    ("orphanet_phenotypes", "Orphanet phenotypes", {"dataset": "phenotypes"}),
    ("orphanet_diseases", "Orphanet diseases", {"dataset": "diseases"}),
    ("orphanet_genes", "Orphanet genes", {"dataset": "genes"}),
    ("pubmed", "PubMed", {"term": "rare disease OR differential diagnosis OR red flag", "max_records": 10000}),
    ("pmc", "PMC", {"term": "rare disease OR differential diagnosis OR red flag", "max_records": 5000}),
    ("openfda_label", "OpenFDA (labels)", {"endpoint": "label", "max_records": 50000}),
    ("openfda_event", "OpenFDA (events)", {"endpoint": "event", "max_records": 50000}),
    ("openfda_ndc", "OpenFDA (ndc)", {"endpoint": "ndc", "max_records": 50000}),
    ("rxnorm", "RxNorm", {"query": "aspirin", "max_records": 2000}),
    ("who", "WHO", {"endpoint": "documents", "limit": 1000}),
    ("ncbi_bookshelf", "NCBI Bookshelf", {"term": "differential diagnosis", "max_records": 1000}),
    ("openstax", "OpenStax (pharmacology)", {"book": "pharmacology"}),
]

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text())
        except Exception:
            pass
    return {"completed": [], "last_error": None}


def save_checkpoint(completed: list, last_error: str = None):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(
        json.dumps({"completed": completed, "last_error": last_error}, indent=2)
    )


def main():
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

    fetcher_map = {
        "orphanet": OrphanetFetcher(),
        "orphanet_phenotypes": OrphanetFetcher(),
        "orphanet_diseases": OrphanetFetcher(),
        "orphanet_genes": OrphanetFetcher(),
        "pubmed": PubMedFetcher(),
        "pmc": PMCFetcher(),
        "openfda_label": OpenFDAFetcher(),
        "openfda_event": OpenFDAFetcher(),
        "openfda_ndc": OpenFDAFetcher(),
        "rxnorm": RxNormFetcher(),
        "who": WHOFetcher(),
        "ncbi_bookshelf": NCBIBookshelfFetcher(),
        "openstax": OpenStaxFetcher(),
    }

    ap = argparse.ArgumentParser(description="Checkpointed fetch across sources")
    ap.add_argument("--reset", action="store_true", help="Reset checkpoint before fetch.")
    ap.add_argument(
        "--profile",
        choices=("default", "full"),
        default="default",
        help="default=lighter fetch, full=all supported source variants with larger limits.",
    )
    args = ap.parse_args()

    config = CONFIG_FULL if args.profile == "full" else CONFIG_DEFAULT

    cp = load_checkpoint()
    completed = list(cp.get("completed", []))
    if cp.get("last_error"):
        print(f"Resuming after last error: {cp['last_error'][:200]}")

    if args.reset:
        completed = []
        save_checkpoint(completed, None)
        print("Fetch checkpoint reset.")

    for key, name, kwargs in config:
        if key in completed:
            print(f"[skip] {name} (already done)")
            continue
        print(f"\n>>> Fetching {name} ...")
        try:
            fetcher = fetcher_map[key]
            fetcher.fetch(**kwargs)
            completed.append(key)
            save_checkpoint(completed, None)
            print(f"[OK] {name}")
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)}"
            save_checkpoint(completed, err_msg)
            print(f"\n[FAIL] {name}: {err_msg}")
            print("Checkpoint saved. Run again to resume from next source.")
            sys.exit(1)

    save_checkpoint(completed, None)
    print("\n--- All sources fetched ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
