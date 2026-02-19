#!/usr/bin/env python3
"""
Keep only the best file per source (most records, then newest). Delete the rest.
Run from project root.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"


def record_count(path: Path, source: str, subdir: str) -> int:
    """Return number of records in the file for ranking."""
    try:
        with open(path) as f:
            data = json.load(f)
        body = data.get("data", data)
        if source == "pubmed":
            return len(body.get("articles", []))
        if source == "pmc":
            return len(body.get("articles", []))
        if source == "openfda":
            return len(body.get("results", []))
        if source == "rxnorm":
            return len(body.get("drugs", [])) or len(body.get("raw_response", {}).get("approximateGroup", {}).get("candidate", []))
        if source == "who":
            return len(body.get("results", body) if isinstance(body, list) else body.get("items", []))
        if source == "ncbi_bookshelf":
            return len(body.get("books", []))
        if source == "orphanet":
            st = body.get("HPODisorderSetStatusList", {}) or body.get("data", {})
            disorders = st.get("HPODisorderSetStatus", [])
            return len(disorders) if isinstance(disorders, list) else 1
        if source == "openstax":
            return len(body.get("chapters", []))
        return 0
    except Exception:
        return -1


# (source, subdir relative to raw)
SOURCES = [
    ("pubmed", ""),
    ("pmc", ""),
    ("openfda", "label"),
    ("openfda", "event"),
    ("rxnorm", ""),
    ("who", "documents"),
    ("ncbi_bookshelf", "sections"),
    ("orphanet", "phenotypes"),
    ("openstax", "extracted"),
]


def main():
    deleted = 0
    kept = []

    for source, subdir in SOURCES:
        dir_path = RAW / source
        if subdir:
            dir_path = dir_path / subdir
        if not dir_path.exists():
            continue
        files = list(dir_path.glob("*.json"))
        if len(files) <= 1:
            continue

        # Rank by (record_count desc, mtime desc)
        def key(p):
            c = record_count(p, source.split("/")[0] if "/" in source else source, subdir)
            return (-c, -p.stat().st_mtime)

        ordered = sorted(files, key=key)
        keep = ordered[0]
        to_remove = ordered[1:]
        kept.append(str(keep))
        for p in to_remove:
            p.unlink()
            deleted += 1
            print(f"Deleted: {p.relative_to(PROJECT_ROOT)}")

    print(f"\nKept {len(kept)} files, deleted {deleted} duplicates.")
    for k in kept:
        print(f"  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
