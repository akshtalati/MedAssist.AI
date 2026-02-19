#!/usr/bin/env python3
"""Remove manifest files for deleted raw fetches."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
META = PROJECT_ROOT / "data" / "metadata"


def main():
    kept_ids = set()
    for f in RAW.rglob("*.json"):
        try:
            with open(f) as fp:
                h = json.load(fp).get("_header", {})
            fid = h.get("fetch_id")
            if fid:
                kept_ids.add(fid)
        except Exception:
            pass

    deleted = 0
    for m in META.glob("*_manifest.json"):
        fid = m.stem.replace("_manifest", "")
        if fid not in kept_ids:
            m.unlink()
            deleted += 1
            print(f"Deleted manifest: {m.name}")
    print(f"Removed {deleted} orphan manifests. Kept {len(kept_ids)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
