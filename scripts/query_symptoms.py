#!/usr/bin/env python3
"""
Query diseases by symptoms. Phase 2 - makes raw data useful.

Usage:
  python scripts/query_symptoms.py "fever" "vomiting"
  python scripts/query_symptoms.py "headache" "nausea"
  python scripts/query_symptoms.py "abdominal pain" --any   # match ANY symptom (union)
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing import SymptomIndex


def main():
    parser = argparse.ArgumentParser(description="Query diseases by symptoms (Phase 2)")
    parser.add_argument("symptoms", nargs="+", help="Symptom terms (e.g. fever vomiting)")
    parser.add_argument("--any", action="store_true", help="Match ANY symptom (default: match ALL)")
    parser.add_argument("--build", action="store_true", help="Build index first if missing")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    index = SymptomIndex()
    if not index.index_path.exists():
        if args.build:
            from src.indexing import build_symptom_index
            index = build_symptom_index()
        else:
            print("Symptom index not found. Run: python scripts/build_symptom_index.py", file=sys.stderr)
            return 1

    results = index.query(args.symptoms, match_all=not args.any)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"Query: {' + '.join(args.symptoms)} {'(match any)' if args.any else '(match all)'}")
        print(f"Found {len(results)} disease(s)\n")
        for r in results[:20]:
            print(f"  • {r['disease_name']} (ORPHA:{r['orpha_code']})")
            print(f"    {r['orpha_url']}")
        if len(results) > 20:
            print(f"  ... and {len(results) - 20} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
