#!/usr/bin/env python3
"""Build symptom→disease index from Orphanet phenotypes data."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing import build_symptom_index


def main():
    index = build_symptom_index()
    print(f"Built symptom index: {index.index_path}")
    print(f"  Symptoms: {len(index._symptom_to_diseases)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
