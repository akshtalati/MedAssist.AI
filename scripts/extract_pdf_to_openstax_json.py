#!/usr/bin/env python3
"""
Extract text from a local PDF and write OpenStax-style JSON for the Snowflake loader.

Usage:
  python scripts/extract_pdf_to_openstax_json.py path/to/book.pdf
  python scripts/extract_pdf_to_openstax_json.py path/to/book.pdf --book mybook --title "My Book Title"

Output: data/raw/openstax/extracted/{book}_{timestamp}.json
Then run: python scripts/load_to_snowflake.py --openstax (or --all) to load into Snowflake.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "raw" / "openstax" / "extracted"


def main():
    parser = argparse.ArgumentParser(description="Extract PDF to OpenStax-style JSON for loading")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    parser.add_argument("--book", "-b", default="", help="Book slug (default: stem of PDF filename)")
    parser.add_argument("--title", "-t", default="", help="Book title (default: book slug)")
    parser.add_argument("--out-dir", type=Path, default=EXTRACTED_DIR, help="Output directory for JSON")
    args = parser.parse_args()

    pdf_path = args.pdf_path.resolve()
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    book_slug = args.book or pdf_path.stem.lower().replace(" ", "-").replace("_", "-")
    title = args.title or book_slug.replace("-", " ").title()

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("Error: PyMuPDF required. Run: pip install PyMuPDF", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fetch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    chapters = []
    char_count = 0
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            chapters.append({"page": i + 1, "content": text})
            char_count += len(text)
    doc.close()

    payload = {
        "metadata": {
            "title": title,
            "book_slug": book_slug,
            "source_url": "",
            "license": "CC BY 4.0",
            "raw_pdf_path": str(pdf_path.relative_to(PROJECT_ROOT) if PROJECT_ROOT in pdf_path.parents else pdf_path.name),
            "page_count": len(chapters),
            "char_count": char_count,
        },
        "chapters": chapters,
    }
    out_path = args.out_dir / f"{book_slug}_{fetch_id}.json"
    full_payload = {
        "_header": {
            "source": "openstax",
            "fetch_id": fetch_id,
            "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "schema_version": "1.0",
        },
        "data": payload,
    }
    with open(out_path, "w") as f:
        json.dump(full_payload, f, indent=2, default=str)

    print(f"Wrote {len(chapters)} pages to {out_path}")
    print(f"Load into Snowflake: python scripts/load_to_snowflake.py --openstax")


if __name__ == "__main__":
    main()
