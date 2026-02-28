#!/usr/bin/env python3
"""
Export Orphanet web crawl (.md + .metadata.json) to one JSON file per page
for GCS upload and Snowflake COPY. Output: data/raw/orphanet_web_json/*.json

Run before: python scripts/upload_to_gcp.py --all
So that gs://medassist-data-gcs/medassist/data/raw/orphanet_web_json/ is populated.

Usage:
  python scripts/export_orphanet_web_to_json.py
  python scripts/export_orphanet_web_to_json.py --out-dir data/raw/orphanet_web_json
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Orphanet web crawl to JSON for GCS/Snowflake")
    parser.add_argument("--web-dir", type=Path, default=None, help="Path to data/raw/orphanet/web (default: from config)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output dir (default: data/raw/orphanet_web_json)")
    args = parser.parse_args()

    raw = PROJECT_ROOT / "data" / "raw"
    web_dir = args.web_dir or raw / "orphanet" / "web"
    out_dir = args.out_dir or raw / "orphanet_web_json"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not web_dir.exists():
        print("Web dir not found:", web_dir, file=sys.stderr)
        return 1

    count = 0
    for md_path in sorted(web_dir.glob("*.md")):
        if md_path.name.startswith("."):
            continue
        meta_path = web_dir / (md_path.stem + ".metadata.json")
        try:
            content = md_path.read_text(encoding="utf-8")
            url = ""
            orpha_code = md_path.stem
            fetched_at = ""
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                url = meta.get("url", "")
                orpha_code = str(meta.get("orpha_code", orpha_code))
                fetched_at = meta.get("fetched_at", "")
            title = ""
            json_path = web_dir / (md_path.stem + ".json")
            if json_path.exists():
                try:
                    doc = json.loads(json_path.read_text(encoding="utf-8"))
                    title = doc.get("title", "")
                except Exception:
                    pass
            safe = re.sub(r"[^\w\-]", "_", orpha_code)
            out_file = out_dir / f"{safe}.json"
            doc = {
                "orpha_code": orpha_code,
                "url": url,
                "title": title,
                "content": content,
                "fetched_at": fetched_at,
            }
            out_file.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            count += 1
        except Exception as e:
            print(f"Skip {md_path.name}: {e}", file=sys.stderr)
    print(f"Exported {count} pages to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
