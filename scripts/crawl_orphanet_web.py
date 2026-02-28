#!/usr/bin/env python3
"""Crawl orpha.net disease pages: iterate all ORPHA codes, fetch each page, save as Markdown + metadata.

Uses ORPHA codes from local Orphadata JSON (data/raw/orphanet/product6/ or product4/ or diseases/).
Stores only JSON/MD (no HTML). Each file has metadata: url, orpha_code, fetched_at.

Usage:
  python scripts/crawl_orphanet_web.py
  python scripts/crawl_orphanet_web.py --orphanet-dir data/raw/orphanet --format md
  python scripts/crawl_orphanet_web.py --limit 10
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_data_paths


def _extract_orpha_codes_from_payload(payload: dict) -> set[str]:
    """Extract all OrphaNumber (or OrphaCode) from Orphadata JSON data payload."""
    codes = set()
    disorder_list = payload.get("DisorderList") or {}
    if isinstance(disorder_list, dict):
        disorders = disorder_list.get("Disorder", [])
        if not isinstance(disorders, list):
            disorders = [disorders]
        for d in disorders:
            code = (d.get("OrphaNumber") or d.get("OrphaCode") or "")
            if code:
                codes.add(str(code).strip())
    return codes


def load_orpha_codes_from_orphanet_dir(orphanet_dir: Path) -> set[str]:
    """Find latest Orphadata JSON under orphanet_dir (product6, product4, or diseases) and extract all ORPHA codes."""
    all_codes = set()
    # Prefer product6 (diseases) or product4 (phenotypes); fallback diseases/phenotypes subdirs
    for subdir in ("product6", "product4", "diseases", "phenotypes"):
        d = orphanet_dir / subdir
        if not d.is_dir():
            continue
        json_files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in json_files:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                payload = data.get("data", data)
                if isinstance(payload, dict):
                    all_codes |= _extract_orpha_codes_from_payload(payload)
                if all_codes:
                    return all_codes
            except Exception:
                continue
    return all_codes


def fetch_page(url: str, session, rate_limit_sec: float = 1.5) -> str | None:
    """GET url, return response text or None."""
    time.sleep(rate_limit_sec)
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception:
        return None


def html_to_markdown(html: str) -> str:
    """Convert HTML to Markdown. Requires markdownify."""
    try:
        from markdownify import markdownify as md
        return md(html or "", heading_style="ATX", strip=["script", "style"])
    except ImportError:
        # Fallback: strip tags crudely
        return re.sub(r"<[^>]+>", "", html or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Crawl orpha.net disease pages; save as Markdown + metadata")
    parser.add_argument("--orphanet-dir", type=Path, default=None, help="Path to data/raw/orphanet (default: from config)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output dir for web/ (default: orphanet_dir/web)")
    parser.add_argument("--format", choices=("md", "json"), default="md", help="Save as .md or structured .json")
  parser.add_argument("--limit", type=int, default=None, help="Max number of pages to fetch (default: all)")
  parser.add_argument("--delay", type=float, default=1.5, help="Seconds between requests")
  parser.add_argument("--lists", action="store_true", help="Also crawl alphabetical list pages (en/disease/list/0, a..z)")
  args = parser.parse_args()

    paths = get_data_paths()
    raw_root = paths["raw"]
    if not isinstance(raw_root, Path):
        raw_root = PROJECT_ROOT / str(raw_root)
    orphanet_dir = args.orphanet_dir or raw_root / "orphanet"
    out_dir = args.out_dir or orphanet_dir / "web"
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = load_orpha_codes_from_orphanet_dir(orphanet_dir)
    if not codes:
        print("No ORPHA codes found. Run fetch_orphanet_all.py (or fetch product4/product6) first.", file=sys.stderr)
        return 1

    codes = sorted(codes)
    if args.limit:
        codes = codes[: args.limit]
    print(f"Crawling {len(codes)} disease pages -> {out_dir} (format={args.format}, delay={args.delay}s)", flush=True)

    import requests
    session = requests.Session()
    session.headers["User-Agent"] = "MedAssist.AI-crawler/1.0 (local research)"

    ok = 0
    failed = 0
    for i, orpha_code in enumerate(codes):
        url = f"https://www.orpha.net/consor/cgi-bin/OC_Exp.php?lng=en&Expert={orpha_code}"
        html = fetch_page(url, session, rate_limit_sec=args.delay)
        if not html:
            failed += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(codes)} ...", flush=True)
            continue

        fetched_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        safe_code = re.sub(r"[^\w\-]", "_", orpha_code)

        if args.format == "md":
            content = html_to_markdown(html)
            content_path = out_dir / f"{safe_code}.md"
            content_path.write_text(content, encoding="utf-8")
            meta = {"url": url, "orpha_code": orpha_code, "fetched_at": fetched_at, "content_path": content_path.name}
            meta_path = out_dir / f"{safe_code}.metadata.json"
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        else:
            # Structured JSON: title + body as markdown
            title = ""
            if "<title>" in html:
                m = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
                if m:
                    title = m.group(1).strip()
            body_md = html_to_markdown(html)
            doc = {
                "url": url,
                "orpha_code": orpha_code,
                "fetched_at": fetched_at,
                "title": title,
                "content": body_md,
            }
            content_path = out_dir / f"{safe_code}.json"
            content_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
            meta_path = out_dir / f"{safe_code}.metadata.json"
            meta_path.write_text(
                json.dumps({"url": url, "orpha_code": orpha_code, "fetched_at": fetched_at, "content_path": content_path.name}, indent=2),
                encoding="utf-8",
            )

        ok += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(codes)} ...", flush=True)

    print(f"Done: {ok} saved, {failed} failed.", flush=True)

    if args.lists:
        list_dir = out_dir / "list"
        list_dir.mkdir(parents=True, exist_ok=True)
        letters = ["0"] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
        print(f"Crawling {len(letters)} list pages -> {list_dir}", flush=True)
        for letter in letters:
            url = f"https://www.orpha.net/en/disease/list/{letter}"
            html = fetch_page(url, session, rate_limit_sec=args.delay)
            if not html:
                continue
            fetched_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            if args.format == "md":
                content = html_to_markdown(html)
                content_path = list_dir / f"{letter}.md"
                content_path.write_text(content, encoding="utf-8")
            else:
                title = ""
                if "<title>" in html:
                    m = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
                    if m:
                        title = m.group(1).strip()
                content = html_to_markdown(html)
                doc = {"url": url, "orpha_code": None, "fetched_at": fetched_at, "title": title, "content": content}
                content_path = list_dir / f"{letter}.json"
                content_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
            meta = {"url": url, "orpha_code": None, "fetched_at": fetched_at, "content_path": content_path.name}
            meta_path = list_dir / f"{letter}.metadata.json"
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("List pages done.", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
