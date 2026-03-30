#!/usr/bin/env python3
"""Create curated PMC output + quality report from latest/raw PMC fetch JSON."""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "pmc"
OUT_DIR = PROJECT_ROOT / "data" / "normalized"


def _norm_text(v: str) -> str:
    return re.sub(r"\s+", " ", (v or "")).strip()


def _norm_nullable(v) -> str:
    text = _norm_text("" if v is None else str(v))
    return "" if text.lower() in {"none", "null", "nan"} else text


def _pick_input_file(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return p
    files = sorted(RAW_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No PMC raw files found in: {RAW_DIR}")
    return files[0]


def _stable_id(title: str, pub_date: str, abstract: str) -> str:
    basis = f"{title}|{pub_date}|{abstract[:500]}"
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:40]
    return f"PMCGEN_{digest}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Transform PMC raw JSON into curated JSON.")
    ap.add_argument("--input-file", help="Explicit input JSON path. Default: latest in data/raw/pmc/")
    ap.add_argument("--output", help="Curated output path. Default: data/normalized/pmc_curated_<run_id>.json")
    ap.add_argument(
        "--report-output",
        help="Quality report path. Default: data/normalized/pmc_quality_report_<run_id>.json",
    )
    ap.add_argument("--strict", action="store_true", help="Fail if valid ratio is below threshold.")
    ap.add_argument("--min-valid-ratio", type=float, default=0.60, help="Threshold for --strict mode.")
    args = ap.parse_args()

    in_file = _pick_input_file(args.input_file)
    payload = json.loads(in_file.read_text())
    articles = payload.get("data", {}).get("articles", [])

    seen: dict[str, dict] = {}
    dropped_no_key = 0
    dropped_empty_content = 0
    duplicates_removed = 0

    for a in articles:
        title = _norm_nullable(a.get("title", ""))
        abstract = _norm_nullable(a.get("abstract", ""))
        journal = _norm_nullable(a.get("journal", ""))
        pub_date = _norm_nullable(a.get("pub_date", ""))
        pmcid = _norm_nullable(a.get("pmcid", ""))

        if not pmcid:
            if not (title or abstract):
                dropped_no_key += 1
                continue
            pmcid = _stable_id(title, pub_date, abstract)

        if not (title or abstract):
            dropped_empty_content += 1
            continue

        row = {
            "pmcid": pmcid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "pub_date": pub_date,
            "authors": a.get("authors", []),
        }

        prev = seen.get(pmcid)
        if prev is None:
            seen[pmcid] = row
            continue

        # Keep record with richer text payload when duplicate keys appear.
        prev_score = len(prev.get("abstract", "")) + len(prev.get("title", ""))
        row_score = len(abstract) + len(title)
        if row_score > prev_score:
            seen[pmcid] = row
        duplicates_removed += 1

    curated = list(seen.values())
    input_rows = len(articles)
    valid_rows = len(curated)
    valid_ratio = (valid_rows / input_rows) if input_rows else 0.0

    if args.strict and valid_ratio < args.min_valid_ratio:
        print(
            f"FAIL: valid ratio {valid_ratio:.3f} < {args.min_valid_ratio:.3f}. "
            "Inspect source shape/filters."
        )
        return 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else OUT_DIR / f"pmc_curated_{run_id}.json"
    report_path = (
        Path(args.report_output)
        if args.report_output
        else OUT_DIR / f"pmc_quality_report_{run_id}.json"
    )

    out_payload = {
        "_header": {
            "source": "pmc",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "input_file": str(in_file.relative_to(PROJECT_ROOT)),
            "schema_version": "1.0",
        },
        "data": {"articles": curated},
    }
    report_payload = {
        "source": "pmc",
        "input_file": str(in_file.relative_to(PROJECT_ROOT)),
        "input_rows": input_rows,
        "valid_rows": valid_rows,
        "dropped_no_key": dropped_no_key,
        "dropped_empty_content": dropped_empty_content,
        "duplicates_removed": duplicates_removed,
        "valid_ratio": round(valid_ratio, 6),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    out_path.write_text(json.dumps(out_payload, indent=2))
    report_path.write_text(json.dumps(report_payload, indent=2))

    print(f"Input: {in_file}")
    print(f"Curated: {out_path}")
    print(f"Report: {report_path}")
    print(
        "Stats: "
        f"input={input_rows}, valid={valid_rows}, drop_no_key={dropped_no_key}, "
        f"drop_empty={dropped_empty_content}, deduped={duplicates_removed}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
