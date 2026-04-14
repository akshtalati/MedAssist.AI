#!/usr/bin/env python3
"""HTTP eval runner: start encounter + assess-fast; score top-k hint overlap (optional live API)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import requests

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "synthetic_cases.json"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("cases") or [])


def _hint_hit(candidates: list[dict[str, Any]], hints: list[str], k: int) -> bool:
    top = candidates[:k]
    blob = " ".join(
        f"{(c.get('disease_name') or '')} {(c.get('disease_code') or '')}".lower() for c in top
    )
    return any(h.lower() in blob for h in hints if h)


def run_suite(base_url: str, cases: list[dict[str, Any]], *, top_k: int = 3) -> dict[str, Any]:
    base = base_url.rstrip("/")
    latencies: list[float] = []
    hits1 = 0
    hitsk = 0
    n = 0
    details: list[dict[str, Any]] = []

    for case in cases:
        cid = case.get("id", "?")
        hints = list(case.get("expected_hints") or [])
        start_body = case.get("start") or {}
        n += 1
        t0 = time.perf_counter()
        r0 = requests.post(f"{base}/encounters/start", json=start_body, timeout=120)
        if r0.status_code != 200:
            details.append({"id": cid, "error": f"start {r0.status_code}", "body": r0.text[:200]})
            continue
        eid = r0.json().get("encounter_id")
        r1 = requests.post(f"{base}/encounters/{eid}/assess-fast", timeout=120)
        latencies.append(time.perf_counter() - t0)
        if r1.status_code != 200:
            details.append({"id": cid, "error": f"assess {r1.status_code}", "body": r1.text[:200]})
            continue
        body = r1.json()
        cands = list(body.get("top_candidates") or [])
        h1 = _hint_hit(cands, hints, 1)
        hk = _hint_hit(cands, hints, top_k)
        if h1:
            hits1 += 1
        if hk:
            hitsk += 1
        details.append(
            {
                "id": cid,
                "top1_hit": h1,
                f"top{top_k}_hit": hk,
                "fallback_mode": body.get("fallback_mode"),
                "latency_s": round(latencies[-1], 3),
            }
        )

    return {
        "n_cases": n,
        "top1_accuracy": hits1 / n if n else 0.0,
        f"top{top_k}_accuracy": hitsk / n if n else 0.0,
        "latency_median_s": float(statistics.median(latencies)) if latencies else None,
        "details": details,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Run MedAssist synthetic evals against a live API.")
    p.add_argument(
        "--base-url",
        default=os.environ.get("MEDASSIST_API_BASE", "http://127.0.0.1:8000"),
        help="API base URL (or MEDASSIST_API_BASE)",
    )
    p.add_argument("--fixtures", type=Path, default=FIXTURE_PATH)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()
    cases = _load_cases(args.fixtures)
    if not cases:
        print("No cases in fixtures.", file=sys.stderr)
        return 2
    report = run_suite(args.base_url, cases, top_k=args.top_k)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
