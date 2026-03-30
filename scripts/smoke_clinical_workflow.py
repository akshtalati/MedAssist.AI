#!/usr/bin/env python3
"""End-to-end smoke test for encounter workflow endpoints."""

from __future__ import annotations

import argparse
import json
import sys

import requests


def post(base: str, path: str, payload: dict | None = None, timeout: int = 180) -> dict:
    r = requests.post(base.rstrip("/") + path, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get(base: str, path: str, timeout: int = 180) -> dict:
    r = requests.get(base.rstrip("/") + path, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test encounter-based Cortex workflow")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="FastAPI base URL")
    args = parser.parse_args()

    start_payload = {
        "age": 33,
        "sex": "Female",
        "known_conditions": ["migraine"],
        "medications": ["ibuprofen"],
        "allergies": ["penicillin"],
        "history_summary": "2 days fever with vomiting and headache.",
        "symptoms": [
            {"symptom": "fever", "onset": "2 days", "severity": "moderate", "duration": "intermittent"},
            {"symptom": "vomiting", "onset": "1 day", "severity": "moderate", "duration": "episodic"},
            {"symptom": "headache", "onset": "2 days", "severity": "mild", "duration": "constant"},
        ],
    }

    print("1) start encounter")
    started = post(args.base, "/encounters/start", start_payload)
    encounter_id = started["encounter_id"]
    print("encounter_id=", encounter_id)

    print("2) initial assessment")
    initial = post(args.base, f"/encounters/{encounter_id}/initial-assessment")
    assert initial.get("assessment"), "initial assessment is empty"
    assert initial.get("top_candidates") is not None, "no candidates returned"
    initial_names = [c.get("disease_name") for c in (initial.get("top_candidates") or []) if c.get("disease_name")]

    print("3) follow-up loop x3")
    for i in range(3):
        nq = post(args.base, f"/encounters/{encounter_id}/next-question")
        turn_no = int(nq["turn_no"])
        question_text = nq["question"]
        assert question_text, "next question is empty"
        ans = (
            "No neck stiffness, no rash, no recent travel, no focal weakness."
            if i == 0
            else "No confusion, no syncope, oral intake reduced."
        )
        upd = post(args.base, f"/encounters/{encounter_id}/answer", {"turn_no": turn_no, "answer": ans})
        assert upd.get("assessment"), f"updated assessment missing at turn {turn_no}"

    print("4) verify full context")
    ctx = get(args.base, f"/encounters/{encounter_id}/context")
    structured = ctx.get("structured_context") or {}
    qa_hist = structured.get("qa_history") or []
    assert len(qa_hist) >= 3, "qa history did not persist all turns"
    assert "Patient profile:" in (ctx.get("context_text") or ""), "context text missing profile header"

    final_candidates = [d.get("disease_name") for d in (structured.get("differential") or []) if d.get("disease_name")]
    drift = sorted(set(initial_names) ^ set(final_candidates))

    print("\nSmoke test passed.")
    print("Initial candidate count:", len(initial_names))
    print("Final differential rows:", len(final_candidates))
    print("Candidate drift sample:", drift[:5])
    print("QA turns persisted:", len(qa_hist))
    print("\nContext snapshot:")
    print(json.dumps({"encounter_id": encounter_id, "qa_turns": len(qa_hist)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.HTTPError as exc:
        print(f"HTTP error: {exc}")
        if exc.response is not None:
            print(exc.response.text)
        raise SystemExit(1)
    except Exception as exc:
        print(f"Smoke test failed: {exc}")
        raise SystemExit(1)
