"""Smoke tests for eval harness (offline); integration requires RUN_EVAL_API=1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.run_evals import FIXTURE_PATH, _hint_hit, _load_cases, run_suite


def test_fixtures_load():
    cases = _load_cases(FIXTURE_PATH)
    assert len(cases) >= 10
    assert all("start" in c and "expected_hints" in c for c in cases)


def test_hint_hit_matching():
    cands = [
        {"disease_name": "Rare fever syndrome", "disease_code": "ORPHA:123"},
        {"disease_name": "Other", "disease_code": ""},
    ]
    assert _hint_hit(cands, ["fever", "ORPHA"], 1) is True
    assert _hint_hit(cands, ["nomatch"], 2) is False


@pytest.mark.integration
def test_live_api_eval_smoke(tmp_path):
    import os

    import requests

    base = os.environ.get("MEDASSIST_API_BASE", "http://127.0.0.1:8000").rstrip("/")
    r = requests.get(f"{base}/docs", timeout=5)
    assert r.status_code == 200
    cases = _load_cases(FIXTURE_PATH)[:3]
    report = run_suite(base, cases, top_k=3)
    assert report["n_cases"] == 3
    (tmp_path / "last_eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
