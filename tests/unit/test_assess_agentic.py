import pytest
from fastapi.testclient import TestClient

import api.main as main


def test_assess_agentic_adds_trace(monkeypatch):
    pytest.importorskip("langgraph")

    class _FakeGraph:
        def invoke(self, state):
            return {
                "result": {
                    "encounter_id": state["encounter_id"],
                    "provider_used": "knowledge_graph_context",
                    "degraded_mode": "none",
                    "errors": [],
                    "contract": {},
                    "top_candidates": [],
                    "assessment": "",
                    "kg_version": "v1",
                    "kg_build_id": "b1",
                    "evidence_summary": "",
                    "evidence_sources": [],
                    "fallback_mode": "journal_first",
                },
                "agent_trace": ["langgraph:deterministic_assess"],
            }

    monkeypatch.setattr(main, "_get_agentic_assess_graph", lambda: _FakeGraph())
    client = TestClient(main.app)
    res = client.post("/encounters/enc-1/assess-agentic")
    assert res.status_code == 200
    body = res.json()
    assert body["agent_provider"] == "langgraph"
    assert "langgraph" in body["agent_trace"][0]


def test_assess_agentic_404_when_missing_encounter(monkeypatch):
    pytest.importorskip("langgraph")

    class _FakeGraph:
        def invoke(self, state):
            return {"result": None, "agent_trace": []}

    monkeypatch.setattr(main, "_get_agentic_assess_graph", lambda: _FakeGraph())
    client = TestClient(main.app)
    res = client.post("/encounters/missing/assess-agentic")
    assert res.status_code == 404
