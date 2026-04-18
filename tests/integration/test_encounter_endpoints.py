from fastapi.testclient import TestClient

import api.main as main


def _mock_encounter():
    return {
        "encounter_id": "enc-1",
        "age": 33,
        "sex": "Female",
        "known_conditions": [],
        "medications": [],
        "allergies": ["penicillin"],
        "history_summary": "fever/vomiting",
        "symptoms": [{"symptom": "fever"}, {"symptom": "vomiting"}],
        "qa_history": [],
        "differential": [],
    }


def test_assess_fast_returns_contract_and_metadata(monkeypatch):
    monkeypatch.setattr(main, "ensure_clinical_tables", lambda: None)
    monkeypatch.setattr(main, "get_encounter", lambda encounter_id: _mock_encounter())
    monkeypatch.setattr(
        main,
        "_attempt_rank_with_degradation",
        lambda encounter_id, encounter=None, limit=12: (
            [{"disease_name": "Disease A", "disease_code": "ORPHA:1", "score": 1.0, "rationale": "test", "source": "knowledge_graph"}],
            "none",
            [],
        ),
    )
    monkeypatch.setattr(main, "get_latest_kg_build_meta", lambda: ("v1", "b1", None))
    monkeypatch.setattr(main, "save_differential", lambda *a, **k: None)
    monkeypatch.setattr(main, "append_audit_row", lambda **k: None)
    monkeypatch.setattr(
        main,
        "retrieve_evidence_journal_first",
        lambda question, limit=30: (
            [{"source": "pubmed", "title": "A", "url": "https://pubmed.ncbi.nlm.nih.gov/1/", "text": "x"}],
            "journal_first",
        ),
    )
    client = TestClient(main.app)
    res = client.post("/encounters/enc-1/assess-fast")
    assert res.status_code == 200
    body = res.json()
    assert body["provider_used"] == "knowledge_graph_context"
    assert "contract" in body
    assert "degraded_mode" in body
    assert "evidence_summary" in body
    assert "evidence_sources" in body
    assert body["fallback_mode"] == "journal_first"


def test_idempotent_start_replay(monkeypatch):
    monkeypatch.setattr(main, "ensure_clinical_tables", lambda: None)
    monkeypatch.setattr(main, "seed_graph_from_symptom_map", lambda: None)
    monkeypatch.setattr(main, "fetch_idempotent_response", lambda e, k, h: ({"encounter_id": "replayed"}, False))
    client = TestClient(main.app)
    res = client.post(
        "/encounters/start",
        headers={"Idempotency-Key": "abc"},
        json={"age": 20, "sex": "Female", "known_conditions": [], "medications": [], "allergies": [], "history_summary": "", "symptoms": []},
    )
    assert res.status_code == 200
    assert res.json()["encounter_id"] == "replayed"


def test_ask_returns_evidence_and_fallback_mode(monkeypatch):
    monkeypatch.setattr(
        main,
        "retrieve_evidence_journal_first",
        lambda question, limit=30: (
            [{"source": "pubmed", "title": "Study", "url": "https://pubmed.ncbi.nlm.nih.gov/2/", "text": "snippet"}],
            "web_assisted",
        ),
    )
    monkeypatch.setattr(main, "fetch_symptom_disease_context", lambda question, limit=50: "")
    monkeypatch.setattr(main, "cortex_complete", lambda prompt: "### Summary\nok")
    monkeypatch.setattr(
        main,
        "_evaluate_evidence_quality",
        lambda entries: {"sufficient": True, "score": 0.85, "trusted_count": 1, "count": 1},
    )
    client = TestClient(main.app)
    res = client.post("/ask", json={"question": "fever and headache", "brief": False})
    assert res.status_code == 200
    body = res.json()
    assert body["provider"] == "cortex"
    assert body["fallback_mode"] == "web_assisted"
    assert isinstance(body.get("evidence_sources"), list)


def test_ask_doctor_falls_back_to_cortex_when_agent_not_configured(monkeypatch):
    monkeypatch.setattr(main, "is_cortex_agent_ready", lambda: False)
    monkeypatch.setattr(
        main,
        "retrieve_evidence_journal_first",
        lambda question, limit=30: (
            [{"source": "pubmed", "title": "Study", "url": "https://pubmed.ncbi.nlm.nih.gov/2/", "text": "snippet"}],
            "web_assisted",
        ),
    )
    monkeypatch.setattr(main, "fetch_symptom_disease_context", lambda question, limit=50: "")
    monkeypatch.setattr(main, "cortex_complete", lambda prompt: "### Summary\nok")
    monkeypatch.setattr(
        main,
        "_evaluate_evidence_quality",
        lambda entries: {"sufficient": True, "score": 0.85, "trusted_count": 1, "count": 1},
    )
    client = TestClient(main.app)
    res = client.post("/ask-doctor", json={"question": "fever and headache", "brief": False})
    assert res.status_code == 200
    body = res.json()
    assert body["provider"] == "cortex"
    assert body.get("fallback_chain") == ["cortex"]


def test_ask_doctor_uses_agent_when_ready(monkeypatch):
    monkeypatch.setattr(main, "is_cortex_agent_ready", lambda: True)
    monkeypatch.setattr(
        main,
        "retrieve_evidence_journal_first",
        lambda question, limit=30: (
            [{"source": "pubmed", "title": "Study", "url": "https://pubmed.ncbi.nlm.nih.gov/2/", "text": "snippet"}],
            "journal_first",
        ),
    )
    monkeypatch.setattr(main, "fetch_symptom_disease_context", lambda question, limit=50: "")
    monkeypatch.setattr(
        main,
        "_evaluate_evidence_quality",
        lambda entries: {"sufficient": True, "score": 0.85, "trusted_count": 1, "count": 1},
    )
    monkeypatch.setattr(main, "agent_config_from_env", lambda: {"database": "D", "schema": "S", "agent_name": "A"})
    monkeypatch.setattr(
        main,
        "run_cortex_agent_object",
        lambda **kwargs: (
            {"role": "assistant", "content": [{"type": "text", "text": "From agent."}]},
            None,
        ),
    )
    client = TestClient(main.app)
    res = client.post("/ask-doctor", json={"question": "what is X", "brief": False})
    assert res.status_code == 200
    body = res.json()
    assert body["provider"] == "snowflake_cortex_agent"
    assert body.get("fallback_chain") == ["snowflake_cortex_agent"]

