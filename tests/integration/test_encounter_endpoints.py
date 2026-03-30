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
        lambda encounter_id, limit=12: (
            [{"disease_name": "Disease A", "disease_code": "ORPHA:1", "score": 1.0, "rationale": "test", "source": "knowledge_graph"}],
            "none",
            [],
        ),
    )
    monkeypatch.setattr(main, "get_latest_kg_build_meta", lambda: ("v1", "b1", None))
    monkeypatch.setattr(main, "save_differential", lambda *a, **k: None)
    monkeypatch.setattr(main, "append_audit_row", lambda **k: None)
    client = TestClient(main.app)
    res = client.post("/encounters/enc-1/assess-fast")
    assert res.status_code == 200
    body = res.json()
    assert body["provider_used"] == "knowledge_graph_context"
    assert "contract" in body
    assert "degraded_mode" in body


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

