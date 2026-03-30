from fastapi.testclient import TestClient

import api.main as main


def test_deep_path_falls_back_when_cortex_fails(monkeypatch):
    encounter = {
        "encounter_id": "enc-chaos",
        "age": 50,
        "sex": "Male",
        "known_conditions": [],
        "medications": [],
        "allergies": [],
        "history_summary": "test",
        "symptoms": [{"symptom": "fever"}],
        "qa_history": [],
        "differential": [],
    }
    monkeypatch.setattr(main, "ensure_clinical_tables", lambda: None)
    monkeypatch.setattr(main, "get_encounter", lambda encounter_id: encounter)
    monkeypatch.setattr(
        main,
        "_attempt_rank_with_degradation",
        lambda encounter_id, limit=12: (
            [{"disease_name": "A", "disease_code": "", "score": 1.0, "rationale": "r", "source": "kg"}],
            "none",
            [],
        ),
    )
    monkeypatch.setattr(main, "get_latest_kg_build_meta", lambda: ("v1", "b1", None))
    monkeypatch.setattr(main, "save_differential", lambda *a, **k: None)
    monkeypatch.setattr(main, "cortex_complete", lambda prompt: "[Cortex error: timeout]")
    monkeypatch.setattr(main, "append_audit_row", lambda **k: None)
    client = TestClient(main.app)
    res = client.post("/encounters/enc-chaos/assess-deep")
    assert res.status_code == 200
    body = res.json()
    assert body["degraded_mode"] == "no_llm"
    assert body["provider_used"] == "knowledge_graph_context"

