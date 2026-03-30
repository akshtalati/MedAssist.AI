from src.followup_policy import next_question


def test_followup_policy_avoids_repeat():
    encounter = {
        "symptoms": [{"symptom": "fever"}],
        "qa_history": [{"turn_no": 1, "question": "What is the highest recorded temperature and any rigors/chills?"}],
    }
    q, reason = next_question(encounter, max_turns=5)
    assert "highest recorded temperature" not in q.lower()
    assert reason in {"information_gain_rule", "default"}

