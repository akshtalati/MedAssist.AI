from src.followup_policy import next_question


def test_followup_policy_avoids_repeat():
    encounter = {
        "symptoms": [{"symptom": "fever"}],
        "qa_history": [
            {
                "turn_no": 1,
                "question": (
                    "What is the highest documented temperature, and have there been rigors, "
                    "night sweats, or focal infection symptoms?"
                ),
            }
        ],
    }
    q, reason = next_question(encounter, max_turns=5)
    assert "highest documented" not in q.lower()
    assert reason in {"default", "differential_narrowing", "information_gain_rule"}


def test_skip_questions_falls_through():
    encounter = {
        "symptoms": [{"symptom": "fever"}],
        "qa_history": [],
    }
    first, _ = next_question(encounter, max_turns=5)
    second, reason = next_question(encounter, max_turns=5, skip_questions=(first,))
    assert second != first
    assert reason in {"information_gain_rule", "differential_narrowing", "default", "default_after_skip"}


def test_differential_narrowing_when_no_symptom_rule():
    """Symptom not in QUESTION_RULES → use top two differential names."""
    encounter = {
        "symptoms": [{"symptom": "malaise"}],
        "qa_history": [],
        "differential": [
            {"iteration": 1, "rank_no": 1, "disease_name": "Condition A"},
            {"iteration": 1, "rank_no": 2, "disease_name": "Condition B"},
        ],
    }
    q, reason = next_question(encounter, max_turns=5)
    assert "Condition A" in q and "Condition B" in q
    assert reason == "differential_narrowing"
