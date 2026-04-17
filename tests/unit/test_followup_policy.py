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
    q, reason, choices = next_question(encounter, max_turns=5)
    assert "highest documented" not in q.lower()
    assert reason in {"default", "differential_narrowing", "information_gain_rule"}
    if reason == "information_gain_rule":
        assert choices is not None and len(choices) > 0
    else:
        assert choices is None


def test_skip_questions_falls_through():
    encounter = {
        "symptoms": [{"symptom": "fever"}],
        "qa_history": [],
    }
    first, _, _ = next_question(encounter, max_turns=5)
    second, reason, ch2 = next_question(encounter, max_turns=5, skip_questions=(first,))
    assert second != first
    assert reason in {"information_gain_rule", "differential_narrowing", "default", "default_after_skip"}
    if reason == "information_gain_rule":
        assert ch2 is not None


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
    q, reason, choices = next_question(encounter, max_turns=5)
    assert "Condition A" in q and "Condition B" in q
    assert reason == "differential_narrowing"
    assert choices is None


def test_chest_pain_rule_returns_answer_choices():
    encounter = {
        "symptoms": [{"symptom": "chest pain"}],
        "qa_history": [],
    }
    q, reason, choices = next_question(encounter, max_turns=5)
    assert reason == "information_gain_rule"
    assert "pleuritic" in q.lower()
    assert choices is not None
    assert "Pleuritic" in choices
    assert "Exertional" in choices
    assert "Pressure-like" in choices
