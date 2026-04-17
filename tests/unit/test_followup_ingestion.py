from src.clinical_workflow import _extract_affirmed_followup_terms


def test_extract_affirmed_followup_terms_skips_negated_findings():
    text = (
        "Photophobia present. No neck stiffness, no confusion, "
        "no focal weakness, and no thunderclap onset."
    )
    terms = _extract_affirmed_followup_terms(text, max_terms=20)
    joined = " | ".join(terms)
    assert "photophobia" in joined
    assert "neck stiffness" not in joined
    assert "confusion" not in joined
    assert "focal weakness" not in joined
    assert "thunderclap onset" not in joined


def test_extract_affirmed_followup_terms_keeps_specific_phrases():
    text = "Headache is unilateral throbbing with visual aura and nausea."
    terms = _extract_affirmed_followup_terms(text, max_terms=20)
    joined = " | ".join(terms)
    assert "visual aura" in joined
    assert "unilateral" in joined
    assert "throbbing" in joined
    assert "nausea" in joined
