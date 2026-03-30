"""Follow-up question policy and safety rail helpers."""

from __future__ import annotations

from typing import Any


MAX_TURNS_DEFAULT = 5


QUESTION_RULES = [
    ("fever", "What is the highest recorded temperature and any rigors/chills?"),
    ("headache", "Any neck stiffness, photophobia, confusion, or focal neurologic deficits?"),
    ("vomiting", "Is vomiting bilious/projectile and can oral intake be maintained?"),
    ("chest pain", "Is chest pain exertional, pleuritic, or associated with diaphoresis?"),
    ("shortness of breath", "What are current oxygen saturation and respiratory rate?"),
]


def next_question(encounter: dict[str, Any], max_turns: int = MAX_TURNS_DEFAULT) -> tuple[str, str]:
    qa_history = encounter.get("qa_history", [])
    if len(qa_history) >= max_turns:
        return "No further follow-up questions: max turns reached.", "max_turns"

    symptoms = [str(s.get("symptom", "")).lower() for s in encounter.get("symptoms", [])]
    asked_text = " ".join((q.get("question") or "").lower() for q in qa_history)
    for trigger, question in QUESTION_RULES:
        if trigger in symptoms and question.lower() not in asked_text:
            return question, "information_gain_rule"
    return "What symptom started first, and how has severity changed over time?", "default"


def safety_rails(encounter: dict[str, Any], confidence: float) -> dict[str, list[str] | str]:
    symptoms = " ".join(str(s.get("symptom", "")).lower() for s in encounter.get("symptoms", []))
    red_flags: list[str] = []
    if "chest pain" in symptoms:
        red_flags.append("Rule out acute coronary syndrome, pulmonary embolism, and aortic syndromes urgently.")
    if "shortness of breath" in symptoms:
        red_flags.append("Assess oxygenation and work of breathing immediately.")
    if "confusion" in symptoms or "seizure" in symptoms:
        red_flags.append("Consider urgent neurologic and metabolic evaluation.")

    contraindications: list[str] = []
    allergies = [a.lower() for a in (encounter.get("allergies") or [])]
    if "penicillin" in allergies:
        contraindications.append("Avoid penicillin-class empiric choices unless desensitization/override is indicated.")

    uncertainty_note = ""
    if confidence < 0.55:
        uncertainty_note = "Evidence is currently limited; broaden differential and ask targeted follow-up before narrowing."
    return {
        "red_flags": red_flags,
        "contraindications": contraindications,
        "uncertainty_note": uncertainty_note,
    }
