"""Follow-up question policy and safety rail helpers."""

from __future__ import annotations

from typing import Any


MAX_TURNS_DEFAULT = 25

# Ordered: symptom keyword → question + short UI answer choices (Streamlit adds "None of the above").
# Choices are disjoint labels persisted as the clinician answer string prefix "Selected: …".
QUESTION_RULES: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "fever",
        "What is the highest documented temperature, and have there been rigors, night sweats, or focal infection symptoms?",
        (
            "Documented fever with rigors or night sweats",
            "Fever without rigors or night sweats",
            "No documented fever / subjective only",
            "Unknown temperature / not measured",
        ),
    ),
    (
        "headache",
        "Is there neck stiffness, photophobia, confusion, focal weakness, or a thunderclap onset?",
        (
            "Neck stiffness and/or photophobia",
            "Thunderclap / sudden severe onset",
            "Focal weakness or confusion",
            "None of those features",
            "Mixed / unsure",
        ),
    ),
    (
        "vomiting",
        "Is emesis bilious or bloody, and is the patient tolerating oral fluids?",
        (
            "Bilious emesis",
            "Bloody emesis",
            "Non-bilious, tolerating sips",
            "Cannot keep fluids down",
            "Unknown / mixed",
        ),
    ),
    (
        "nausea",
        "Any associated abdominal pain, recent medications or toxins, and can they keep fluids down?",
        (
            "With abdominal pain",
            "Recent new medication or toxin concern",
            "Keeping fluids down",
            "Cannot keep fluids down",
            "None of those / unsure",
        ),
    ),
    (
        "chest pain",
        "Is the pain pleuritic, exertional, or pressure-like, and are there associated diaphoresis or radiation?",
        (
            "Pleuritic",
            "Exertional",
            "Pressure-like",
            "Diaphoresis or radiation",
            "Mixed pattern / unsure",
        ),
    ),
    (
        "shortness of breath",
        "What are the current SpO₂ and respiratory rate at rest, and any orthopnea or pleuritic pain?",
        (
            "SpO₂ and RR documented at rest",
            "Orthopnea present",
            "Pleuritic component",
            "Vitals not yet available",
            "Mixed / other",
        ),
    ),
    (
        "abdominal pain",
        "Where is the pain localized, is peritoneal irritation suggested, and any obstipation or melena?",
        (
            "Localized pain (can point to quadrant)",
            "Diffuse pain",
            "Peritoneal signs suspected",
            "Obstipation or melena",
            "None of those / unsure",
        ),
    ),
    (
        "seizure",
        "Was there witnessed tonic-clonic activity, postictal confusion, tongue bite, or first-ever seizure at this age?",
        (
            "Witnessed tonic-clonic activity",
            "Postictal confusion",
            "Tongue bite reported",
            "First-ever seizure at this age",
            "None of those / unsure",
        ),
    ),
    (
        "cough",
        "Is the cough productive, any hemoptysis, and duration acute vs subacute vs chronic?",
        (
            "Productive cough",
            "Hemoptysis",
            "Acute duration (<3 weeks)",
            "Subacute or chronic",
            "Unknown / mixed",
        ),
    ),
    (
        "rash",
        "Describe morphology and distribution, pruritus, mucosal involvement, and recent new drugs or exposures?",
        (
            "New drug or exposure in timeline",
            "Pruritic rash",
            "Mucosal involvement",
            "Morphology/distribution documented",
            "None of those yet / unsure",
        ),
    ),
    (
        "fatigue",
        "Is fatigue acute vs chronic, any weight loss, night sweats, or bleeding to suggest malignancy or anemia?",
        (
            "Acute fatigue",
            "Chronic fatigue",
            "Weight loss or night sweats",
            "Bleeding symptoms",
            "None of those / unsure",
        ),
    ),
    (
        "dizziness",
        "Is vertigo true spinning vs presyncope, and any focal neuro signs or recent otologic symptoms?",
        (
            "True spinning vertigo",
            "Presyncope / lightheadedness",
            "Focal neuro signs",
            "Recent otologic symptoms",
            "None of those / unsure",
        ),
    ),
    (
        "vision changes",
        "Monocular vs binocular, sudden vs gradual, and any eye pain or neurologic deficits?",
        (
            "Monocular",
            "Binocular",
            "Sudden onset",
            "Gradual onset",
            "Eye pain or neuro deficits",
            "Mixed / unsure",
        ),
    ),
    (
        "confusion",
        "Acute change from baseline, attention deficits, infection exposure, toxins, or metabolic triggers?",
        (
            "Acute change from baseline",
            "Infection exposure concern",
            "Toxin or substance concern",
            "Metabolic trigger suspected",
            "None of those / unsure",
        ),
    ),
    (
        "joint pain",
        "Pattern (mono vs poly), symmetry, morning stiffness, and any recent infection or travel?",
        (
            "Monoarticular",
            "Polyarticular",
            "Symmetric pattern",
            "Morning stiffness prominent",
            "Recent infection or travel",
            "None of those / unsure",
        ),
    ),
    (
        "diarrhea",
        "Stool frequency, blood or mucus, recent antibiotics, travel, or sick contacts?",
        (
            "Blood or mucus in stool",
            "Recent antibiotics",
            "Travel or sick contacts",
            "High stool frequency",
            "None of those / unsure",
        ),
    ),
    (
        "syncope",
        "Prodrome, exertional trigger, cardiac history, and any injury from the fall?",
        (
            "Prodrome reported",
            "Exertional trigger",
            "Significant cardiac history",
            "Injury from fall",
            "None of those / unsure",
        ),
    ),
    (
        "palpitations",
        "Regular vs irregular, sustained episodes, associated chest pain or syncope?",
        (
            "Regular rhythm sensation",
            "Irregular rhythm sensation",
            "Sustained episodes",
            "With chest pain or syncope",
            "None of those / unsure",
        ),
    ),
]


def _latest_differential_rows(differential: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not differential:
        return []
    iterations = [int(d.get("iteration") or 0) for d in differential]
    max_it = max(iterations)
    rows = [d for d in differential if int(d.get("iteration") or 0) == max_it]
    rows.sort(key=lambda x: int(x.get("rank_no") or 999))
    return rows


def _already_asked(asked_blob: str, question: str) -> bool:
    """Rough dedupe: avoid repeating the same intent if a similar question was asked."""
    qn = (question or "").lower().strip()
    if len(qn) < 12:
        return qn in asked_blob
    # First sentence / clause often enough to detect repeat
    head = qn[: min(120, len(qn))]
    return head in asked_blob


def _matches_skip(question: str, skip_questions: tuple[str, ...]) -> bool:
    qn = (question or "").strip().lower()
    if not qn or not skip_questions:
        return False
    for sk in skip_questions:
        sn = (sk or "").strip().lower()
        if not sn:
            continue
        if qn == sn:
            return True
        head = 72
        if len(sn) >= 12 and (qn[:head] == sn[:head] or sn[:head] in qn or qn[:head] in sn):
            return True
    return False


def next_question(
    encounter: dict[str, Any],
    max_turns: int = MAX_TURNS_DEFAULT,
    skip_questions: tuple[str, ...] = (),
) -> tuple[str, str, tuple[str, ...] | None]:
    qa_history = encounter.get("qa_history", [])
    if len(qa_history) >= max_turns:
        return (
            "Maximum follow-up turns reached for this encounter. Summarize and consider disposition or further workup.",
            "max_turns",
            None,
        )

    symptoms = [str(s.get("symptom", "")).lower() for s in encounter.get("symptoms", [])]
    asked_text = " ".join((q.get("question") or "").lower() for q in qa_history)

    for trigger, question, choices in QUESTION_RULES:
        if trigger in symptoms and not _already_asked(asked_text, question):
            if not _matches_skip(question, skip_questions):
                return question, "information_gain_rule", choices

    diff_rows = _latest_differential_rows(encounter.get("differential") or [])
    names: list[str] = []
    for row in diff_rows:
        name = row.get("disease_name")
        if name and name not in names:
            names.append(name)
        if len(names) >= 3:
            break

    if len(names) >= 2:
        q = (
            f'Between “{names[0]}” and “{names[1]}” (from the current ranked differentials), '
            "what single history detail, vital sign, or focused exam finding would most help discriminate?"
        )
        if not _already_asked(asked_text, q) and not _matches_skip(q, skip_questions):
            return q, "differential_narrowing", None

    if len(names) == 1:
        q = (
            f'For the leading consideration “{names[0]}”, what critical data is still missing '
            "(timeline, exposures, vitals, medications, pregnancy status, or targeted exam) that would change management?"
        )
        if not _already_asked(asked_text, q) and not _matches_skip(q, skip_questions):
            return q, "differential_narrowing", None

    default_q = (
        "In one minute: which symptom began first, how has severity changed over the last 24–48 hours, "
        "and what is the patient’s baseline functional status?"
    )
    if not _matches_skip(default_q, skip_questions):
        return default_q, "default", None

    return (
        "What are the patient’s current vital signs, and have there been any new neuro, chest, or abdominal red flags since intake?",
        "default_after_skip",
        None,
    )


def safety_rails(encounter: dict[str, Any], confidence: float) -> dict[str, list[str] | str]:
    symptoms = " ".join(str(s.get("symptom", "")).lower() for s in encounter.get("symptoms", []))
    meds_l = " ".join(str(m).lower() for m in (encounter.get("medications") or []))
    hist_l = (encounter.get("history_summary") or "").lower()
    red_flags: list[str] = []
    if "chest pain" in symptoms:
        red_flags.append("Rule out acute coronary syndrome, pulmonary embolism, and aortic syndromes urgently.")
    if "shortness of breath" in symptoms:
        red_flags.append("Assess oxygenation and work of breathing immediately.")
    if "confusion" in symptoms or "seizure" in symptoms:
        red_flags.append("Consider urgent neurologic and metabolic evaluation.")
    if "headache" in symptoms and ("contracept" in meds_l or "oral contrace" in hist_l or "ocp" in meds_l):
        red_flags.append(
            "Headache in a patient on estrogen-containing contraceptives: consider venous sinus thrombosis / "
            "IIH-related differentials and urgent evaluation when papilledema or focal signs are present."
        )

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
