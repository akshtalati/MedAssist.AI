#!/usr/bin/env python3
"""Doctor-side Streamlit app for MedAssist.AI."""

from __future__ import annotations

import re
from typing import Any

import requests
import streamlit as st

from src.indexing import SymptomIndex

COMMON_SYMPTOMS = [
    "fever",
    "vomiting",
    "nausea",
    "headache",
    "abdominal pain",
    "chest pain",
    "shortness of breath",
    "fatigue",
    "weight loss",
    "joint pain",
    "rash",
    "cough",
    "diarrhea",
    "seizure",
    "vision changes",
    "dizziness",
    "palpitations",
    "edema",
    "syncope",
    "confusion",
]


RED_FLAG_RULES = {
    "chest pain": "Rule out acute coronary syndrome, PE, and aortic dissection urgently.",
    "shortness of breath": "Check oxygen saturation and evaluate for respiratory failure, PE, or sepsis.",
    "seizure": "Assess airway protection, glucose, and urgent neuro causes.",
    "syncope": "Evaluate for arrhythmia, hemorrhage, and neurologic emergency.",
    "confusion": "Assess for delirium causes: sepsis, metabolic derangement, intoxication, stroke.",
    "vision changes": "Rule out stroke, acute glaucoma, and temporal arteritis where relevant.",
    "fever": "Screen for sepsis when paired with hypotension, tachycardia, or altered mental status.",
}


DIFFERENTIAL_RULES = [
    (
        {"fever", "vomiting", "abdominal pain"},
        ["Acute gastroenteritis", "Appendicitis", "Pancreatitis", "Cholangitis", "Pyelonephritis"],
    ),
    (
        {"headache", "vision changes"},
        ["Migraine with aura", "Idiopathic intracranial hypertension", "Temporal arteritis", "Intracranial bleed"],
    ),
    (
        {"cough", "fever", "shortness of breath"},
        ["Community-acquired pneumonia", "Influenza/COVID-19", "Pulmonary embolism", "Acute heart failure"],
    ),
    (
        {"fatigue", "weight loss"},
        ["Malignancy", "Hyperthyroidism", "Chronic infection", "Inflammatory disease", "Depression"],
    ),
]


def inject_theme() -> None:
    st.markdown(
        """
        <style>
            /* Hide deploy button, record screen, and main menu extras */
            [data-testid="stToolbar"] { display: none !important; }
            #MainMenu { visibility: hidden; }
            header { visibility: hidden; }
            footer { visibility: hidden; }

            .block-container { padding-top: 1.2rem; max-width: 900px; }
            .hero { margin-bottom: 0.8rem; }

            /* Light mode */
            @media (prefers-color-scheme: light) {
                .stApp {
                    background:
                        radial-gradient(circle at 10% 10%, #fef4e8 0%, transparent 35%),
                        radial-gradient(circle at 90% 5%, #e2f3eb 0%, transparent 30%),
                        linear-gradient(180deg, #f8f6f2 0%, #f3efe8 100%);
                    color: #1b1f24;
                }
                .card {
                    background: #fbfaf7;
                    border: 1px solid #d6d0c6;
                    border-radius: 12px;
                    padding: 0.7rem 0.85rem;
                    margin: 0.35rem 0;
                }
                .danger {
                    border-left: 5px solid #ad2f45;
                    background: #fff4f6;
                    padding: 0.7rem 0.9rem;
                    border-radius: 10px;
                    margin: 0.45rem 0;
                }
                .ok {
                    border-left: 5px solid #2f7f5f;
                    background: #f2fbf6;
                    padding: 0.7rem 0.9rem;
                    border-radius: 10px;
                    margin: 0.45rem 0;
                }
            }

            /* Dark mode */
            @media (prefers-color-scheme: dark) {
                .stApp {
                    background:
                        radial-gradient(circle at 10% 10%, #1a2332 0%, transparent 35%),
                        radial-gradient(circle at 90% 5%, #162420 0%, transparent 30%),
                        linear-gradient(180deg, #0e1117 0%, #131720 100%);
                    color: #e6edf3;
                }
                .card {
                    background: #161b22;
                    border: 1px solid #30363d;
                    border-radius: 12px;
                    padding: 0.7rem 0.85rem;
                    margin: 0.35rem 0;
                    color: #e6edf3;
                }
                .danger {
                    border-left: 5px solid #f85149;
                    background: #2d1215;
                    padding: 0.7rem 0.9rem;
                    border-radius: 10px;
                    margin: 0.45rem 0;
                    color: #f0b8b8;
                }
                .ok {
                    border-left: 5px solid #3fb950;
                    background: #12261e;
                    padding: 0.7rem 0.9rem;
                    border-radius: 10px;
                    margin: 0.45rem 0;
                    color: #a3d9b1;
                }
            }

            /* Streamlit-specific dark mode override (uses data attribute) */
            [data-testid="stAppViewContainer"][data-theme="dark"] .card,
            .stApp[data-testid="stAppViewContainer"].st-emotion-cache-dark .card {
                background: #161b22;
                border-color: #30363d;
                color: #e6edf3;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_symptom_index() -> SymptomIndex:
    return SymptomIndex()


def parse_custom_symptoms(raw: str) -> list[str]:
    if not raw.strip():
        return []
    parts = re.split(r"[,\n;]+", raw)
    return [p.strip().lower() for p in parts if p.strip()]


def build_case_text(
    age: int,
    sex: str,
    history: str,
    symptoms: list[str],
    meds: str,
    allergies: str,
    notes: str,
) -> str:
    symptom_text = ", ".join(symptoms) if symptoms else "none provided"
    return (
        f"Patient context:\n"
        f"- Age: {age}\n"
        f"- Sex: {sex}\n"
        f"- Symptoms: {symptom_text}\n"
        f"- Medical history: {history or 'not provided'}\n"
        f"- Current medications: {meds or 'not provided'}\n"
        f"- Allergies: {allergies or 'not provided'}\n"
        f"- Additional notes: {notes or 'not provided'}\n"
        f"\nQuestion: Give differential diagnosis, red flags, and next steps."
    )


def red_flags_for(symptoms: list[str], notes: str) -> list[str]:
    out: list[str] = []
    lowered = " ".join(symptoms).lower() + " " + notes.lower()
    for key, message in RED_FLAG_RULES.items():
        if key in lowered:
            out.append(f"{key.title()}: {message}")
    if any(x in lowered for x in ["hypotension", "bp 80", "bp 90", "tachycardia"]):
        out.append("Hemodynamic instability markers noted. Prioritize urgent stabilization.")
    return out


def local_differentials(symptoms: list[str]) -> list[str]:
    present = set(symptoms)
    picked: list[str] = []
    for symptom_set, diffs in DIFFERENTIAL_RULES:
        if symptom_set.issubset(present):
            picked.extend(diffs)
    if not picked and present:
        picked = [
            "Infectious causes",
            "Inflammatory/autoimmune causes",
            "Medication-related adverse effects",
            "Endocrine/metabolic causes",
            "Neurologic or cardiopulmonary causes",
        ]
    seen = set()
    deduped = []
    for d in picked:
        if d not in seen:
            seen.add(d)
            deduped.append(d)
    return deduped[:8]


def query_rare_diseases(symptoms: list[str], max_rows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = load_symptom_index()
    any_matches = index.query(symptoms, match_all=False) if symptoms else []
    all_matches = index.query(symptoms, match_all=True) if symptoms else []
    return all_matches[:max_rows], any_matches[:max_rows]


def ask_llm(api_url: str, question: str, brief: bool) -> dict:
    url = api_url.rstrip("/") + "/ask"
    payload = {"question": question, "brief": brief}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return {
        "answer": (data.get("answer") or "").strip(),
        "provider": data.get("provider", "unknown"),
    }


def ask_cortex(api_url: str, question: str) -> str:
    url = api_url.rstrip("/") + "/ask-cortex"
    payload = {"question": question}
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return (r.json().get("answer") or "").strip()


def ask_both(api_url: str, question: str, brief: bool) -> dict:
    url = api_url.rstrip("/") + "/ask-both"
    payload = {"question": question, "brief": brief}
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return {
        "gemini": (data.get("answer_gemini") or "").strip(),
        "cortex": (data.get("answer_cortex") or "").strip(),
    }


def api_start_encounter(api_url: str, payload: dict) -> str:
    url = api_url.rstrip("/") + "/encounters/start"
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["encounter_id"]


def api_initial_assessment(api_url: str, encounter_id: str, mode: str = "fast") -> dict:
    endpoint = "assess-deep" if mode == "deep" else "assess-fast"
    url = api_url.rstrip("/") + f"/encounters/{encounter_id}/{endpoint}"
    r = requests.post(url, timeout=180)
    r.raise_for_status()
    return r.json()


def api_next_question(api_url: str, encounter_id: str) -> dict:
    url = api_url.rstrip("/") + f"/encounters/{encounter_id}/next-question"
    r = requests.post(url, timeout=120)
    r.raise_for_status()
    return r.json()


def api_answer_question(api_url: str, encounter_id: str, turn_no: int, answer: str) -> dict:
    url = api_url.rstrip("/") + f"/encounters/{encounter_id}/answer"
    r = requests.post(url, json={"turn_no": turn_no, "answer": answer}, timeout=180)
    r.raise_for_status()
    return r.json()


def api_get_context(api_url: str, encounter_id: str) -> dict:
    url = api_url.rstrip("/") + f"/encounters/{encounter_id}/context"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.json()


def main() -> None:
    st.set_page_config(
        page_title="MedAssist.AI Doctor Console",
        page_icon="stethoscope",
        layout="wide",
    )
    inject_theme()

    st.markdown(
        """
        <div class="hero">
          <h3 style="margin:0;">MedAssist.AI • Minimal Doctor View</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("intake_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=0, max_value=120, value=35)
        with c2:
            sex = st.selectbox("Sex", ["Female", "Male", "Other / Not specified"])
        symptoms_pick = st.multiselect("Symptoms", COMMON_SYMPTOMS)
        symptoms_custom = st.text_input("Additional symptoms (comma separated)")
        pre_existing_conditions = st.text_input("Pre-existing conditions (comma separated)")
        history = st.text_area("History and key exam findings", height=100)
        medications = st.text_input("Current medications (comma separated)")
        allergies = st.text_input("Allergies (comma separated)")
        api_url = st.text_input("API URL", value="http://127.0.0.1:8000")
        assessment_mode = st.selectbox("Assessment mode", ["fast", "deep"], index=0, help="Fast is deterministic/low-latency; Deep uses Cortex.")
        submitted = st.form_submit_button("Generate")

    if submitted:
        symptoms = sorted(set([s.lower() for s in symptoms_pick] + parse_custom_symptoms(symptoms_custom)))
        encounter_payload = {
            "age": int(age),
            "sex": sex,
            "known_conditions": parse_custom_symptoms(pre_existing_conditions),
            "medications": parse_custom_symptoms(medications),
            "allergies": parse_custom_symptoms(allergies),
            "history_summary": history,
            "symptoms": [{"symptom": s} for s in symptoms],
        }
        encounter_id = None
        assessment = None
        try:
            encounter_id = api_start_encounter(api_url, encounter_payload)
            assessment = api_initial_assessment(api_url, encounter_id, mode=assessment_mode)
        except Exception as exc:
            st.error(f"Encounter workflow API failed: {exc}")
        st.session_state["case"] = {
            "symptoms": symptoms,
            "case_text": build_case_text(age, sex, history, symptoms, "", "", ""),
            "red_flags": red_flags_for(symptoms, history),
            "local_dx": local_differentials(symptoms),
            "all_matches": query_rare_diseases(symptoms, max_rows=8)[0],
            "api_url": api_url,
            "encounter_id": encounter_id,
            "encounter_assessment": assessment,
            "latest_question": None,
            "latest_turn_no": None,
        }

    if "case" not in st.session_state:
        st.info("Fill the intake form and click **Generate**.")
        return

    case = st.session_state["case"]
    symptoms = case["symptoms"]
    case_text = case["case_text"]
    red_flags = case["red_flags"]
    local_dx = case["local_dx"]
    all_matches = case["all_matches"]

    st.subheader("Clinical Recommendations")
    st.markdown("<div class='card'><b>Initial Differential</b></div>", unsafe_allow_html=True)
    if local_dx:
        for item in local_dx[:5]:
            st.markdown(f"- {item}")
    else:
        st.write("No symptom-derived differential yet.")

    st.markdown("<div class='card'><b>Red Flags</b></div>", unsafe_allow_html=True)
    if red_flags:
        for rf in red_flags[:4]:
            st.markdown(f"<div class='danger'>{rf}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='ok'>No rule-based red flags detected.</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><b>Rare Disease Hints (Strict Match)</b></div>", unsafe_allow_html=True)
    if all_matches:
        st.dataframe(all_matches[:8], use_container_width=True, hide_index=True)
    else:
        st.write("No strict rare-disease matches.")

    with st.expander("AI-Powered Recommendations", expanded=True):
        api_url = st.text_input("API URL", value=case.get("api_url", "http://127.0.0.1:8000"))
        ai_mode = st.radio(
            "AI Provider",
            ["Auto (Cortex → Gemini fallback)", "Cortex Only (MedRAG)", "Gemini Only", "Compare Both"],
            horizontal=True,
        )

        if st.button("Generate AI Recommendation"):
            if ai_mode == "Compare Both":
                with st.spinner("Querying Gemini and Cortex..."):
                    try:
                        results = ask_both(api_url=api_url, question=case_text, brief=True)
                        col_g, col_c = st.columns(2)
                        with col_g:
                            st.markdown("<div class='card'><b>Gemini (Vertex AI)</b></div>", unsafe_allow_html=True)
                            if results["gemini"]:
                                st.markdown(results["gemini"])
                            else:
                                st.warning("Gemini returned an empty answer.")
                        with col_c:
                            st.markdown("<div class='card'><b>Cortex (Snowflake)</b></div>", unsafe_allow_html=True)
                            if results["cortex"]:
                                st.markdown(results["cortex"])
                            else:
                                st.warning("Cortex returned an empty answer.")
                    except Exception as exc:
                        st.error(f"API call failed: {exc}")

            elif ai_mode == "Cortex Only (MedRAG)":
                with st.spinner("Querying Snowflake Cortex (MedRAG)..."):
                    try:
                        answer = ask_cortex(api_url=api_url, question=case_text)
                        if answer:
                            st.markdown("<div class='card'><b>Cortex MedRAG Response</b></div>", unsafe_allow_html=True)
                            st.markdown(answer)
                        else:
                            st.warning("Cortex returned an empty answer.")
                    except Exception as exc:
                        st.error(f"Cortex API call failed: {exc}")

            else:
                label = "Querying AI..." if ai_mode == "Gemini Only" else "Querying (Cortex → Gemini fallback)..."
                with st.spinner(label):
                    try:
                        result = ask_llm(api_url=api_url, question=case_text, brief=True)
                        if result["answer"]:
                            provider = result["provider"]
                            badge = "Gemini" if provider == "gemini" else "Cortex" if provider == "cortex" else provider
                            st.markdown(f"<div class='card'><b>AI Response</b> &nbsp;<code>{badge}</code></div>", unsafe_allow_html=True)
                            st.markdown(result["answer"])
                        else:
                            st.warning("API returned an empty answer.")
                    except Exception as exc:
                        st.error(f"API call failed: {exc}")

    with st.expander("Clinical Intake + Iterative Q&A (Snowflake + Cortex)", expanded=True):
        encounter_id = case.get("encounter_id")
        assessment = case.get("encounter_assessment")

        if not encounter_id:
            st.warning("Encounter session not created. Fill intake and click Generate.")
        else:
            st.write(f"**Encounter ID:** `{encounter_id}`")
            if assessment:
                st.write(f"**Provider:** `{assessment.get('provider_used') or assessment.get('provider')}`")
                if assessment.get("degraded_mode"):
                    st.write(f"**Degraded mode:** `{assessment.get('degraded_mode')}`")
                st.markdown("### Initial Assessment")
                st.markdown(assessment.get("assessment", ""))
                top_candidates = assessment.get("top_candidates") or []
                if top_candidates:
                    st.markdown("### Ranked candidates")
                    st.dataframe(top_candidates, use_container_width=True, hide_index=True)

            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("Generate next question"):
                    try:
                        nq = api_next_question(api_url, encounter_id)
                        st.session_state["case"]["latest_question"] = nq.get("question")
                        st.session_state["case"]["latest_turn_no"] = nq.get("turn_no")
                    except Exception as exc:
                        st.error(f"Failed to generate next question: {exc}")

            latest_q = st.session_state["case"].get("latest_question")
            latest_turn = st.session_state["case"].get("latest_turn_no")
            if latest_q:
                st.markdown(f"**Q{latest_turn}:** {latest_q}")
                answer_text = st.text_area("Doctor answer", key=f"ans_{encounter_id}_{latest_turn}")
                if st.button("Submit answer and update differential"):
                    if not answer_text.strip():
                        st.warning("Please enter an answer.")
                    else:
                        try:
                            upd = api_answer_question(api_url, encounter_id, int(latest_turn), answer_text.strip())
                            st.session_state["case"]["encounter_assessment"] = upd
                            st.success("Updated assessment generated.")
                            st.markdown(upd.get("assessment", ""))
                            if upd.get("top_candidates"):
                                st.dataframe(upd["top_candidates"], use_container_width=True, hide_index=True)
                        except Exception as exc:
                            st.error(f"Failed to submit answer: {exc}")

            if st.button("Refresh full context"):
                try:
                    ctx = api_get_context(api_url, encounter_id)
                    st.session_state["case"]["context_snapshot"] = ctx
                except Exception as exc:
                    st.error(f"Failed to fetch context: {exc}")

            if st.session_state["case"].get("context_snapshot"):
                with st.expander("Full persisted context", expanded=False):
                    st.json(st.session_state["case"]["context_snapshot"])

    st.caption(
        "For clinical support only. This tool does not replace physician judgment, protocol-based care, or emergency escalation."
    )


if __name__ == "__main__":
    main()
