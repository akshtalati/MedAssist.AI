#!/usr/bin/env python3
"""Doctor-side Streamlit app for MedAssist.AI."""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from typing import Any
from urllib.parse import quote_plus

import requests
import streamlit as st

from src.indexing import SymptomIndex
from src.clinical_workflow import normalize_encounter_dict

DEFAULT_API_URL = os.environ.get("MEDASSIST_API_BASE", "http://127.0.0.1:8000")

# region agent log
_DEBUG_LOG_PATH = "/Users/akshtalati/Desktop/genai/MedAssist.AI/.cursor/debug-0a57e9.log"


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    try:
        payload = {
            "sessionId": "0a57e9",
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


# endregion

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
        "fallback_mode": data.get("fallback_mode"),
        "evidence_summary": data.get("evidence_summary"),
        "evidence_sources": data.get("evidence_sources") or [],
        "evidence_quality": data.get("evidence_quality") or {},
        "insufficient_evidence": bool(data.get("insufficient_evidence")),
        "follow_up_questions": data.get("follow_up_questions") or [],
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


def api_get_kg_preview(
    api_url: str,
    encounter_id: str,
    *,
    max_edges: int = 2500,
) -> dict[str, Any]:
    url = api_url.rstrip("/") + f"/encounters/{encounter_id}/kg-preview"
    r = requests.get(
        url,
        params={"max_edges": max_edges},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def disease_read_more_url(disease_code: str | None, disease_name: str | None) -> tuple[str, str]:
    """
    Returns (url, label) where label describes the link target (Orphanet vs PubMed search).
    """
    name = (disease_name or "").strip() or "disease"
    raw = (disease_code or "").strip()
    digits = re.sub(r"\D", "", raw.replace("ORPHA", "").replace(":", ""))
    if len(digits) >= 3:
        return f"https://www.orpha.net/en/disease/detail/{digits}", "Orphanet"
    q = quote_plus(name)
    return f"https://pubmed.ncbi.nlm.nih.gov/?term={q}", "PubMed search"


def _latest_iteration_rows(differential: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not differential:
        return []
    iterations = [int(d.get("iteration") or 0) for d in differential]
    max_it = max(iterations)
    rows = [d for d in differential if int(d.get("iteration") or 0) == max_it]
    rows.sort(key=lambda x: int(x.get("rank_no") or 999))
    return rows


def differential_to_candidate_rows(differential: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    """Shape DB differential rows like API top_candidates for display."""
    out: list[dict[str, Any]] = []
    for d in _latest_iteration_rows(differential)[:limit]:
        out.append(
            {
                "disease_name": d.get("disease_name"),
                "disease_code": d.get("disease_code"),
                "score": d.get("score"),
                "rationale": d.get("rationale"),
                "source": d.get("source"),
            }
        )
    return out


def candidate_name_list(candidates: list[dict[str, Any]] | None) -> list[str]:
    if not candidates:
        return []
    return [str(c.get("disease_name") or "").strip() for c in candidates if c.get("disease_name")]


def format_candidate_diff_markdown(prev_names: list[str], new_names: list[str]) -> str:
    prev_set = {x for x in prev_names if x}
    new_set = {x for x in new_names if x}
    added = [x for x in new_names if x and x not in prev_set]
    removed = [x for x in prev_names if x and x not in new_set]
    parts: list[str] = []
    if added:
        parts.append("**New in top list:** " + ", ".join(added))
    if removed:
        parts.append("**Dropped from top list:** " + ", ".join(removed))
    if not added and not removed:
        if prev_names == new_names:
            parts.append("Top candidate *names* unchanged (same set and order).")
        else:
            parts.append("Same disease *names* in the top list; **order may have changed**.")
    return "\n\n".join(parts) if parts else "_No comparison available._"


def render_ranked_candidates_md(candidates: list[dict[str, Any]]) -> None:
    if not candidates:
        st.write("No ranked candidates.")
        return
    lines: list[str] = []
    n_orpha = 0
    n_pubmed = 0
    for i, c in enumerate(candidates, start=1):
        name = (c.get("disease_name") or "").strip() or "Unknown"
        code = c.get("disease_code")
        url, kind = disease_read_more_url(str(code) if code else None, name)
        if kind == "Orphanet":
            n_orpha += 1
        else:
            n_pubmed += 1
        score = c.get("score")
        tail = f" (score {score:.2f})" if isinstance(score, (int, float)) else ""
        lines.append(f"{i}. [{name}]({url}) — _{kind}_{tail}")
    # region agent log
    _agent_debug_log(
        "H5",
        "streamlit_app:render_ranked_candidates_md",
        "link_kind_counts",
        {"n_candidates": len(candidates), "n_orphanet": n_orpha, "n_pubmed_fallback": n_pubmed},
    )
    # endregion
    st.markdown("\n".join(lines))


def render_evidence_block(summary: str | None, sources: list[dict[str, Any]] | None) -> None:
    st.markdown("### Evidence (brief)")
    st.caption(summary or "No explicit evidence summary available.")
    rows = sources or []
    if not rows:
        st.info("No evidence links returned.")
        return
    lines: list[str] = []
    for i, row in enumerate(rows[:5], start=1):
        title = (row.get("title") or row.get("source") or "evidence").strip()
        url = (row.get("url") or "").strip()
        source = (row.get("source") or "unknown").strip()
        if url:
            lines.append(f"{i}. [{title}]({url}) — _{source}_")
        else:
            lines.append(f"{i}. {title} — _{source}_")
    st.markdown("\n".join(lines))


def render_followup_chat(messages: list[dict[str, str]]) -> None:
    if not messages:
        st.info("No follow-up chat yet.")
        return
    for msg in messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        with st.chat_message("assistant" if role == "assistant" else "user"):
            st.markdown(content or "")


def render_compact_assessment(assessment: dict[str, Any]) -> None:
    contract = (assessment or {}).get("contract") or {}
    if not contract:
        # Fallback for paths that do not return structured contract.
        st.markdown((assessment or {}).get("assessment") or "_No assessment text returned._")
        return
    st.markdown("### Clinical Snapshot")
    summary = (contract.get("summary") or "").strip()
    if summary:
        st.markdown(f"- {summary[:320]}{'...' if len(summary) > 320 else ''}")
    diff = contract.get("differential") or []
    if diff:
        items = [str(x.get("disease_name") or "").strip() for x in diff[:5] if x.get("disease_name")]
        if items:
            st.markdown("### Top Differential (shortlist)")
            for i, name in enumerate(items, start=1):
                st.markdown(f"{i}. {name}")
    red_flags = contract.get("red_flags") or []
    st.markdown("### Red Flags")
    if red_flags:
        for rf in red_flags[:4]:
            st.markdown(f"- {rf}")
    else:
        st.markdown("- No immediate rule-based red flags.")
    next_steps = contract.get("next_steps") or []
    if next_steps:
        st.markdown("### Immediate Next Steps")
        for step in next_steps[:3]:
            st.markdown(f"- {step}")


def render_qa_timeline(structured: dict[str, Any] | None) -> None:
    qa = (structured or {}).get("qa_history") or []
    if not qa:
        st.info("No follow-up Q&A recorded yet.")
        return
    for row in qa:
        turn = row.get("turn_no", "?")
        q = row.get("question") or ""
        a = row.get("answer")
        st.markdown(f"**Turn {turn}**")
        st.markdown(f"- **Question:** {q}")
        if a:
            st.markdown(f"- **Answer:** {a}")
        else:
            st.markdown("- **Answer:** _(pending)_")


def render_kg_sidebar(api_url: str, encounter_id: str) -> None:
    """All KG HAS_SYMPTOM links from this encounter’s symptoms to diseases (row cap only)."""
    with st.sidebar.expander("Knowledge graph (all matching diseases)", expanded=True):
        st.caption(
            "Every disease in the seeded graph that shares a symptom with this encounter. "
            "Raise the cap if some rows are cut off."
        )
        max_edg = st.slider(
            "Max rows (safety cap)",
            min_value=500,
            max_value=5000,
            value=2500,
            step=100,
            key="kg_max_edges",
            help="Snowflake returns up to this many disease–symptom pairs; increase to see more.",
        )
        try:
            kg = api_get_kg_preview(api_url, encounter_id, max_edges=max_edg)
        except Exception as exc:
            # region agent log
            _agent_debug_log(
                "H4",
                "streamlit_app:render_kg_sidebar",
                "kg_api_error",
                {"error": str(exc)[:200]},
            )
            # endregion
            st.warning(f"Could not load KG: {exc}")
            return
        if kg.get("note"):
            st.info(kg["note"])
        st.caption(f"`{kg.get('kg_build_id')}` · {kg.get('kg_version')}")
        edges = kg.get("edges") or []
        # region agent log
        _agent_debug_log(
            "H4",
            "streamlit_app:render_kg_sidebar",
            "kg_preview_ok",
            {
                "encounter_id": encounter_id,
                "n_edges": len(edges),
                "distinct_diseases": kg.get("distinct_diseases"),
                "truncated": bool(kg.get("truncated")),
                "has_note": bool(kg.get("note")),
            },
        )
        # endregion
        if not edges:
            st.info("No graph edges for these symptoms yet (KG may be empty or unmatched).")
            return
        nd = int(kg.get("distinct_diseases") or 0)
        st.metric("Distinct diseases", nd, help="Unique disease nodes linked to your symptoms")
        st.metric("Edges loaded", len(edges))
        if kg.get("truncated"):
            st.warning(
                f"Hit row cap ({max_edg}). Increase **Max rows** to load additional disease–symptom pairs."
            )
        st.dataframe(
            edges,
            use_container_width=True,
            hide_index=True,
            height=min(420, 120 + min(len(edges), 24) * 28),
        )
        by_disease: dict[str, list[str]] = defaultdict(list)
        for e in edges:
            d = str(e.get("disease") or "").strip()
            s = str(e.get("symptom") or "").strip()
            if d and s:
                by_disease[d].append(s)
        with st.expander("Grouped by disease (full list)", expanded=False):
            for d in sorted(by_disease.keys()):
                syms = sorted(set(by_disease[d]))
                st.markdown(f"**{d}** ({len(syms)} symptoms)")
                for s in syms:
                    st.markdown(f"- _{s}_")


def rehydrate_case_from_context(ctx: dict[str, Any], api_url: str) -> dict[str, Any]:
    structured = normalize_encounter_dict(ctx.get("structured_context") or {})
    ctx_out = {**ctx, "structured_context": structured}
    eid = ctx_out.get("encounter_id") or structured.get("encounter_id")
    symptoms_list = [str(s.get("symptom", "")).lower() for s in structured.get("symptoms") or [] if s.get("symptom")]
    age = int(structured.get("age") or 0)
    sex = str(structured.get("sex") or "Other / Not specified")
    history = str(structured.get("history_summary") or "")
    diff = structured.get("differential") or []
    top = differential_to_candidate_rows(diff)
    case_text = build_case_text(age, sex, history, symptoms_list, "", "", "")
    case_dict = {
        "symptoms": symptoms_list,
        "case_text": case_text,
        "red_flags": red_flags_for(symptoms_list, history),
        "local_dx": local_differentials(symptoms_list),
        "all_matches": query_rare_diseases(symptoms_list, max_rows=8)[0],
        "api_url": api_url,
        "encounter_id": eid,
        "encounter_assessment": {
            "assessment": ctx_out.get("context_text") or "",
            "top_candidates": top,
            "provider_used": "persisted_session",
            "degraded_mode": None,
        },
        "latest_question": None,
        "latest_turn_no": None,
        "latest_policy_reason": None,
        "context_snapshot": ctx_out,
        "last_candidate_diff": None,
        "assessment_history": [
            {
                "label": "Loaded encounter (session)",
                "assessment": ctx_out.get("context_text") or "",
                "top_candidates": top,
            }
        ],
    }
    # region agent log
    _agent_debug_log(
        "H1",
        "streamlit_app:rehydrate_case_from_context",
        "rehydrated",
        {
            "encounter_id": eid,
            "n_symptoms_structured": len(structured.get("symptoms") or []),
            "n_symptoms_case": len(symptoms_list),
            "age": age,
        },
    )
    # endregion
    return case_dict


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
          <h3 style="margin:0;">MedAssist.AI • Unified Doctor Workspace</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.setdefault("api_url", DEFAULT_API_URL)
    st.session_state.setdefault("case", None)
    st.session_state.setdefault("latest_question", None)
    st.session_state.setdefault("latest_turn_no", None)
    st.session_state.setdefault("latest_policy_reason", None)
    st.session_state.setdefault("followup_chat", [])

    tab_diff, tab_qa = st.tabs(["Differential + Evidence", "Doctor Q&A"])

    with tab_diff:
        st.subheader("Patient Intake and Differential")
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
            api_url = st.text_input("API URL", value=st.session_state["api_url"])
            submitted = st.form_submit_button("Generate Differential")

        st.session_state["api_url"] = api_url

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
            try:
                encounter_id = api_start_encounter(api_url, encounter_payload)
                # Use a single deterministic clinical path (KG-led) for consistency.
                assessment = api_initial_assessment(api_url, encounter_id, mode="fast")
                ctx0 = api_get_context(api_url, encounter_id)
                st.session_state["case"] = {
                    "encounter_id": encounter_id,
                    "assessment": assessment,
                    "context": ctx0,
                    "case_text": build_case_text(age, sex, history, symptoms, medications, allergies, ""),
                }
                st.session_state["followup_chat"] = []
                try:
                    nq = api_next_question(api_url, encounter_id)
                    first_q = (nq.get("question") or "").strip()
                    st.session_state["latest_question"] = first_q or None
                    st.session_state["latest_turn_no"] = nq.get("turn_no")
                    st.session_state["latest_policy_reason"] = nq.get("policy_reason")
                    if first_q:
                        st.session_state["followup_chat"].append({"role": "assistant", "content": first_q})
                except Exception:
                    st.session_state["latest_question"] = None
                    st.session_state["latest_turn_no"] = None
                    st.session_state["latest_policy_reason"] = None
                st.success("Differential generated.")
            except Exception as exc:
                st.error(f"Encounter workflow API failed: {exc}")

        case = st.session_state.get("case")
        if not case:
            st.info("Fill intake and click **Generate Differential**.")
        else:
            assessment = case.get("assessment") or {}
            encounter_id = case.get("encounter_id")
            st.markdown(f"**Encounter ID:** `{encounter_id}`")
            st.markdown(f"**Provider:** `{assessment.get('provider_used') or assessment.get('provider', 'unknown')}`")
            if assessment.get("degraded_mode"):
                st.caption(f"Degraded mode: `{assessment.get('degraded_mode')}`")
            if assessment.get("fallback_mode"):
                st.caption(f"Evidence mode: `{assessment.get('fallback_mode')}`")

            top_candidates = assessment.get("top_candidates") or []
            if top_candidates:
                st.markdown("### Prioritized Differential")
                render_ranked_candidates_md(top_candidates)
                source_counts: dict[str, int] = {}
                for row in top_candidates:
                    src = str(row.get("source") or "unknown")
                    source_counts[src] = source_counts.get(src, 0) + 1
                if source_counts:
                    src_txt = ", ".join(f"{k}:{v}" for k, v in sorted(source_counts.items()))
                    st.caption(f"Differential sources -> {src_txt}")
            else:
                st.warning("No ranked candidates returned.")

            st.markdown("### Clinical Assessment")
            render_compact_assessment(assessment)
            with st.expander("Full assessment details", expanded=False):
                st.markdown(assessment.get("assessment") or "_No assessment text returned._")
            render_evidence_block(assessment.get("evidence_summary"), assessment.get("evidence_sources"))

            st.markdown("### Follow-up Chat")
            render_followup_chat(st.session_state.get("followup_chat") or [])
            if st.session_state.get("latest_policy_reason"):
                st.caption(f"Current question policy: `{st.session_state.get('latest_policy_reason')}`")

            doctor_reply = st.chat_input("Type your follow-up answer and press Enter", key=f"chat_input_{encounter_id}")
            if doctor_reply:
                latest_turn = st.session_state.get("latest_turn_no")
                if latest_turn is None:
                    st.warning("No active follow-up question. Generate differential again to restart chat.")
                else:
                    st.session_state["followup_chat"].append({"role": "user", "content": doctor_reply.strip()})
                    try:
                        upd = api_answer_question(api_url, encounter_id, int(latest_turn), doctor_reply.strip())
                        st.session_state["case"]["assessment"] = upd
                        st.session_state["case"]["context"] = api_get_context(api_url, encounter_id)
                        nq = api_next_question(api_url, encounter_id)
                        next_q = (nq.get("question") or "").strip()
                        st.session_state["latest_question"] = next_q or None
                        st.session_state["latest_turn_no"] = nq.get("turn_no")
                        st.session_state["latest_policy_reason"] = nq.get("policy_reason")
                        if next_q:
                            st.session_state["followup_chat"].append({"role": "assistant", "content": next_q})
                        st.success("Differential updated from your reply.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to process follow-up chat: {exc}")

            if st.button("Refresh encounter context", key="btn_refresh_ctx"):
                try:
                    st.session_state["case"]["context"] = api_get_context(api_url, encounter_id)
                    st.success("Context refreshed.")
                except Exception as exc:
                    st.error(f"Failed to fetch context: {exc}")

    with tab_qa:
        st.subheader("Doctor Q&A")
        st.caption("RAG-first answers with strict journal-first evidence and web fallback when needed.")
        api_url = st.text_input("API URL", value=st.session_state["api_url"], key="qa_api_url")
        st.session_state["api_url"] = api_url
        ai_mode = st.radio(
            "Provider",
            ["Auto (Cortex → Gemini fallback)", "Cortex Only (MedRAG)", "Gemini Only", "Compare Both"],
            horizontal=True,
            key="qa_mode",
        )
        question = st.text_area("Ask a clinical question", height=110, key="doctor_rag_question")
        include_case = st.checkbox("Include current encounter context", value=True, key="qa_include_case")
        if st.button("Ask", key="doctor_rag_submit"):
            q = (question or "").strip()
            if not q:
                st.warning("Enter a question.")
            else:
                case = st.session_state.get("case")
                case_text = (case or {}).get("case_text") or ""
                final_question = f"Patient case summary:\n{case_text}\n\nClinician question: {q}" if include_case and case_text else q
                try:
                    if ai_mode == "Compare Both":
                        results = ask_both(api_url=api_url, question=final_question, brief=False)
                        col_g, col_c = st.columns(2)
                        with col_g:
                            st.markdown("### Gemini")
                            st.markdown(results.get("gemini") or "_Empty response_")
                        with col_c:
                            st.markdown("### Cortex")
                            st.markdown(results.get("cortex") or "_Empty response_")
                    elif ai_mode == "Cortex Only (MedRAG)":
                        st.markdown(ask_cortex(api_url=api_url, question=final_question) or "_Empty response_")
                    else:
                        result = ask_llm(api_url=api_url, question=final_question, brief=False)
                        st.markdown(f"**Provider:** `{result.get('provider', '?')}`")
                        if result.get("fallback_mode"):
                            st.caption(f"Evidence mode: `{result.get('fallback_mode')}`")
                        eq = result.get("evidence_quality") or {}
                        if eq:
                            st.caption(
                                f"Evidence quality: score={eq.get('score')} "
                                f"(trusted={eq.get('trusted_count')}, total={eq.get('count')})"
                            )
                        if result.get("insufficient_evidence"):
                            st.warning("Evidence is insufficient for a confident clinical answer.")
                        st.markdown(result.get("answer") or "_Empty response_")
                        fu = result.get("follow_up_questions") or []
                        if fu:
                            st.markdown("### What to ask next")
                            for qx in fu[:4]:
                                st.markdown(f"- {qx}")
                        render_evidence_block(result.get("evidence_summary"), result.get("evidence_sources"))
                except Exception as exc:
                    st.error(f"Q&A failed: {exc}")

    st.caption(
        "For clinical support only. This tool does not replace physician judgment, protocol-based care, or emergency escalation."
    )


if __name__ == "__main__":
    main()
