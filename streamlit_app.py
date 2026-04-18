#!/usr/bin/env python3
"""Doctor-side Streamlit app for MedAssist.AI."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st

from src.indexing import SymptomIndex
from src.clinical_workflow import normalize_encounter_dict
from src.followup_policy import MAX_TURNS_DEFAULT

DEFAULT_API_URL = os.environ.get("MEDASSIST_API_BASE", "http://127.0.0.1:8000")
# Doctor Q&A uses /ask-doctor: Snowflake Cortex Agent first (if PAT+agent env set), else Cortex → Gemini.
_DOCTOR_QA_AI_MODE = "Doctor default (Agent → Cortex → Gemini)"
_log = logging.getLogger("medassist.streamlit")

# region agent log
_QA_PERF_LOG = Path(__file__).resolve().parent / ".cursor" / "debug-064a4f.log"
_QA_PERF_INGEST = "http://127.0.0.1:7299/ingest/6c1651b6-79fe-48a7-a0a7-0d0f9a35fdde"


def _qa_perf064(hypothesis_id: str, message: str, data: dict[str, Any]) -> None:
    """NDJSON timing for Doctor Q&A tab (session 064a4f; no PII — lengths and modes only)."""
    payload = {
        "sessionId": "064a4f",
        "timestamp": int(time.time() * 1000),
        "hypothesisId": hypothesis_id,
        "location": "streamlit_app.py:tab_qa",
        "message": message,
        "data": data,
    }
    line = json.dumps(payload, default=str) + "\n"
    try:
        _QA_PERF_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_QA_PERF_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    try:
        requests.post(
            _QA_PERF_INGEST,
            json=payload,
            headers={"Content-Type": "application/json", "X-Debug-Session-Id": "064a4f"},
            timeout=2,
        )
    except Exception:
        pass


# endregion


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    if os.environ.get("DEBUG", "").strip().lower() not in ("1", "true", "yes"):
        return
    _log.debug("%s %s %s", hypothesis_id, location, json.dumps(data, default=str))


def _req_headers() -> dict[str, str]:
    """Headers for API calls. With AUTH_DISABLED=1 on the API, Bearer is not required."""
    return {}

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
    # Light: mostly light surfaces + dark text. Dark: mostly dark surfaces + light text.
    st.markdown(
        """
        <style>
            [data-testid="stToolbar"] { display: none !important; }
            .block-container { padding-top: 1.2rem; max-width: 1400px; }

            /* ----- Light theme ----- */
            [data-testid="stAppViewContainer"][data-theme="light"] {
                --med-deep: #1a5fb4;
                --med-cerulean: #2bb0ed;
                --med-navy: #0c1d2e;
                --med-sky: #d4ebf7;
                --med-offwhite: #f4f8fc;
                --med-text-muted: #3d5a73;
            }
            [data-testid="stAppViewContainer"][data-theme="light"] .stApp {
                background: linear-gradient(
                    180deg,
                    var(--med-offwhite) 0%,
                    var(--med-sky) 52%,
                    var(--med-offwhite) 100%
                ) !important;
            }
            [data-testid="stAppViewContainer"][data-theme="light"] .block-container {
                border-top: 4px solid var(--med-deep);
                box-shadow:
                    0 1px 0 rgba(43, 176, 237, 0.28),
                    0 0 0 1px rgba(26, 95, 180, 0.1);
            }
            [data-testid="stAppViewContainer"][data-theme="light"] h2,
            [data-testid="stAppViewContainer"][data-theme="light"] h3 {
                color: var(--med-navy) !important;
            }
            [data-testid="stAppViewContainer"][data-theme="light"] .stMarkdown a {
                color: var(--med-cerulean) !important;
                font-weight: 600;
            }
            [data-testid="stAppViewContainer"][data-theme="light"] .stMarkdown a:hover {
                color: var(--med-deep) !important;
            }

            /* ----- Dark theme ----- */
            [data-testid="stAppViewContainer"][data-theme="dark"] {
                --med-bg-deep: #0a1628;
                --med-bg-mid: #121f33;
                --med-bg-elevated: #1a2d45;
                --med-text: #e8eef4;
                --med-text-muted: #9db4c4;
                --med-accent: #4cc9f0;
                --med-accent-dim: #2bb0ed;
            }
            [data-testid="stAppViewContainer"][data-theme="dark"] .stApp {
                background: linear-gradient(
                    180deg,
                    var(--med-bg-deep) 0%,
                    var(--med-bg-mid) 45%,
                    var(--med-bg-deep) 100%
                ) !important;
            }
            [data-testid="stAppViewContainer"][data-theme="dark"] .block-container {
                border-top: 4px solid var(--med-accent-dim);
                box-shadow:
                    0 0 0 1px rgba(148, 184, 210, 0.12),
                    0 4px 24px rgba(0, 0, 0, 0.35);
            }
            [data-testid="stAppViewContainer"][data-theme="dark"] h2,
            [data-testid="stAppViewContainer"][data-theme="dark"] h3 {
                color: var(--med-text) !important;
            }
            [data-testid="stAppViewContainer"][data-theme="dark"] .stMarkdown a {
                color: var(--med-accent) !important;
                font-weight: 600;
            }
            [data-testid="stAppViewContainer"][data-theme="dark"] .stMarkdown a:hover {
                color: #7dd8f5 !important;
            }

            /* ----- Centered app header + 3D title (both themes) ----- */
            .med-app-header {
                text-align: center;
                margin: 0.25rem 0 1.35rem 0;
                padding: 0.5rem 0.5rem 0.85rem 0.5rem;
            }
            .med-app-header .med-title-3d {
                margin: 0;
                font-weight: 800;
                letter-spacing: 0.04em;
                line-height: 1.15;
            }
            .med-app-header .med-header-sub {
                margin: 0.65rem 0 0 0;
                font-size: 0.95rem;
                line-height: 1.4;
            }
            [data-testid="stAppViewContainer"][data-theme="light"] .med-app-header .med-title-3d {
                color: #0c1d2e;
                text-shadow:
                    -1px -1px 0 rgba(255, 255, 255, 0.95),
                    1px 1px 0 rgba(26, 95, 180, 0.35),
                    2px 3px 0 rgba(12, 29, 46, 0.12),
                    0 6px 14px rgba(12, 29, 46, 0.18);
            }
            [data-testid="stAppViewContainer"][data-theme="light"] .med-app-header .med-header-sub {
                color: var(--med-text-muted, #3d5a73);
            }
            [data-testid="stAppViewContainer"][data-theme="dark"] .med-app-header .med-title-3d {
                color: #e8eef4;
                text-shadow:
                    -1px -1px 0 rgba(255, 255, 255, 0.22),
                    1px 1px 0 rgba(0, 0, 0, 0.85),
                    2px 3px 0 rgba(0, 0, 0, 0.55),
                    0 5px 18px rgba(0, 0, 0, 0.6);
            }
            [data-testid="stAppViewContainer"][data-theme="dark"] .med-app-header .med-header-sub {
                color: var(--med-text-muted, #9db4c4);
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
    url = api_url.rstrip("/") + "/ask-doctor"
    payload = {"question": question, "brief": brief}
    r = requests.post(url, json=payload, headers=_req_headers(), timeout=120)
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
    r = requests.post(url, json=payload, headers=_req_headers(), timeout=180)
    r.raise_for_status()
    return (r.json().get("answer") or "").strip()


def ask_both(api_url: str, question: str, brief: bool) -> dict:
    url = api_url.rstrip("/") + "/ask-both"
    payload = {"question": question, "brief": brief}
    r = requests.post(url, json=payload, headers=_req_headers(), timeout=180)
    r.raise_for_status()
    data = r.json()
    return {
        "gemini": (data.get("answer_gemini") or "").strip(),
        "cortex": (data.get("answer_cortex") or "").strip(),
    }


def api_start_encounter(api_url: str, payload: dict) -> str:
    url = api_url.rstrip("/") + "/encounters/start"
    r = requests.post(url, json=payload, headers=_req_headers(), timeout=120)
    r.raise_for_status()
    return r.json()["encounter_id"]


def api_initial_assessment(api_url: str, encounter_id: str, mode: str = "fast") -> dict:
    endpoint = "assess-deep" if mode == "deep" else "assess-fast"
    url = api_url.rstrip("/") + f"/encounters/{encounter_id}/{endpoint}"
    r = requests.post(url, headers=_req_headers(), timeout=180)
    r.raise_for_status()
    return r.json()


def api_next_question(api_url: str, encounter_id: str) -> dict:
    url = api_url.rstrip("/") + f"/encounters/{encounter_id}/next-question"
    r = requests.post(url, headers=_req_headers(), timeout=120)
    r.raise_for_status()
    return r.json()


def api_answer_question(api_url: str, encounter_id: str, turn_no: int, answer: str) -> dict:
    url = api_url.rstrip("/") + f"/encounters/{encounter_id}/answer"
    r = requests.post(
        url,
        json={"turn_no": turn_no, "answer": answer},
        headers=_req_headers(),
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def api_get_context(api_url: str, encounter_id: str) -> dict:
    url = api_url.rstrip("/") + f"/encounters/{encounter_id}/context"
    r = requests.get(url, headers=_req_headers(), timeout=120)
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
        headers=_req_headers(),
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
    rows_df: list[dict[str, Any]] = []
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
        cf = c.get("confidence_score")
        rows_df.append(
            {
                "Rank": i,
                "Disease": name,
                "Source": kind,
                "Score": float(score) if isinstance(score, (int, float)) else float("nan"),
                "Conf": float(cf) if isinstance(cf, (int, float)) else float("nan"),
                "Reference": url,
            }
        )
    # region agent log
    _agent_debug_log(
        "H5",
        "streamlit_app:render_ranked_candidates_md",
        "link_kind_counts",
        {"n_candidates": len(candidates), "n_orphanet": n_orpha, "n_pubmed_fallback": n_pubmed},
    )
    # endregion
    df = pd.DataFrame(rows_df)
    st.dataframe(
        df,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "Disease": st.column_config.TextColumn("Disease", width="large"),
            "Source": st.column_config.TextColumn("Source", width="small"),
            "Score": st.column_config.NumberColumn("Score", format="%.2f"),
            "Conf": st.column_config.NumberColumn("Conf", format="%.2f", help="Model-estimated confidence"),
            "Reference": st.column_config.LinkColumn("Reference", display_text="Open"),
        },
        hide_index=True,
        use_container_width=True,
    )


def render_evidence_block(summary: str | None, sources: list[dict[str, Any]] | None) -> None:
    st.markdown("### Evidence (brief)")
    st.caption(summary or "No explicit evidence summary available.")
    rows = sources or []
    if not rows:
        st.info("No evidence links returned.")
        return
    slice_rows = rows[:5]
    if not all((row.get("url") or "").strip() for row in slice_rows):
        lines: list[str] = []
        for i, row in enumerate(slice_rows, start=1):
            title = (row.get("title") or row.get("source") or "evidence").strip()
            url = (row.get("url") or "").strip()
            source = (row.get("source") or "unknown").strip()
            if url:
                lines.append(f"{i}. [{title}]({url}) — _{source}_")
            else:
                lines.append(f"{i}. {title} — _{source}_")
        st.markdown("\n".join(lines))
        return
    table: list[dict[str, Any]] = []
    for row in slice_rows:
        title = (row.get("title") or row.get("source") or "evidence").strip()
        url = (row.get("url") or "").strip()
        source = (row.get("source") or "unknown").strip()
        table.append({"Title": title, "Source": source, "Reference": url})
    df = pd.DataFrame(table)
    st.dataframe(
        df,
        column_config={
            "Title": st.column_config.TextColumn("Title", width="large", max_chars=120),
            "Source": st.column_config.TextColumn("Source", width="small"),
            "Reference": st.column_config.LinkColumn("Reference", display_text="Open"),
        },
        hide_index=True,
        use_container_width=True,
    )


def render_followup_chat(messages: list[dict[str, str]]) -> None:
    if not messages:
        st.info("No follow-up chat yet.")
        return
    for msg in messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        with st.chat_message("assistant" if role == "assistant" else "user"):
            st.markdown(content or "")


def _sync_followup_complete_from_case(case: dict[str, Any]) -> None:
    """Align UI with persisted QA so follow-up entry stays off after the turn limit."""
    ctx = (case or {}).get("context") or {}
    structured = normalize_encounter_dict(ctx.get("structured_context") or {})
    qa = structured.get("qa_history") or []
    answered = sum(1 for r in qa if (r.get("answer") or "").strip())
    if len(qa) >= MAX_TURNS_DEFAULT and answered >= MAX_TURNS_DEFAULT:
        st.session_state["followup_complete"] = True
        st.session_state["latest_turn_no"] = None
        st.session_state.pop("followup_answer_choices", None)


def render_compact_assessment(assessment: dict[str, Any], *, show_differential_shortlist: bool = True) -> None:
    contract = (assessment or {}).get("contract") or {}
    if not contract:
        # Fallback for paths that do not return structured contract.
        st.markdown((assessment or {}).get("assessment") or "_No assessment text returned._")
        return
    st.markdown("#### Summary")
    summary = (contract.get("summary") or "").strip()
    if summary:
        st.markdown(f"- {summary[:320]}{'...' if len(summary) > 320 else ''}")
    diff = contract.get("differential") or []
    if show_differential_shortlist and diff:
        items = [str(x.get("disease_name") or "").strip() for x in diff[:5] if x.get("disease_name")]
        if items:
            st.markdown("#### Top conditions (from contract)")
            for i, name in enumerate(items, start=1):
                st.markdown(f"{i}. {name}")
    red_flags = contract.get("red_flags") or []
    st.markdown("#### Red flags")
    if red_flags:
        st.warning("\n".join(f"- {rf}" for rf in red_flags[:4]))
    else:
        st.markdown("- No immediate rule-based red flags.")
    next_steps = contract.get("next_steps") or []
    if next_steps:
        st.markdown("#### Next steps")
        st.success("\n".join(f"- {step}" for step in next_steps[:3]))


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
        st.caption("Lower cap speeds loads; raise it if the table looks truncated.")
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
        df_kg = pd.DataFrame(edges)
        kg_cfg = {
            c: st.column_config.TextColumn(str(c), max_chars=64, width="medium")
            for c in df_kg.columns
        }
        st.dataframe(
            df_kg,
            column_config=kg_cfg,
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
        initial_sidebar_state="expanded",
    )
    inject_theme()

    st.session_state.setdefault("api_url", DEFAULT_API_URL)
    st.session_state.setdefault("case", None)
    st.session_state.setdefault("latest_question", None)
    st.session_state.setdefault("latest_turn_no", None)
    st.session_state.setdefault("latest_policy_reason", None)
    st.session_state.setdefault("followup_complete", False)
    st.session_state.setdefault("followup_chat", [])
    st.session_state.setdefault("doctor_qa_history", [])
    # UI role for optional Admin tab (API auth is controlled by AUTH_DISABLED / JWT on the server).
    st.session_state.setdefault(
        "user_role",
        os.environ.get("MEDASSIST_STREAMLIT_ROLE", "doctor").strip().lower(),
    )

    with st.sidebar:
        st.markdown("### Backend")
        st.session_state["api_url"] = st.text_input(
            "API base URL",
            value=st.session_state["api_url"],
            help="FastAPI base URL (no login here). Default: MEDASSIST_API_BASE env or http://127.0.0.1:8000. Change if your API uses another host/port.",
        )

    st.markdown(
        """
        <div class="med-app-header">
          <h1 class="med-title-3d">MedAssist.AI</h1>
          <p class="med-header-sub">Clinical workspace · decision support only · not a medical device</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = ["Differential + Evidence", "Doctor Q&A"]
    if st.session_state.get("user_role") == "admin":
        tabs.append("Admin · Quality")
    tab_objs = st.tabs(tabs)
    tab_diff = tab_objs[0]
    tab_qa = tab_objs[1]
    tab_admin = tab_objs[2] if len(tab_objs) > 2 else None

    with tab_diff:
        st.subheader("Patient Intake and Differential")
        with st.form("intake_form"):
            with st.container(border=True):
                st.caption("Demographics")
                c1, c2 = st.columns(2)
                with c1:
                    age = st.number_input("Age", min_value=0, max_value=120, value=35)
                with c2:
                    sex = st.selectbox("Sex", ["Female", "Male", "Other / Not specified"])
            with st.container(border=True):
                st.caption("Symptoms")
                picked = st.pills(
                    "Common symptoms (click to toggle)",
                    COMMON_SYMPTOMS,
                    selection_mode="multi",
                    help="Faster than typing; same list as before. Does not speed up Snowflake assessment after submit.",
                )
                symptoms_pick = list(picked) if picked else []
                symptoms_custom = st.text_input("Additional symptoms (comma separated)")
            with st.container(border=True):
                st.caption("Clinical context")
                pre_existing_conditions = st.text_input("Pre-existing conditions (comma separated)")
                history = st.text_area("History and key exam findings", height=100)
                medications = st.text_input("Current medications (comma separated)")
                allergies = st.text_input("Allergies (comma separated)")
            submitted = st.form_submit_button(
                "Generate Differential",
                help="Creates a new encounter in the API, runs KG-led assess-fast, loads context, and seeds the first follow-up question. Use after changing intake fields.",
            )

        api_url = st.session_state["api_url"]

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
            with st.status("Generating differential…", expanded=True) as status:
                try:
                    status.update(label="Creating encounter…", state="running", expanded=True)
                    encounter_id = api_start_encounter(api_url, encounter_payload)
                    # Use a single deterministic clinical path (KG-led) for consistency.
                    status.update(label="Running clinical assessment…", state="running", expanded=True)
                    assessment = api_initial_assessment(api_url, encounter_id, mode="fast")
                    status.update(label="Loading encounter context…", state="running", expanded=True)
                    ctx0 = api_get_context(api_url, encounter_id)
                    st.session_state["case"] = {
                        "encounter_id": encounter_id,
                        "assessment": assessment,
                        "context": ctx0,
                        "case_text": build_case_text(age, sex, history, symptoms, medications, allergies, ""),
                    }
                    st.session_state["followup_chat"] = []
                    st.session_state["followup_complete"] = False
                    st.session_state.pop("followup_pending_followup", None)
                    st.session_state.pop("_followup_chat_input_echo", None)
                    st.session_state.pop("followup_answer_choices", None)
                    status.update(label="Preparing follow-up question…", state="running", expanded=True)
                    try:
                        nq = api_next_question(api_url, encounter_id)
                        if nq.get("policy_reason") == "max_turns" or nq.get("turn_no") is None:
                            st.session_state["latest_question"] = None
                            st.session_state["latest_turn_no"] = None
                            st.session_state["latest_policy_reason"] = nq.get("policy_reason")
                            st.session_state.pop("followup_answer_choices", None)
                            st.session_state["followup_complete"] = True
                        else:
                            first_q = (nq.get("question") or "").strip()
                            st.session_state["latest_question"] = first_q or None
                            st.session_state["latest_turn_no"] = nq.get("turn_no")
                            st.session_state["latest_policy_reason"] = nq.get("policy_reason")
                            ac0 = nq.get("answer_choices")
                            st.session_state["followup_answer_choices"] = (
                                list(ac0) if isinstance(ac0, list) and len(ac0) > 0 else None
                            )
                            st.session_state["followup_complete"] = False
                            if first_q:
                                st.session_state["followup_chat"].append(
                                    {"role": "assistant", "content": first_q}
                                )
                    except Exception:
                        st.session_state["latest_question"] = None
                        st.session_state["latest_turn_no"] = None
                        st.session_state["latest_policy_reason"] = None
                        st.session_state.pop("followup_answer_choices", None)
                        st.session_state["followup_complete"] = False
                    status.update(label="Differential ready.", state="complete", expanded=False)
                    st.success("Differential generated.")
                except Exception as exc:
                    status.update(label="Encounter workflow failed", state="error", expanded=True)
                    st.error(f"Encounter workflow API failed: {exc}")

        case = st.session_state.get("case")
        if not case:
            st.info("Fill intake and click **Generate Differential**.")
        else:
            assessment = case.get("assessment") or {}
            encounter_id = case.get("encounter_id")

            top_candidates = assessment.get("top_candidates") or []
            if top_candidates:
                st.divider()
                st.markdown("### Prioritized Differential")
                render_ranked_candidates_md(top_candidates)
                source_counts: dict[str, int] = {}
                for row in top_candidates:
                    src = str(row.get("source") or "unknown")
                    source_counts[src] = source_counts.get(src, 0) + 1
                if len(source_counts) > 1:
                    src_cols = st.columns(len(source_counts))
                    for i, (k, v) in enumerate(sorted(source_counts.items())):
                        with src_cols[i]:
                            st.metric(label=k, value=v)
            else:
                st.divider()
                st.warning("No ranked candidates returned.")

            st.divider()
            st.markdown("### Clinical guidance")
            render_compact_assessment(assessment, show_differential_shortlist=not bool(top_candidates))
            with st.expander("Full assessment details", expanded=False):
                st.markdown(assessment.get("assessment") or "_No assessment text returned._")
            render_evidence_block(assessment.get("evidence_summary"), assessment.get("evidence_sources"))

            st.divider()
            st.markdown("### Follow-up Chat")
            render_followup_chat(st.session_state.get("followup_chat") or [])
            if st.session_state.get("followup_complete"):
                st.info(
                    f"Follow-up is complete (maximum {MAX_TURNS_DEFAULT} questions for this encounter). "
                    "Summarize and consider disposition or further workup."
                )

            pend_key = "followup_pending_followup"
            stale = st.session_state.get(pend_key)
            if stale and stale.get("encounter_id") != encounter_id:
                st.session_state.pop(pend_key, None)

            _sync_followup_complete_from_case(case)

            ac_raw = st.session_state.get("followup_answer_choices")
            answer_choices = ac_raw if isinstance(ac_raw, list) and len(ac_raw) > 0 else None
            choices_mode = bool(answer_choices)
            have_pending = bool(st.session_state.get(pend_key))
            fu_done = bool(st.session_state.get("followup_complete"))
            if fu_done:
                doctor_reply = None
            elif choices_mode or have_pending:
                # Avoid a disabled chat bar beside checkbox answers; free-text uses chat_input only when needed.
                doctor_reply = None
            else:
                doctor_reply = st.chat_input(
                    "Reply…",
                    key=f"chat_input_{encounter_id}",
                )
            payload = st.session_state.pop(pend_key, None)

            def _append_user_followup_line(t: str) -> None:
                fc = st.session_state["followup_chat"]
                t = t.strip()
                if fc and fc[-1].get("role") == "user" and (fc[-1].get("content") or "").strip() == t:
                    _qa_perf064("H_dup", "followup_user_append_skipped", {"text_len": len(t), "chat_len": len(fc)})
                    return
                fc.append({"role": "user", "content": t})
                _qa_perf064("H_dup", "followup_user_appended", {"text_len": len(t), "chat_len": len(fc)})

            if payload and isinstance(payload, dict) and payload.get("encounter_id") == encounter_id:
                if st.session_state.get("followup_complete"):
                    st.warning("Follow-up is already complete; that reply was not sent.")
                else:
                    text = (payload.get("text") or "").strip()
                    turn_no = int(payload["turn_no"])
                    latest_turn = st.session_state.get("latest_turn_no")
                    if latest_turn is None or int(latest_turn) != turn_no:
                        st.warning("That follow-up is no longer active; your draft was not sent.")
                        _qa_perf064(
                            "H_dup",
                            "followup_pending_stale_turn",
                            {"turn_no": turn_no, "latest": latest_turn},
                        )
                    elif not text:
                        pass
                    else:
                        st.caption("Sending your reply…")
                        _append_user_followup_line(text)
                        try:
                            prev_names = candidate_name_list(
                                (st.session_state["case"].get("assessment") or {}).get("top_candidates")
                            )
                            upd = api_answer_question(api_url, encounter_id, int(latest_turn), text)
                            new_names = candidate_name_list(upd.get("top_candidates"))
                            st.session_state["case"]["assessment"] = upd
                            st.session_state["case"]["context"] = api_get_context(api_url, encounter_id)
                            with st.expander("Differential changes after your reply", expanded=True):
                                st.markdown(format_candidate_diff_markdown(prev_names, new_names))
                            nq = api_next_question(api_url, encounter_id)
                            if nq.get("policy_reason") == "max_turns" or nq.get("turn_no") is None:
                                st.session_state["latest_question"] = None
                                st.session_state["latest_turn_no"] = None
                                st.session_state["latest_policy_reason"] = nq.get("policy_reason")
                                st.session_state.pop("followup_answer_choices", None)
                                st.session_state["followup_complete"] = True
                            else:
                                next_q = (nq.get("question") or "").strip()
                                st.session_state["latest_question"] = next_q or None
                                st.session_state["latest_turn_no"] = nq.get("turn_no")
                                st.session_state["latest_policy_reason"] = nq.get("policy_reason")
                                acn = nq.get("answer_choices")
                                st.session_state["followup_answer_choices"] = (
                                    list(acn) if isinstance(acn, list) and len(acn) > 0 else None
                                )
                                st.session_state["followup_complete"] = False
                                if next_q:
                                    st.session_state["followup_chat"].append(
                                        {"role": "assistant", "content": next_q}
                                    )
                            st.session_state["_followup_chat_input_echo"] = text
                            st.success("Differential updated from your reply.")
                            _qa_perf064("H_dup", "followup_commit_ok", {"text_len": len(text)})
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Failed to process follow-up chat: {exc}")
                            _qa_perf064("H_dup", "followup_commit_err", {"err": type(exc).__name__})
            elif choices_mode and not have_pending and not st.session_state.get("followup_complete"):
                st.caption(
                    "Select one or more options (checkboxes), then **Send reply**. Options speed data entry only; Snowflake assessment time is unchanged."
                )
                none_l = "None of the above (specify below)"
                turn_key = int(st.session_state.get("latest_turn_no") or 0)
                st.markdown("**Your answer**")
                picked_regular: list[str] = []
                for i, opt in enumerate(list(answer_choices or [])):
                    if st.checkbox(
                        opt,
                        key=f"followup_cb_{encounter_id}_{turn_key}_{i}",
                    ):
                        picked_regular.append(opt)
                none_checked = st.checkbox(
                    none_l,
                    key=f"followup_cb_none_{encounter_id}_{turn_key}",
                )
                other_txt = ""
                if none_checked:
                    other_txt = st.text_area(
                        "Describe your answer",
                        height=72,
                        key=f"followup_other_{encounter_id}_{turn_key}",
                        placeholder="Required when using “None of the above”.",
                    )
                if st.button("Send reply", key=f"followup_send_{encounter_id}_{turn_key}"):
                    latest_turn = st.session_state.get("latest_turn_no")
                    if latest_turn is None:
                        st.warning("No active follow-up question. Generate differential again to restart chat.")
                    elif none_checked and picked_regular:
                        st.warning(
                            "Use either the listed options or “None of the above”, not both."
                        )
                    elif none_checked and not (other_txt or "").strip():
                        st.warning('Enter details for “None of the above”, or uncheck it and pick listed options.')
                    elif not none_checked and not picked_regular:
                        st.warning("Select at least one checkbox above.")
                    else:
                        if none_checked:
                            composed = f"Other: {(other_txt or '').strip()}"
                        else:
                            composed = "Selected: " + "; ".join(picked_regular)
                        st.session_state[pend_key] = {
                            "text": composed,
                            "turn_no": int(latest_turn),
                            "encounter_id": encounter_id,
                        }
                        _qa_perf064("H_dup", "followup_pending_set", {"text_len": len(composed), "turn": int(latest_turn)})
                        st.rerun()
            elif (
                not choices_mode
                and doctor_reply
                and not st.session_state.get("followup_complete")
            ):
                reply_txt = doctor_reply.strip()
                if reply_txt == (st.session_state.get("_followup_chat_input_echo") or "").strip():
                    _qa_perf064("H_dup", "followup_ghost_submit_ignored", {"text_len": len(reply_txt)})
                else:
                    latest_turn = st.session_state.get("latest_turn_no")
                    if latest_turn is None:
                        st.warning("No active follow-up question. Generate differential again to restart chat.")
                    else:
                        st.session_state[pend_key] = {
                            "text": reply_txt,
                            "turn_no": int(latest_turn),
                            "encounter_id": encounter_id,
                        }
                        _qa_perf064("H_dup", "followup_pending_set", {"text_len": len(reply_txt), "turn": int(latest_turn)})
                        st.rerun()

    with tab_qa:
        st.subheader("Doctor Q&A")
        st.caption("RAG-first answers with strict journal-first evidence and web fallback when needed.")

        def _prior_session_block(hist: list[dict[str, Any]], max_chars: int = 1800) -> str:
            parts: list[str] = []
            for h in hist[-5:]:
                u = (h.get("user_q") or "").strip()
                pv = (h.get("answer_preview") or "").strip()
                if u:
                    parts.append(f"Earlier Q: {u}\nEarlier A (excerpt): {pv}\n")
            return "\n".join(parts).strip()[:max_chars]

        qa_hist: list[dict[str, Any]] = st.session_state["doctor_qa_history"]
        if qa_hist:
            st.markdown("### Recent answers (this session)")
            for idx, h in enumerate(reversed(qa_hist[-6:])):
                t = (h.get("user_q") or "Question")[:88]
                if len(h.get("user_q") or "") > 88:
                    t += "…"
                with st.expander(t, expanded=(idx == 0)):
                    st.markdown(h.get("body_md") or "_Empty._")
            if st.button("Clear Q&A history", key="doctor_qa_clear_hist"):
                st.session_state["doctor_qa_history"] = []
                st.rerun()

        with st.container(border=True):
            api_url = st.session_state["api_url"]
            ai_mode = _DOCTOR_QA_AI_MODE
            question = st.text_area("Ask a clinical question", height=110, key="doctor_rag_question")
            include_case = st.checkbox(
                "Include current encounter context",
                value=True,
                key="qa_include_case",
                help="Prepends the intake-derived case summary from the Differential tab. Prior Q&A replies in this session are added automatically when you have history.",
            )
            ask_clicked = st.button("Ask", key="doctor_rag_submit")
        if ask_clicked:
            q = (question or "").strip()
            if not q:
                st.warning("Enter a question.")
            else:
                case = st.session_state.get("case")
                case_text = (case or {}).get("case_text") or ""
                prior_blk = _prior_session_block(st.session_state["doctor_qa_history"])
                blocks: list[str] = []
                if include_case and case_text:
                    blocks.append(f"Patient case summary:\n{case_text}")
                if prior_blk:
                    blocks.append(f"Prior dialogue in this browser session:\n{prior_blk}")
                blocks.append(f"Clinician question: {q}")
                final_question = "\n\n".join(blocks)
                # region agent log
                _qa_perf064(
                    "H2",
                    "final_question_built",
                    {
                        "user_q_chars": len(q),
                        "final_q_chars": len(final_question),
                        "has_prior_blk": bool(prior_blk),
                        "include_case": bool(include_case and case_text),
                        "brief": False,
                    },
                )
                # endregion
                try:
                    with st.spinner("Retrieving evidence and generating answer…"):
                        st.divider()
                        t_api = time.perf_counter()
                        if ai_mode == "Compare Both":
                            _qa_perf064("H1", "ask_route_start", {"mode": "compare_both"})
                            results = ask_both(api_url=api_url, question=final_question, brief=False)
                            _qa_perf064(
                                "H1",
                                "ask_route_done",
                                {"elapsed_ms": round((time.perf_counter() - t_api) * 1000, 2), "mode": "compare_both"},
                            )
                            col_g, col_c = st.columns(2)
                            with col_g:
                                st.markdown("### Gemini")
                                st.markdown(results.get("gemini") or "_Empty response_")
                            with col_c:
                                st.markdown("### Cortex")
                                st.markdown(results.get("cortex") or "_Empty response_")
                            g = results.get("gemini") or ""
                            c = results.get("cortex") or ""
                            body = f"### Gemini\n{g}\n\n### Cortex\n{c}"
                            st.session_state["doctor_qa_history"].append(
                                {
                                    "user_q": q,
                                    "answer_preview": (g + "\n" + c)[:500],
                                    "body_md": body,
                                }
                            )
                            _qa_perf064("H3", "qa_history_appended", {"history_len": len(st.session_state["doctor_qa_history"])})
                            st.rerun()
                        elif ai_mode == "Cortex Only (MedRAG)":
                            _qa_perf064("H1", "ask_route_start", {"mode": "cortex_only"})
                            cx = ask_cortex(api_url=api_url, question=final_question) or "_Empty response_"
                            _qa_perf064(
                                "H1",
                                "ask_route_done",
                                {"elapsed_ms": round((time.perf_counter() - t_api) * 1000, 2), "mode": "cortex_only"},
                            )
                            st.markdown(cx)
                            st.session_state["doctor_qa_history"].append(
                                {"user_q": q, "answer_preview": str(cx)[:500], "body_md": str(cx)}
                            )
                            _qa_perf064("H3", "qa_history_appended", {"history_len": len(st.session_state["doctor_qa_history"])})
                            st.rerun()
                        else:
                            _qa_perf064("H1", "ask_route_start", {"mode": ai_mode})
                            result = ask_llm(api_url=api_url, question=final_question, brief=False)
                            _qa_perf064(
                                "H1",
                                "ask_route_done",
                                {
                                    "elapsed_ms": round((time.perf_counter() - t_api) * 1000, 2),
                                    "mode": ai_mode,
                                    "provider": result.get("provider"),
                                },
                            )
                            st.markdown(f"**Provider:** `{result.get('provider', '?')}`")
                            fc = result.get("fallback_chain")
                            if fc:
                                st.caption(f"LLM path: `{' → '.join(fc)}`")
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
                            ans = result.get("answer") or "_Empty response_"
                            st.markdown(ans)
                            fu = result.get("follow_up_questions") or []
                            if fu:
                                st.markdown("### What to ask next")
                                for qx in fu[:4]:
                                    st.markdown(f"- {qx}")
                            render_evidence_block(result.get("evidence_summary"), result.get("evidence_sources"))
                            parts_hist: list[str] = [
                                f"**Provider:** `{result.get('provider', '?')}`",
                            ]
                            if result.get("fallback_chain"):
                                parts_hist.append(
                                    "LLM path: `" + " → ".join(str(x) for x in result["fallback_chain"]) + "`"
                                )
                            if result.get("fallback_mode"):
                                parts_hist.append(f"Evidence mode: `{result.get('fallback_mode')}`")
                            if eq:
                                parts_hist.append(
                                    f"Evidence quality: score={eq.get('score')} "
                                    f"(trusted={eq.get('trusted_count')}, total={eq.get('count')})"
                                )
                            if result.get("insufficient_evidence"):
                                parts_hist.append("_(Insufficient evidence flag set.)_")
                            parts_hist.append(ans)
                            if fu:
                                parts_hist.append("### What to ask next")
                                for qx in fu[:4]:
                                    parts_hist.append(f"- {qx}")
                            hist_body = "\n\n".join(parts_hist) + "\n\n### Evidence (brief)\n" + (result.get("evidence_summary") or "")
                            st.session_state["doctor_qa_history"].append(
                                {
                                    "user_q": q,
                                    "answer_preview": (result.get("answer") or "")[:500],
                                    "body_md": hist_body,
                                }
                            )
                            _qa_perf064("H3", "qa_history_appended", {"history_len": len(st.session_state["doctor_qa_history"])})
                            st.rerun()
                except Exception as exc:
                    st.error(f"Q&A failed: {exc}")
                    _qa_perf064("H4", "ask_route_error", {"error_type": type(exc).__name__})

    if tab_admin is not None:
        with tab_admin:
            st.subheader("Operations and quality")
            st.caption("Aggregates from Snowflake audit; no patient narrative.")
            api_a = st.session_state["api_url"].rstrip("/")
            if st.button("Refresh metrics summary", key="admin_refresh_metrics"):
                try:
                    mr = requests.get(f"{api_a}/admin/metrics/summary", headers=_req_headers(), timeout=120)
                    mr.raise_for_status()
                    st.session_state["admin_metrics"] = mr.json()
                except Exception as exc:
                    st.error(f"Metrics failed: {exc}")
            m = st.session_state.get("admin_metrics")
            if m:
                st.json(m)
            if st.button("Download audit CSV (bounded)", key="admin_dl_audit"):
                try:
                    ar = requests.get(
                        f"{api_a}/admin/audit/export.csv",
                        params={"limit": 3000},
                        headers=_req_headers(),
                        timeout=180,
                    )
                    ar.raise_for_status()
                    st.session_state["admin_audit_csv"] = ar.content
                except Exception as exc:
                    st.error(f"Export failed: {exc}")
            if st.session_state.get("admin_audit_csv"):
                st.download_button(
                    "Save CSV",
                    data=st.session_state["admin_audit_csv"],
                    file_name="medassist_audit.csv",
                    mime="text/csv",
                    key="admin_save_csv",
                )
            st.markdown("See `docs/runbooks/` and Grafana folder `deploy/grafana/` for dashboards.")

    if not st.session_state.get("_analytics_ping"):
        try:
            requests.post(
                f"{st.session_state['api_url'].rstrip('/')}/analytics/event",
                params={"tab": "main", "event": "session_start"},
                headers=_req_headers(),
                timeout=5,
            )
        except Exception:
            pass
        st.session_state["_analytics_ping"] = True

    st.divider()
    st.caption(
        "For clinical support only. This tool does not replace physician judgment, protocol-based care, or emergency escalation."
    )


if __name__ == "__main__":
    main()
