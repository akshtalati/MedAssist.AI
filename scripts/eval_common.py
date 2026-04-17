#!/usr/bin/env python3
"""Shared evaluation helpers for MedAssist CLI scripts."""

from __future__ import annotations

import re
from collections import Counter

REQUIRED_SECTIONS = [
    ("summary", r"Summary"),
    ("differential", r"Common\s+Differential\s+Diagnosis|Differential\s+Diagnosis"),
    ("rare_diseases", r"Rare\s+Diseases|Orphanet"),
    ("red_flags", r"Red[- ]Flag|Red-Flag\s+Features"),
    ("next_steps", r"Suggested\s+Next\s+Steps|Next\s+Steps"),
    ("references", r"References|\*\*References\*\*"),
]

CLINICAL_TOKENS = {
    "fever",
    "cough",
    "rash",
    "headache",
    "pain",
    "dyspnea",
    "vomiting",
    "nausea",
    "diarrhea",
    "seizure",
    "fatigue",
    "hypotension",
    "tachycardia",
    "anemia",
    "infection",
}


def eval_answer_structure(text: str, label: str) -> dict:
    if not (text or "").strip():
        return {"label": label, "ok": False, "error": "empty answer", "sections": {}}

    body = text.strip()
    sections = {name: bool(re.search(pattern, body, re.IGNORECASE)) for name, pattern in REQUIRED_SECTIONS}
    word_count = len(body.split())
    completeness = sum(1 for present in sections.values() if present) / max(1, len(REQUIRED_SECTIONS))
    return {
        "label": label,
        "ok": all(sections.values()),
        "sections": sections,
        "word_count": word_count,
        "has_content": word_count >= 50,
        "section_completeness": round(completeness, 3),
    }


def concept_overlap_score(question: str, answer: str) -> float:
    q_tokens = _important_tokens(question)
    a_tokens = _important_tokens(answer)
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & a_tokens) / len(q_tokens)
    return round(overlap, 3)


def citation_quality(answer: str) -> dict:
    urls = re.findall(r"https?://[^\s\)]+", answer or "", flags=re.IGNORECASE)
    counts = Counter(_source_bucket(u) for u in urls)
    trusted = counts.get("trusted", 0)
    total = len(urls)
    score = (trusted / total) if total else 0.0
    return {
        "total_urls": total,
        "trusted_urls": trusted,
        "score": round(score, 3),
    }


def _important_tokens(text: str) -> set[str]:
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", (text or "").lower())
    stop = {"what", "which", "with", "from", "into", "when", "where", "that", "this", "have", "will"}
    return {t for t in toks if t not in stop and (t in CLINICAL_TOKENS or len(t) >= 5)}


def _source_bucket(url: str) -> str:
    u = (url or "").lower()
    trusted_hosts = ("pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov/pmc", "nih.gov", "who.int", "orpha.net")
    return "trusted" if any(host in u for host in trusted_hosts) else "other"
