"""Assessment response contract and safety formatting helpers."""

from __future__ import annotations

from typing import Any


REQUIRED_SECTIONS = [
    "summary",
    "differential",
    "red_flags",
    "next_steps",
    "follow_up_questions",
    "confidence",
    "insufficient_evidence",
]


def normalize_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure required contract fields exist with deterministic fallbacks."""
    out = dict(payload or {})
    out.setdefault("summary", "Evidence reviewed from available context.")
    out.setdefault("differential", [])
    out.setdefault("red_flags", [])
    out.setdefault("next_steps", [])
    out.setdefault("follow_up_questions", [])
    out.setdefault("confidence", 0.35)
    out.setdefault("insufficient_evidence", False)
    out.setdefault("uncertainty_note", "")
    out.setdefault("contraindications", [])
    return out


def validate_contract(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    missing = [k for k in REQUIRED_SECTIONS if k not in payload]
    if missing:
        return False, missing
    return True, []


def contract_to_markdown(payload: dict[str, Any]) -> str:
    p = normalize_contract(payload)
    lines: list[str] = [
        "## Updated Clinical Assessment",
        "",
        "### Summary",
        f"- {p['summary']}",
        "",
        "### Prioritized Differential",
    ]
    for item in p["differential"][:8]:
        lines.append(
            f"- {item.get('disease_name', 'Unknown')} ({item.get('disease_code') or 'no-code'})"
            f" - score {float(item.get('score', 0.0)):.2f}; {item.get('rationale', '')}"
        )
    if not p["differential"]:
        lines.append("- None from current context.")

    lines.extend(["", "### Red Flags"])
    for rf in p["red_flags"][:6]:
        lines.append(f"- {rf}")
    if not p["red_flags"]:
        lines.append("- No immediate rule-based red flags detected.")

    lines.extend(["", "### Next Steps"])
    for s in p["next_steps"][:6]:
        lines.append(f"- {s}")
    if not p["next_steps"]:
        lines.append("- Continue focused history and targeted testing.")

    lines.extend(["", "### Follow-up Questions"])
    for q in p["follow_up_questions"][:6]:
        lines.append(f"- {q}")
    if not p["follow_up_questions"]:
        lines.append("- Clarify symptom onset and progression.")

    lines.extend(
        [
            "",
            "### Confidence",
            f"- Score: {float(p['confidence']):.2f}",
            f"- Insufficient evidence: {'yes' if p['insufficient_evidence'] else 'no'}",
        ]
    )
    if p.get("uncertainty_note"):
        lines.append(f"- Uncertainty note: {p['uncertainty_note']}")
    if p.get("contraindications"):
        lines.append("- Contraindications:")
        for c in p["contraindications"][:5]:
            lines.append(f"  - {c}")
    return "\n".join(lines)
