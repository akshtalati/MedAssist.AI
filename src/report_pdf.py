"""Build a clinical summary PDF for an encounter (research / demo use only)."""

from __future__ import annotations

from typing import Any

from fpdf import FPDF


class _MedPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, "MedAssist.AI — Clinical summary (decision support only)", ln=True)
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def build_encounter_pdf(
    *,
    encounter_id: str,
    encounter: dict[str, Any],
    assessment_markdown: str | None,
    top_candidates: list[dict[str, Any]] | None,
    evidence_summary: str | None,
    kg_version: str | None,
    kg_build_id: str | None,
    fallback_mode: str | None,
    org_id: str | None,
) -> bytes:
    pdf = _MedPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)

    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(0, 6, "Disclaimer")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        0,
        5,
        "For education, research, and engineering demonstration only. Not a medical device. "
        "Does not replace professional judgment or emergency care.",
    )
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Encounter metadata", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, f"Encounter ID: {encounter_id}")
    if org_id:
        pdf.multi_cell(0, 5, f"Organization: {org_id}")
    if kg_version or kg_build_id:
        pdf.multi_cell(0, 5, f"KG version: {kg_version or '—'}  |  Build: {kg_build_id or '—'}")
    if fallback_mode:
        pdf.multi_cell(0, 5, f"Evidence mode: {fallback_mode}")
    pdf.ln(2)

    age = encounter.get("age")
    sex = encounter.get("sex")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Patient context (as entered)", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, f"Age / sex: {age} / {sex}")
    symptoms = encounter.get("symptoms") or []
    if symptoms:
        lines = ", ".join(str(s.get("symptom", "")) for s in symptoms if s.get("symptom"))
        pdf.multi_cell(0, 5, f"Symptoms: {lines}")
    hist = (encounter.get("history_summary") or "").strip()
    if hist:
        pdf.multi_cell(0, 5, f"History: {hist[:2000]}")
    pdf.ln(2)

    if top_candidates:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Prioritized differential (top candidates)", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for i, c in enumerate(top_candidates[:12], start=1):
            name = (c.get("disease_name") or "").strip() or "—"
            score = c.get("score")
            sc = f"{float(score):.3f}" if isinstance(score, (int, float)) else "—"
            pdf.multi_cell(0, 5, f"{i}. {name} (score {sc})")
        pdf.ln(2)

    if evidence_summary:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Evidence (brief)", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, evidence_summary[:4000])
        pdf.ln(2)

    if assessment_markdown:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Assessment", ln=True)
        pdf.set_font("Helvetica", "", 9)
        # Strip markdown noise lightly — keep text readable
        text = assessment_markdown.replace("###", "").replace("##", "").replace("#", "")
        pdf.multi_cell(0, 5, text[:12000])

    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return out.encode("latin-1")
