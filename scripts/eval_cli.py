#!/usr/bin/env python3
"""Unified CLI evaluation for MedAssist answers and Snowflake data coverage."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

from scripts.eval_common import citation_quality, concept_overlap_score, eval_answer_structure
from src.snowflake_client import get_connection

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_OUT = Path("evals/reports/latest_eval.json")
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"
DEBUG_LOG_PATH = Path(
    "/Users/atharvakurlekar/Library/CloudStorage/OneDrive-NortheasternUniversity/Data Engineering/med/MedAssist.AI/.cursor/debug-f7c391.log"
)
DEBUG_SESSION_ID = "f7c391"

TABLE_THRESHOLDS = {
    "RAW.PUBMED_ARTICLES": 1000,
    "RAW.PMC_ARTICLES": 500,
    "RAW.OPENFDA_LABELS": 100,
    "RAW.OPENFDA_EVENTS": 100,
    "RAW.OPENFDA_NDC": 100,
    "RAW.RXNORM_DRUGS": 100,
    "RAW.WHO_DOCUMENTS": 50,
    "RAW.NCBI_BOOKSHELF": 100,
    "RAW.OPENSTAX_BOOKS": 100,
    "RAW.ORPHANET_DISEASES": 100,
    "RAW.ORPHANET_PHENOTYPES": 100,
    "RAW.ORPHANET_GENES": 100,
    "RAW.ORPHANET_WEB_PAGES": 10,
    "NORMALIZED.SYMPTOM_DISEASE_MAP": 1000,
    "RAW.FETCH_MANIFESTS": 1,
}

FALLBACK_CORTEX_MODELS = [
    "claude-sonnet-4-6",
    "openai-gpt-4.1",
    "openai-gpt-4.1-mini",
    "llama3.1-70b",
    "mistral-large2",
]

DEFAULT_RARE_CASES = [
    {
        "disease": "Wilson disease",
        "case": "17-year-old with tremor, jaundice, elevated AST/ALT, low ceruloplasmin, and behavioral changes.",
    },
    {
        "disease": "Acute intermittent porphyria",
        "case": "24-year-old with recurrent severe abdominal pain, neuropathy, anxiety, dark urine, and hyponatremia.",
    },
    {
        "disease": "Gaucher disease type 1",
        "case": "30-year-old with splenomegaly, thrombocytopenia, bone pain, and elevated ferritin.",
    },
    {
        "disease": "Fabry disease",
        "case": "28-year-old with burning extremity pain, angiokeratomas, proteinuria, and family history of early stroke.",
    },
    {
        "disease": "Hereditary angioedema",
        "case": "19-year-old with recurrent non-urticarial swelling episodes, abdominal attacks, and low C4.",
    },
]


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


def _load_questions(args: argparse.Namespace) -> list[str]:
    if args.questions:
        return [q.strip() for q in args.questions.read_text(encoding="utf-8").splitlines() if q.strip()]
    if args.question:
        return [args.question]
    raise ValueError("Provide either a single question or --questions file.")


def _ask_both(base_url: str, question: str) -> dict[str, Any]:
    start = time.perf_counter()
    r = requests.post(f"{base_url.rstrip('/')}/ask-both", json={"question": question}, timeout=180)
    r.raise_for_status()
    data = r.json()
    data["_latency_s"] = round(time.perf_counter() - start, 3)
    return data


def _online_consistency_score(question: str, answer: str, limit: int = 5) -> dict[str, Any]:
    try:
        es = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "retmode": "json", "retmax": str(limit), "sort": "relevance", "term": question},
            timeout=12,
        )
        es.raise_for_status()
        ids = ((es.json() or {}).get("esearchresult") or {}).get("idlist") or []
    except Exception as exc:
        return {"score": 0.0, "status": "error", "reason": str(exc)}

    if not ids:
        return {"score": 0.0, "status": "no_hits", "reason": "No live PubMed hits"}

    joined = " ".join(ids[:limit])
    mentions = sum(1 for pmid in ids[:limit] if pmid in (answer or ""))
    concept = concept_overlap_score(joined, answer or "")
    score = round((0.6 * concept) + (0.4 * (mentions / max(1, len(ids[:limit])))), 3)
    return {"score": score, "status": "ok", "pubmed_ids": ids[:limit]}


def run_answers_eval(args: argparse.Namespace) -> dict[str, Any]:
    questions = _load_questions(args)
    rows: list[dict[str, Any]] = []
    for question in questions:
        try:
            resp = _ask_both(args.base_url, question)
            g = resp.get("answer_gemini", "")
            c = resp.get("answer_cortex", "")
            g_struct = eval_answer_structure(g, "gemini")
            c_struct = eval_answer_structure(c, "cortex")
            g_score = round(
                0.5 * g_struct.get("section_completeness", 0.0)
                + 0.25 * concept_overlap_score(question, g)
                + 0.25 * citation_quality(g).get("score", 0.0),
                3,
            )
            c_score = round(
                0.5 * c_struct.get("section_completeness", 0.0)
                + 0.25 * concept_overlap_score(question, c)
                + 0.25 * citation_quality(c).get("score", 0.0),
                3,
            )
            winner = "tie"
            if g_score > c_score:
                winner = "gemini"
            elif c_score > g_score:
                winner = "cortex"
            row = {
                "question": question,
                "latency_s": resp.get("_latency_s"),
                "answer_gemini": g if args.include_answers else None,
                "answer_cortex": c if args.include_answers else None,
                "metrics": {
                    "gemini": {"structure": g_struct, "composite_score": g_score, "citations": citation_quality(g)},
                    "cortex": {"structure": c_struct, "composite_score": c_score, "citations": citation_quality(c)},
                    "winner": winner,
                },
            }
            if args.hybrid_online:
                row["metrics"]["gemini"]["online_consistency"] = _online_consistency_score(question, g)
                row["metrics"]["cortex"]["online_consistency"] = _online_consistency_score(question, c)
            rows.append(row)
        except Exception as exc:
            rows.append({"question": question, "error": str(exc)})

    winners = [r.get("metrics", {}).get("winner") for r in rows if r.get("metrics")]
    latencies = [r.get("latency_s") for r in rows if isinstance(r.get("latency_s"), (int, float))]
    return {
        "mode": "answers",
        "n_questions": len(questions),
        "summary": {
            "gemini_wins": winners.count("gemini"),
            "cortex_wins": winners.count("cortex"),
            "ties": winners.count("tie"),
            "latency_median_s": round(statistics.median(latencies), 3) if latencies else None,
        },
        "results": rows,
    }


def run_snowflake_coverage(args: argparse.Namespace) -> dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    rows: list[dict[str, Any]] = []
    failing: list[str] = []
    try:
        for table, threshold in TABLE_THRESHOLDS.items():
            count = 0
            status = "fail"
            reason = ""
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = int(cur.fetchone()[0] or 0)
                if count >= threshold:
                    status = "pass"
                elif count > 0:
                    status = "warn"
                    reason = f"Below threshold {threshold}"
                else:
                    reason = "Empty table"
            except Exception as exc:
                reason = str(exc)
            rows.append({"table": table, "count": count, "threshold": threshold, "status": status, "reason": reason})
            if status == "fail":
                failing.append(table)
    finally:
        cur.close()
        conn.close()

    if args.strict and failing:
        raise RuntimeError(f"Strict mode failed. Missing/empty tables: {', '.join(failing)}")

    return {
        "mode": "snowflake-coverage",
        "summary": {
            "pass": sum(1 for r in rows if r["status"] == "pass"),
            "warn": sum(1 for r in rows if r["status"] == "warn"),
            "fail": sum(1 for r in rows if r["status"] == "fail"),
        },
        "tables": rows,
    }


def _write_report(report: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _split_csv_models(value: str | None) -> list[str]:
    if not value:
        return []
    return [m.strip() for m in value.split(",") if m.strip()]


def _discover_cortex_models(cur) -> list[str]:
    """
    Attempt to discover available Cortex models in-account.
    Falls back to static list if discovery query is unavailable.
    """
    candidates: list[str] = []
    discovery_queries = [
        "SHOW MODELS IN SNOWFLAKE.CORTEX",
        "SHOW MODELS",
    ]
    for sql in discovery_queries:
        try:
            cur.execute(sql)
            rows = cur.fetchall() or []
            if not rows:
                continue
            for row in rows:
                text_cells = [str(c) for c in row if c is not None]
                joined = " ".join(text_cells).lower()
                if "cortex" in joined or "gpt" in joined or "claude" in joined or "llama" in joined or "mistral" in joined:
                    # heuristic: first column tends to model name
                    candidates.append(str(row[0]).strip())
            if candidates:
                break
        except Exception:
            continue
    seen = set()
    out: list[str] = []
    for m in candidates:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _cortex_complete(cur, model: str, prompt: str) -> str:
    cur.execute(
        "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS response",
        (model, prompt),
    )
    row = cur.fetchone()
    return (row[0] or "").strip() if row else ""


def _build_rare_disease_prompt(disease: str, case: str, ask: str) -> str:
    return (
        "You are a clinical decision-support assistant.\n"
        "Focus on one rare disease case and provide an evidence-aware, safe response.\n\n"
        f"Rare disease: {disease}\n"
        f"Case details: {case}\n"
        f"Task: {ask}\n\n"
        "Return markdown sections exactly:\n"
        "1. Summary\n"
        "2. Key Findings Supporting/Against\n"
        "3. Differential Considerations\n"
        "4. Recommended Diagnostics\n"
        "5. Initial Management and Safety Red Flags\n"
        "6. References (if unsure, state what data is missing)\n"
    )


def _fetch_source_papers(cur, disease: str, limit: int = 5) -> list[dict[str, str]]:
    like = f"%{disease}%"
    sql = """
        SELECT source, title, url, snippet
        FROM (
            SELECT 'pubmed' AS source, title,
                   'https://pubmed.ncbi.nlm.nih.gov/' || CAST(pmid AS VARCHAR) || '/' AS url,
                   LEFT(COALESCE(abstract, ''), 1200) AS snippet
            FROM RAW.PUBMED_ARTICLES
            WHERE title ILIKE %s OR abstract ILIKE %s
            UNION ALL
            SELECT 'pmc' AS source, title,
                   CASE WHEN pmcid LIKE 'PMC%%'
                        THEN 'https://www.ncbi.nlm.nih.gov/pmc/articles/' || pmcid || '/'
                        ELSE 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC' || pmcid || '/' END AS url,
                   LEFT(COALESCE(abstract, ''), 1200) AS snippet
            FROM RAW.PMC_ARTICLES
            WHERE title ILIKE %s OR abstract ILIKE %s
            UNION ALL
            SELECT 'ncbi_bookshelf' AS source, title, COALESCE(url, '') AS url,
                   LEFT(COALESCE(abstract, ''), 1200) AS snippet
            FROM RAW.NCBI_BOOKSHELF
            WHERE title ILIKE %s OR abstract ILIKE %s
            UNION ALL
            SELECT 'openstax' AS source, title, COALESCE(source_url, '') AS url,
                   LEFT(COALESCE(content, ''), 1200) AS snippet
            FROM RAW.OPENSTAX_BOOKS
            WHERE title ILIKE %s OR content ILIKE %s
        ) s
        LIMIT %s
    """
    try:
        cur.execute(sql, (like, like, like, like, like, like, like, like, limit))
        rows = cur.fetchall() or []
    except Exception:
        rows = []
    out: list[dict[str, str]] = []
    for source, title, url, snippet in rows:
        out.append(
            {
                "source": str(source or ""),
                "title": str(title or ""),
                "url": str(url or ""),
                "snippet": str(snippet or ""),
            }
        )
    return out


def _format_sources_for_prompt(papers: list[dict[str, str]]) -> str:
    if not papers:
        return "No source papers retrieved from Snowflake for this disease."
    lines = ["Use these source papers as supporting context:"]
    for i, p in enumerate(papers, start=1):
        lines.append(f"{i}. [{p.get('source')}] {p.get('title')} ({p.get('url')})")
    return "\n".join(lines)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if raw.startswith("```"):
        chunks = raw.split("```")
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.lower().startswith("json"):
                chunk = chunk[4:].strip()
            raw = chunk
            break
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_candidate_facts(cur, model: str, disease: str, paper: dict[str, str]) -> list[dict[str, Any]]:
    prompt = (
        "Extract concise clinical facts for a gold-standard checklist.\n"
        "Return JSON only with key facts, where facts is an array of objects having: "
        "fact, category(one of diagnosis|differential|diagnostics|treatment|safety), confidence(0-1), safety_critical(boolean).\n"
        "Use only the paper info provided.\n\n"
        f"Disease: {disease}\n"
        f"Source: {paper.get('source')}\n"
        f"Title: {paper.get('title')}\n"
        f"URL: {paper.get('url')}\n"
        f"Snippet: {paper.get('snippet')}\n"
    )
    raw = _cortex_complete(cur, model, prompt)
    obj = _extract_json_object(raw) or {}
    facts = obj.get("facts") or []
    out: list[dict[str, Any]] = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        text = str(f.get("fact") or "").strip()
        if not text:
            continue
        out.append(
            {
                "fact": text,
                "category": str(f.get("category") or "diagnosis").strip().lower(),
                "confidence": float(f.get("confidence") or 0.5),
                "safety_critical": bool(f.get("safety_critical")),
                "source_url": paper.get("url", ""),
                "source_title": paper.get("title", ""),
            }
        )
    return out


def _fact_similarity(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]{3,}", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]{3,}", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def _build_consensus_gold_list(candidates: list[dict[str, Any]], max_facts: int = 15) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for c in candidates:
        placed = False
        for cl in clusters:
            if _fact_similarity(c["fact"], cl["representative"]) >= 0.55:
                cl["items"].append(c)
                placed = True
                break
        if not placed:
            clusters.append({"representative": c["fact"], "items": [c]})

    gold: list[dict[str, Any]] = []
    for cl in clusters:
        items = cl["items"]
        distinct_sources = {i["source_url"] for i in items if i.get("source_url")}
        avg_conf = sum(float(i.get("confidence") or 0.5) for i in items) / max(1, len(items))
        # Consensus gate: at least two paper references, or high-confidence extraction.
        if len(distinct_sources) < 2 and avg_conf < 0.85:
            continue
        cat = Counter(i.get("category") or "diagnosis" for i in items).most_common(1)[0][0]
        safety = any(bool(i.get("safety_critical")) for i in items) or cat == "safety"
        weight = 1.0 + (0.75 if safety else 0.0) + (0.25 if len(distinct_sources) >= 3 else 0.0)
        refs = [
            {"title": i.get("source_title", ""), "url": i.get("source_url", "")}
            for i in items
            if i.get("source_url")
        ]
        seen = set()
        dedup_refs = []
        for r in refs:
            u = r["url"]
            if u and u not in seen:
                seen.add(u)
                dedup_refs.append(r)
        gold.append(
            {
                "fact": cl["representative"],
                "category": cat,
                "weight": round(weight, 3),
                "priority": "high" if safety else "medium",
                "safety_critical": safety,
                "evidence_refs": dedup_refs[:5],
            }
        )
    gold.sort(key=lambda x: (x["priority"] != "high", -x["weight"]))
    return gold[:max_facts]


def _score_answer_against_gold(cur, judge_model: str, answer: str, gold_list: list[dict[str, Any]]) -> dict[str, Any]:
    simplified_gold = [
        {
            "id": idx,
            "fact": g["fact"],
            "category": g["category"],
            "weight": g["weight"],
            "safety_critical": g["safety_critical"],
        }
        for idx, g in enumerate(gold_list)
    ]
    prompt = (
        "Evaluate this answer against the provided gold facts.\n"
        "Return JSON only with keys: covered_fact_ids (int[]), contradicted_fact_ids (int[]), safety_covered_ids (int[]), explanation.\n\n"
        f"GOLD_FACTS={json.dumps(simplified_gold, ensure_ascii=True)}\n\n"
        f"ANSWER={answer[:14000]}\n"
    )
    raw = _cortex_complete(cur, judge_model, prompt)
    parsed = _extract_json_object(raw) or {}
    covered = [int(i) for i in (parsed.get("covered_fact_ids") or []) if str(i).isdigit()]
    contradicted = [int(i) for i in (parsed.get("contradicted_fact_ids") or []) if str(i).isdigit()]
    safety_cov = [int(i) for i in (parsed.get("safety_covered_ids") or []) if str(i).isdigit()]
    return {
        "covered_fact_ids": covered,
        "contradicted_fact_ids": contradicted,
        "safety_covered_ids": safety_cov,
        "explanation": parsed.get("explanation") or "",
        "judge_raw": raw,
    }


def _compute_weighted_scores(gold_list: list[dict[str, Any]], judge_result: dict[str, Any], answer: str) -> dict[str, Any]:
    weights = [float(g.get("weight") or 1.0) for g in gold_list]
    total_w = sum(weights) or 1.0
    safety_idx = [i for i, g in enumerate(gold_list) if g.get("safety_critical")]
    safety_total = sum(weights[i] for i in safety_idx) or 1.0

    covered = set(judge_result.get("covered_fact_ids") or [])
    contrad = set(judge_result.get("contradicted_fact_ids") or [])
    safety_cov = set(judge_result.get("safety_covered_ids") or [])

    coverage = sum(weights[i] for i in covered if 0 <= i < len(weights)) / total_w
    safety = sum(weights[i] for i in safety_cov if i in safety_idx and 0 <= i < len(weights)) / safety_total
    contrad_penalty = sum(weights[i] for i in contrad if 0 <= i < len(weights)) / total_w
    structure = float(eval_answer_structure(answer or "", "candidate").get("section_completeness", 0.0))
    final_score = (0.55 * coverage) + (0.25 * safety) + (0.20 * structure) - (0.40 * contrad_penalty)
    final_score = max(0.0, min(1.0, final_score))

    return {
        "gold_coverage_score": round(coverage, 3),
        "safety_score": round(safety, 3),
        "contradiction_penalty": round(contrad_penalty, 3),
        "structure_score": round(structure, 3),
        "final_score": round(final_score, 3),
    }


def _judge_rare_answers(
    cur,
    judge_model: str,
    disease: str,
    case: str,
    answers: dict[str, str],
) -> dict[str, Any]:
    packed = []
    for model, text in answers.items():
        packed.append(f"MODEL: {model}\nANSWER:\n{text[:12000]}")
    judge_prompt = (
        "You are the final evaluator for rare-disease clinical answers.\n"
        "Use strict criteria: clinical accuracy, safety, actionability, uncertainty handling, and structure quality.\n"
        "Rank all models from best to worst.\n"
        "Return JSON only with keys: winner, ranking, scores, rationale.\n"
        "scores must be an object where each model gets 0-10.\n\n"
        f"Disease: {disease}\n"
        f"Case: {case}\n\n"
        + "\n\n".join(packed)
    )
    raw = _cortex_complete(cur, judge_model, judge_prompt)
    # region agent log
    _debug_log(
        run_id="eval-debug",
        hypothesis_id="H1",
        location="scripts/eval_cli.py:_judge_rare_answers:raw",
        message="judge_raw_received",
        data={
            "judge_model": judge_model,
            "raw_prefix": (raw or "")[:180],
            "raw_len": len(raw or ""),
            "n_answers": len(answers),
        },
    )
    # endregion
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        for part in parts:
            frag = part.strip()
            if not frag:
                continue
            if frag.lower().startswith("json"):
                frag = frag[4:].strip()
            cleaned = frag
            break
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            # region agent log
            _debug_log(
                run_id="eval-debug",
                hypothesis_id="H2",
                location="scripts/eval_cli.py:_judge_rare_answers:json",
                message="judge_json_parse_success",
                data={
                    "keys": sorted(list(parsed.keys())),
                    "winner": parsed.get("winner"),
                    "ranking_len": len(parsed.get("ranking") or []),
                },
            )
            # endregion
            return {"judge_model": judge_model, "raw": raw, "parsed": parsed}
    except Exception:
        pass
    valid_models = set(answers.keys())
    fallback: dict[str, Any] = {}
    winner_match = re.search(
        r"\*{0,2}winner\*{0,2}\s*[:\-]\*{0,2}\s*([A-Za-z0-9\.\-_]+)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if winner_match:
        winner = winner_match.group(1).strip()
        if winner in valid_models:
            fallback["winner"] = winner
    rank_match = re.search(
        r"\*{0,2}ranking\*{0,2}\s*[:\-]?\s*(.*?)(?:\n\s*\*{0,2}scores\*{0,2}\s*[:\-]?|\Z)",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if rank_match:
        block = rank_match.group(1)
        ranked = re.findall(r"\d+\.\s*([A-Za-z0-9\.\-_]+)", block)
        ranked = [m for m in ranked if m in valid_models]
        if ranked:
            fallback["ranking"] = ranked
            if "winner" not in fallback:
                fallback["winner"] = ranked[0]
    score_pairs = re.findall(r"([A-Za-z0-9\.\-_]+)\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*/?\s*10?", cleaned)
    if score_pairs:
        scores = {m: float(s) for m, s in score_pairs if m in valid_models}
        if scores:
            fallback["scores"] = scores
    if fallback:
        # region agent log
        _debug_log(
            run_id="eval-debug",
            hypothesis_id="H3",
            location="scripts/eval_cli.py:_judge_rare_answers:fallback",
            message="judge_fallback_parse_used",
            data={
                "fallback_keys": sorted(list(fallback.keys())),
                "winner": fallback.get("winner"),
                "ranking": fallback.get("ranking"),
                "scores_keys": sorted(list((fallback.get("scores") or {}).keys())),
            },
        )
        # endregion
        return {"judge_model": judge_model, "raw": raw, "parsed": fallback}
    # region agent log
    _debug_log(
        run_id="eval-debug",
        hypothesis_id="H4",
        location="scripts/eval_cli.py:_judge_rare_answers:none",
        message="judge_parse_failed",
        data={"raw_prefix": cleaned[:180], "judge_model": judge_model},
    )
    # endregion
    return {"judge_model": judge_model, "raw": raw, "parsed": None}


def run_rare_judge(args: argparse.Namespace) -> dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        model_list = _split_csv_models(args.models)
        if args.all_snowflake_models:
            discovered = _discover_cortex_models(cur)
            if discovered:
                model_list = discovered
        if not model_list:
            model_list = _split_csv_models(os.environ.get("CORTEX_EVAL_MODELS")) or FALLBACK_CORTEX_MODELS

        papers = _fetch_source_papers(cur, args.disease, limit=args.paper_limit)
        prompt = _build_rare_disease_prompt(args.disease, args.case, args.task) + "\n\n" + _format_sources_for_prompt(papers)
        answers: dict[str, str] = {}
        failures: dict[str, str] = {}
        for model in model_list:
            try:
                answers[model] = _cortex_complete(cur, model, prompt)
            except Exception as exc:
                failures[model] = str(exc)

        judge = _judge_rare_answers(cur, args.judge_model, args.disease, args.case, answers)
        parsed = judge.get("parsed") or {}
        winner = parsed.get("winner")
        ranking = parsed.get("ranking")
        return {
            "mode": "rare-judge",
            "disease": args.disease,
            "judge_model": args.judge_model,
            "models_requested": model_list,
            "models_answered": list(answers.keys()),
            "models_failed": failures,
            "winner": winner,
            "ranking": ranking,
            "judge": judge,
            "papers": papers,
            "answers": answers if args.include_answers else {m: f"{len(a)} chars" for m, a in answers.items()},
        }
    finally:
        cur.close()
        conn.close()


def _scoreboard_from_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    points: dict[str, int] = {}
    reasons: dict[str, list[str]] = {}
    for c in cases:
        parsed = ((c.get("judge") or {}).get("parsed") or {})
        ranking = parsed.get("ranking") or []
        rationale = parsed.get("rationale")
        for i, model in enumerate(ranking):
            points[model] = points.get(model, 0) + max(0, len(ranking) - i)
            if i == 0 and rationale:
                reasons.setdefault(model, []).append(str(rationale)[:400])
        # region agent log
        _debug_log(
            run_id="eval-debug",
            hypothesis_id="H5",
            location="scripts/eval_cli.py:_scoreboard_from_cases:case",
            message="scoreboard_case_ranking",
            data={
                "disease": c.get("disease"),
                "winner": parsed.get("winner"),
                "ranking": ranking,
            },
        )
        # endregion
    ordered = sorted(points.items(), key=lambda x: x[1], reverse=True)
    overall_winner = ordered[0][0] if ordered else None
    return {
        "points": points,
        "ranking": [m for m, _ in ordered],
        "overall_winner": overall_winner,
        "winner_reason": (reasons.get(overall_winner) or ["No rationale available"])[0] if overall_winner else None,
    }


def run_rare_batch(args: argparse.Namespace) -> dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        model_list = _split_csv_models(args.models) or FALLBACK_CORTEX_MODELS[:3]
        case_count = max(3, min(5, args.case_count))
        selected_cases = DEFAULT_RARE_CASES[:case_count]
        case_results: list[dict[str, Any]] = []

        for case in selected_cases:
            disease = case["disease"]
            vignette = case["case"]
            papers = _fetch_source_papers(cur, disease, limit=args.paper_limit)
            prompt = _build_rare_disease_prompt(disease, vignette, args.task) + "\n\n" + _format_sources_for_prompt(papers)
            answers: dict[str, str] = {}
            failures: dict[str, str] = {}
            for model in model_list:
                try:
                    answers[model] = _cortex_complete(cur, model, prompt)
                except Exception as exc:
                    failures[model] = str(exc)
            judge = _judge_rare_answers(cur, args.judge_model, disease, vignette, answers)
            parsed = judge.get("parsed") or {}
            case_results.append(
                {
                    "disease": disease,
                    "case": vignette,
                    "papers": papers,
                    "models_answered": list(answers.keys()),
                    "models_failed": failures,
                    "winner": parsed.get("winner"),
                    "ranking": parsed.get("ranking"),
                    "judge": judge,
                    "answers": answers if args.include_answers else {m: f"{len(a)} chars" for m, a in answers.items()},
                }
            )

        aggregate = _scoreboard_from_cases(case_results)
        return {
            "mode": "rare-batch-judge",
            "judge_model": args.judge_model,
            "models_requested": model_list,
            "n_cases": len(case_results),
            "overall_winner": aggregate["overall_winner"],
            "overall_ranking": aggregate["ranking"],
            "overall_points": aggregate["points"],
            "overall_winner_reason": aggregate["winner_reason"],
            "cases": case_results,
        }
    finally:
        cur.close()
        conn.close()


def run_rare_gold_batch(args: argparse.Namespace) -> dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        model_list = _split_csv_models(args.models) or FALLBACK_CORTEX_MODELS[:3]
        case_count = max(3, min(5, args.case_count))
        selected_cases = DEFAULT_RARE_CASES[:case_count]
        cases: list[dict[str, Any]] = []
        points: dict[str, int] = {}
        mean_scores: dict[str, list[float]] = {m: [] for m in model_list}

        for case in selected_cases:
            disease = case["disease"]
            vignette = case["case"]
            papers = _fetch_source_papers(cur, disease, limit=args.paper_limit)

            candidates: list[dict[str, Any]] = []
            for p in papers:
                candidates.extend(_extract_candidate_facts(cur, args.gold_model, disease, p))
            gold_list = _build_consensus_gold_list(candidates, max_facts=15)
            if not gold_list:
                # Fallback minimal facts to prevent empty scoring set.
                gold_list = [
                    {
                        "fact": f"Core diagnostic and management priorities for {disease}",
                        "category": "diagnosis",
                        "weight": 1.0,
                        "priority": "medium",
                        "safety_critical": False,
                        "evidence_refs": [{"title": p.get("title", ""), "url": p.get("url", "")} for p in papers[:2]],
                    }
                ]

            prompt = _build_rare_disease_prompt(disease, vignette, args.task) + "\n\n" + _format_sources_for_prompt(papers)
            answers: dict[str, str] = {}
            model_scores: dict[str, Any] = {}
            model_failures: dict[str, str] = {}

            for model in model_list:
                try:
                    answer = _cortex_complete(cur, model, prompt)
                    answers[model] = answer
                    judge_result = _score_answer_against_gold(cur, args.judge_model, answer, gold_list)
                    metrics = _compute_weighted_scores(gold_list, judge_result, answer)
                    model_scores[model] = {**metrics, "judge_result": judge_result}
                    mean_scores[model].append(metrics["final_score"])
                except Exception as exc:
                    model_failures[model] = str(exc)

            ordered = sorted(model_scores.items(), key=lambda kv: kv[1]["final_score"], reverse=True)
            disease_ranking = [m for m, _ in ordered]
            disease_winner = disease_ranking[0] if disease_ranking else None
            for i, model in enumerate(disease_ranking):
                points[model] = points.get(model, 0) + max(0, len(disease_ranking) - i)

            reason = "No winner (all models failed)."
            if len(ordered) >= 2:
                top_m, top_s = ordered[0]
                sec_m, sec_s = ordered[1]
                reason = (
                    f"{top_m} won with higher final_score ({top_s['final_score']}) than {sec_m} ({sec_s['final_score']}); "
                    f"coverage={top_s['gold_coverage_score']} safety={top_s['safety_score']} contradiction_penalty={top_s['contradiction_penalty']}."
                )
            elif len(ordered) == 1:
                top_m, top_s = ordered[0]
                reason = f"{top_m} only successful model with final_score={top_s['final_score']}."

            cases.append(
                {
                    "disease": disease,
                    "case": vignette,
                    "papers": papers,
                    "disease_gold_list": gold_list,
                    "model_scores": model_scores,
                    "models_failed": model_failures,
                    "winner": disease_winner,
                    "ranking": disease_ranking,
                    "disease_winner_reason": reason,
                    "answers": answers if args.include_answers else {m: f"{len(a)} chars" for m, a in answers.items()},
                }
            )

        overall = sorted(
            model_list,
            key=lambda m: (points.get(m, 0), statistics.mean(mean_scores[m]) if mean_scores[m] else 0.0),
            reverse=True,
        )
        overall_winner = overall[0] if overall else None
        overall_reason = (
            f"{overall_winner} has highest aggregate points ({points.get(overall_winner, 0)}) and mean final_score "
            f"{round(statistics.mean(mean_scores[overall_winner]), 3) if overall_winner and mean_scores[overall_winner] else 0.0}."
            if overall_winner
            else None
        )
        return {
            "mode": "rare-gold-batch",
            "judge_model": args.judge_model,
            "gold_model": args.gold_model,
            "models_requested": model_list,
            "n_cases": len(cases),
            "overall_winner": overall_winner,
            "overall_ranking": overall,
            "overall_points": points,
            "overall_mean_final_score": {
                m: round(statistics.mean(v), 3) if v else 0.0 for m, v in mean_scores.items()
            },
            "overall_winner_reason": overall_reason,
            "cases": cases,
        }
    finally:
        cur.close()
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="MedAssist evaluation CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_answers = sub.add_parser("answers", help="Compare live /ask-both outputs.")
    p_answers.add_argument("question", nargs="?", default=None)
    p_answers.add_argument("--questions", type=Path, default=None)
    p_answers.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p_answers.add_argument("--hybrid-online", action="store_true", help="Add online PubMed consistency metrics.")
    p_answers.add_argument("--include-answers", action="store_true", help="Include full model answers in report.")
    p_answers.add_argument("--out", type=Path, default=DEFAULT_OUT)

    p_cov = sub.add_parser("snowflake-coverage", help="Audit all supported Snowflake source tables.")
    p_cov.add_argument("--strict", action="store_true", help="Exit non-zero if any table fails threshold.")
    p_cov.add_argument("--out", type=Path, default=DEFAULT_OUT)

    p_rare = sub.add_parser("rare-judge", help="Single rare-disease eval: multiple Snowflake models + Cortex judge.")
    p_rare.add_argument("--disease", required=True, help="Rare disease name.")
    p_rare.add_argument(
        "--case",
        required=True,
        help="Single patient vignette/case text for evaluation.",
    )
    p_rare.add_argument(
        "--task",
        default="Provide differential confidence, diagnosis strategy, and immediate management priorities.",
        help="Task instruction shown to all models.",
    )
    p_rare.add_argument(
        "--models",
        default="",
        help="Comma-separated Cortex model IDs to evaluate (e.g. 'openai-gpt-4.1,claude-sonnet-4-6').",
    )
    p_rare.add_argument(
        "--all-snowflake-models",
        action="store_true",
        help="Try to discover and evaluate all available Cortex models in your Snowflake account.",
    )
    p_rare.add_argument("--paper-limit", type=int, default=5, help="How many Snowflake source papers to include per case.")
    p_rare.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Cortex model used as final judge.")
    p_rare.add_argument("--include-answers", action="store_true", help="Include full generated answers in report.")
    p_rare.add_argument("--out", type=Path, default=DEFAULT_OUT)

    p_batch = sub.add_parser("rare-batch", help="Evaluate 3-5 rare disease cases and report overall winner.")
    p_batch.add_argument("--case-count", type=int, default=5, help="Number of rare diseases to evaluate (3 to 5).")
    p_batch.add_argument(
        "--task",
        default="Provide differential confidence, diagnosis strategy, and immediate management priorities.",
        help="Task instruction shown to all models.",
    )
    p_batch.add_argument(
        "--models",
        default="claude-sonnet-4-6,openai-gpt-4.1,mistral-large2,llama3.1-70b",
        help="Comma-separated Cortex model IDs to evaluate.",
    )
    p_batch.add_argument("--paper-limit", type=int, default=5, help="How many Snowflake source papers to include per case.")
    p_batch.add_argument("--judge-model", default="llama3.1-70b", help="Judge model (keep outside compared set).")
    p_batch.add_argument("--include-answers", action="store_true", help="Include full generated answers in report.")
    p_batch.add_argument("--out", type=Path, default=DEFAULT_OUT)

    p_gold = sub.add_parser("rare-gold-batch", help="Gold-list based evaluation over 3-5 rare diseases.")
    p_gold.add_argument("--case-count", type=int, default=5, help="Number of diseases (3 to 5).")
    p_gold.add_argument(
        "--task",
        default="Provide differential confidence, diagnosis strategy, and immediate management priorities.",
        help="Task instruction shown to all models.",
    )
    p_gold.add_argument(
        "--models",
        default="claude-sonnet-4-6,openai-gpt-4.1,mistral-large2,llama3.1-70b",
        help="Comma-separated candidate model IDs.",
    )
    p_gold.add_argument("--gold-model", default="claude-sonnet-4-6", help="Model used to extract candidate gold facts.")
    p_gold.add_argument("--judge-model", default="llama3.1-70b", help="Model used to score answers against gold list.")
    p_gold.add_argument("--paper-limit", type=int, default=5, help="Papers per disease for gold list generation.")
    p_gold.add_argument("--include-answers", action="store_true", help="Include full generated answers in report.")
    p_gold.add_argument("--out", type=Path, default=DEFAULT_OUT)

    args = parser.parse_args()
    if args.command == "answers":
        report = run_answers_eval(args)
    elif args.command == "rare-judge":
        report = run_rare_judge(args)
    elif args.command == "rare-batch":
        report = run_rare_batch(args)
    elif args.command == "rare-gold-batch":
        report = run_rare_gold_batch(args)
    else:
        report = run_snowflake_coverage(args)

    _write_report(report, args.out)
    print(json.dumps(report.get("summary", {}), indent=2))
    print(f"Report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
