#!/usr/bin/env python3
"""
Evaluate MedAssist.AI answers without changing any app code.

How it works:
- Calls your running API (POST /ask-both or POST /ask) over HTTP.
- Runs simple checks on the response text (required sections, length).
- Optionally saves full answers and a summary to a file.
- When comparing two answers (ask-both), optionally uses an LLM judge (Vertex Gemini)
  to say which answer is better and why.
- Use --no-judge to skip the judge.

Usage:
  # Ensure API is running: uvicorn api.main:app --reload
  python scripts/eval_answers.py "Child with fever and cough - differential?"
  python scripts/eval_answers.py --questions questions.txt --out results.json
  python scripts/eval_answers.py "Your question" --no-judge   # skip LLM comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Default base URL for the API (no changes to your code)
DEFAULT_BASE_URL = "http://127.0.0.1:8000"

from scripts.eval_common import eval_answer_structure


def call_ask_both(base_url: str, question: str, brief: bool = False) -> dict:
    """POST /ask-both -> { answer_gemini, answer_cortex }."""
    r = requests.post(
        f"{base_url}/ask-both",
        json={"question": question, "brief": brief},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def call_ask(base_url: str, question: str, brief: bool = False) -> dict:
    """POST /ask -> { answer }."""
    r = requests.post(
        f"{base_url}/ask",
        json={"question": question, "brief": brief},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def eval_answer(text: str, label: str) -> dict:
    """Backward-compatible alias to shared evaluator."""
    return eval_answer_structure(text, label)


def judge_answers(question: str, answer_gemini: str, answer_cortex: str) -> dict | None:
    """
    Use Vertex AI Gemini to compare the two answers and decide which is better.
    Returns {"winner": "Gemini"|"Cortex"|"Tie", "reason": "..."} or None if judge unavailable.
    """
    if not HAS_GENAI or not answer_gemini.strip() or not answer_cortex.strip():
        return None
    try:
        client = genai.Client(
            vertexai=True,
            project="medassistai-488422",
            location="global",
        )
        prompt = (
            "You are an expert medical educator evaluating two AI-generated answers to a clinical question.\n\n"
            "QUESTION:\n" + question + "\n\n"
            "ANSWER A (Gemini):\n" + answer_gemini[:12000] + "\n\n"
            "ANSWER B (Cortex):\n" + answer_cortex[:12000] + "\n\n"
            "Compare them on: (1) medical accuracy and use of evidence, (2) completeness of the 5 required sections, "
            "(3) clarity and usefulness for a clinician, (4) appropriate red flags and next steps.\n\n"
            "Output exactly in this format:\n"
            "Winner: A or B or Tie\n"
            "Reason: [2-4 sentences explaining why]"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=1024),
        )
        text = ""
        if response.candidates:
            c = response.candidates[0]
            if getattr(c, "content", None) and getattr(c.content, "parts", None):
                for part in c.content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text
        if not text:
            text = getattr(response, "text", None) or ""
        text = text.strip()
        winner = "Tie"
        if "Winner:" in text:
            line = [l for l in text.split("\n") if l.strip().lower().startswith("winner:")]
            if line:
                w = line[0].split(":", 1)[1].strip().upper()
                if "A" in w or "GEMINI" in w:
                    winner = "Gemini"
                elif "B" in w or "CORTEX" in w:
                    winner = "Cortex"
        reason = ""
        if "Reason:" in text:
            reason = text.split("Reason:", 1)[1].strip().split("\n")[0].strip()
        return {"winner": winner, "reason": reason or text}
    except Exception as e:
        return {"winner": "Error", "reason": str(e)}


def run_eval(
    base_url: str,
    question: str,
    use_single: bool = False,
    brief: bool = False,
) -> dict:
    """Call API and evaluate response(s)."""
    if use_single:
        data = call_ask(base_url, question, brief=brief)
        answer_gemini = data.get("answer", "")
        evals = {
            "question": question,
            "endpoint": "/ask",
            "answer_gemini": answer_gemini,
            "answer_cortex": None,
            "eval_gemini": eval_answer(answer_gemini, "gemini"),
            "eval_cortex": None,
        }
    else:
        data = call_ask_both(base_url, question, brief=brief)
        answer_gemini = data.get("answer_gemini", "")
        answer_cortex = data.get("answer_cortex", "")
        evals = {
            "question": question,
            "endpoint": "/ask-both",
            "answer_gemini": answer_gemini,
            "answer_cortex": answer_cortex,
            "eval_gemini": eval_answer(answer_gemini, "gemini"),
            "eval_cortex": eval_answer(answer_cortex, "cortex"),
        }
    return evals


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate MedAssist.AI answers via the running API (no code changes).",
    )
    p.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Single question to evaluate",
    )
    p.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Path to file with one question per line (skip empty lines)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write full results JSON here",
    )
    p.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL (default: {DEFAULT_BASE_URL})",
    )
    p.add_argument(
        "--single",
        action="store_true",
        help="Use POST /ask instead of POST /ask-both",
    )
    p.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judge (which answer is better); only run section/length checks",
    )
    p.add_argument(
        "--brief",
        action="store_true",
        help="Use brief mode (bullets + references only) when calling the API",
    )
    args = p.parse_args()

    if args.questions:
        questions = [
            q.strip() for q in args.questions.read_text().splitlines() if q.strip()
        ]
    elif args.question:
        questions = [args.question]
    else:
        p.print_help()
        print("\nExample: python scripts/eval_answers.py 'What is the differential for fever and rash?'")
        sys.exit(1)

    all_results = []
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Evaluating: {q[:60]}...")
        try:
            result = run_eval(args.base_url, q, use_single=args.single, brief=args.brief)
            all_results.append(result)
            eg = result["eval_gemini"]
            print(f"  Gemini: ok={eg['ok']} sections={eg['sections']} words={eg.get('word_count', 0)}")
            if result.get("eval_cortex"):
                ec = result["eval_cortex"]
                print(f"  Cortex: ok={ec['ok']} sections={ec['sections']} words={ec.get('word_count', 0)}")
            # Judge: which answer is better?
            if not args.single and result.get("answer_gemini") and result.get("answer_cortex") and not args.no_judge:
                if HAS_GENAI:
                    print("  Judging which answer is better...")
                    judge = judge_answers(q, result["answer_gemini"], result["answer_cortex"])
                    if judge:
                        result["judge"] = judge
                        print(f"  >>> Winner: {judge['winner']}")
                        print(f"  >>> Reason: {judge['reason']}")
                else:
                    print("  (Install google-genai and set up ADC for judge; use --no-judge to skip)")
        except requests.RequestException as e:
            print(f"  Error: {e}")
            all_results.append({"question": q, "error": str(e)})

    if args.out:
        args.out.write_text(json.dumps(all_results, indent=2))
        print(f"\nWrote full results to {args.out}")

    # Summary
    ok_g = sum(1 for r in all_results if r.get("eval_gemini", {}).get("ok"))
    ok_c = sum(1 for r in all_results if r.get("eval_cortex", {}).get("ok"))
    print(f"\nSummary: Gemini {ok_g}/{len(all_results)} complete, Cortex {ok_c}/{len(all_results)} complete")


if __name__ == "__main__":
    main()
