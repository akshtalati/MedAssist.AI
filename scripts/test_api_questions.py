#!/usr/bin/env python3
"""
Quick test script for MedAssist.AI API.
Usage:
  python scripts/test_api_questions.py
  python scripts/test_api_questions.py --brief
  python scripts/test_api_questions.py --question "Your question here"
  MEDASSIST_API_BASE=http://127.0.0.1:8001 python scripts/test_api_questions.py -q "..." --both
"""
import argparse
import os
import requests

DEFAULT_BASE = os.environ.get("MEDASSIST_API_BASE", "http://127.0.0.1:8000").rstrip("/")

QUESTIONS = [
    "What conditions can present with fever and vomiting?",
    "Differential diagnosis for chronic fatigue and muscle weakness",
    "Rare diseases associated with seizures in children",
    "What are red flags when a patient has headache and vision changes?",
    "Pediatric fever without focus — what should I consider?",
    "Rare genetic disorders that cause developmental delay",
    "Treatment and next steps for suspected bacterial meningitis",
    "Symptoms and diagnosis of Guillain-Barré syndrome",
]


def main():
    parser = argparse.ArgumentParser(description="Test MedAssist.AI /ask endpoint")
    parser.add_argument(
        "--base",
        "-b",
        default=DEFAULT_BASE,
        help=f"API base URL (default: env MEDASSIST_API_BASE or http://127.0.0.1:8000). Current default: {DEFAULT_BASE}",
    )
    parser.add_argument("--question", "-q", help="Single question to ask (skips list)")
    parser.add_argument("--brief", action="store_true", help="Use brief answer format")
    parser.add_argument("--both", action="store_true", help="Use /ask-both (Gemini + Cortex)")
    args = parser.parse_args()
    base = (args.base or DEFAULT_BASE).rstrip("/")
    print(f"Using API base: {base}")

    if args.question:
        questions = [args.question]
    else:
        questions = QUESTIONS

    for i, q in enumerate(questions, start=1):
        print(f"\n{'='*60}\nQuestion {i}: {q}\n{'='*60}")
        payload = {"question": q, "brief": args.brief}
        endpoint = "/ask-both" if args.both else "/ask"
        try:
            r = requests.post(base + endpoint, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            if "answer_gemini" in data:
                print("\n--- Gemini ---\n", data["answer_gemini"][:1500], "\n...")
                print("\n--- Cortex ---\n", (data.get("answer_cortex") or "")[:1500], "\n...")
            else:
                prov = data.get("provider")
                if prov:
                    print("provider:", prov, "fallback_from:", data.get("fallback_from"))
                print("\n", (data.get("answer") or str(data))[:2000], "\n...")
        except requests.exceptions.RequestException as e:
            print("Error:", e)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
