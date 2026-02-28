#!/usr/bin/env python3
"""
Quick test script for MedAssist.AI API.
Usage:
  python scripts/test_api_questions.py
  python scripts/test_api_questions.py --brief
  python scripts/test_api_questions.py --question "Your question here"
"""
import argparse
import requests

BASE = "http://127.0.0.1:8000"

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
    parser.add_argument("--question", "-q", help="Single question to ask (skips list)")
    parser.add_argument("--brief", action="store_true", help="Use brief answer format")
    parser.add_argument("--both", action="store_true", help="Use /ask-both (Gemini + Cortex)")
    args = parser.parse_args()

    if args.question:
        questions = [args.question]
    else:
        questions = QUESTIONS

    for i, q in enumerate(questions, start=1):
        print(f"\n{'='*60}\nQuestion {i}: {q}\n{'='*60}")
        payload = {"question": q, "brief": args.brief}
        endpoint = "/ask-both" if args.both else "/ask"
        try:
            r = requests.post(BASE + endpoint, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            if "answer_gemini" in data:
                print("\n--- Gemini ---\n", data["answer_gemini"][:1500], "\n...")
                print("\n--- Cortex ---\n", (data.get("answer_cortex") or "")[:1500], "\n...")
            else:
                print("\n", data.get("answer", data)[:2000], "\n...")
        except requests.exceptions.RequestException as e:
            print("Error:", e)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
