#!/usr/bin/env python3
"""
Run full MedAssist pipeline with checkpoint. Resume from last step if interrupted.
Checkpoint file: data/.pipeline_checkpoint.json
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_FILE = PROJECT_ROOT / "data" / ".pipeline_checkpoint.json"


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text())
        except Exception:
            pass
    return {"fetch": False, "symptom_index": False}


def save_checkpoint(state):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(state, indent=2))


def run(cmd: list[str], step: str) -> bool:
    print(f"\n>>> {step}")
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        print(f"FAILED: {step}")
        return False
    return True


def main():
    cp = load_checkpoint()
    reset = "--reset" in sys.argv

    if reset:
        cp = {"fetch": False, "symptom_index": False}
        save_checkpoint(cp)
        print("Checkpoint reset.")

    # 1. Fetch (checkpointed; run again to resume on timeout/rate limit)
    if not cp.get("fetch"):
        if run(
            ["python", "scripts/fetch_all_checkpointed.py"],
            "Fetch all sources (checkpointed)",
        ):
            run(["python", "scripts/cleanup_duplicate_raw.py"], "Cleanup duplicates after fetch")
            cp["fetch"] = True
            save_checkpoint(cp)
        else:
            print("To resume: run again (checkpoint saved).")
            sys.exit(1)
    else:
        print("(Skip fetch - already done)")

    # 2. Symptom index
    if not cp.get("symptom_index"):
        if run(
            ["python", "scripts/build_symptom_index.py"],
            "Build symptom index",
        ):
            cp["symptom_index"] = True
            save_checkpoint(cp)
        else:
            sys.exit(1)
    else:
        print("(Skip symptom index - already done)")

    print("\n*** Pipeline complete (data + symptom index; RAG skipped) ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
