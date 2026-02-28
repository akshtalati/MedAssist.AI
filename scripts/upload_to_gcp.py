#!/usr/bin/env python3
"""
Upload MedAssist data to a GCP bucket for use with Vertex AI Search or Snowflake stage.

Target bucket (Option B): gs://medassist-data-gcs/medassist/
After upload, you can:
  - Create a Vertex AI Search data store from gs://medassist-data-gcs/medassist/
  - In Snowflake: CREATE STAGE ... URL = 'gcs://...' and COPY INTO from the stage

Usage:
  python scripts/upload_to_gcp.py
  python scripts/upload_to_gcp.py --bucket gs://my-bucket/medassist
  python scripts/upload_to_gcp.py --all
  python scripts/upload_to_gcp.py --orphanet-only
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_gsutil(args: list[str]) -> bool:
    """Run gsutil; return True on success."""
    try:
        subprocess.run(
            ["gsutil", "-m", "cp", "-r"] + args,
            check=True,
            cwd=PROJECT_ROOT,
        )
        return True
    except FileNotFoundError:
        print("gsutil not found. Install Google Cloud SDK and run 'gcloud auth login'.", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:
        print(f"gsutil failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload MedAssist data to GCP bucket")
    parser.add_argument(
        "--bucket",
        default="gs://medassist-data-gcs/medassist",
        help="GCS base path (e.g. gs://medassist-data-gcs/medassist)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Upload data/raw, data/normalized, data/vectors (default: orphanet only)",
    )
    parser.add_argument(
        "--orphanet-only",
        action="store_true",
        help="Upload only data/raw/orphanet (default if no --all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without uploading",
    )
    parser.add_argument(
        "--snowflake-layout",
        action="store_true",
        dest="snowflake_layout",
        help="Upload to medassist/raw, medassist/normalized, medassist/metadata (no data/ prefix) for Snowflake stages",
    )
    args = parser.parse_args()

    bucket = args.bucket.rstrip("/")
    paths_to_upload = []

    if args.all:
        if getattr(args, "snowflake_layout", False):
            # Paths your Snowflake script expects: medassist/raw/, medassist/normalized/, medassist/metadata/
            for name in ["raw", "normalized", "metadata"]:
                d = PROJECT_ROOT / "data" / name
                if d.exists():
                    paths_to_upload.append((d, f"{bucket}/{name}"))
        else:
            for name in ["raw", "normalized", "metadata", "vectors"]:
                d = PROJECT_ROOT / "data" / name
                if d.exists():
                    paths_to_upload.append((d, f"{bucket}/data/{name}"))
    else:
        # Default: orphanet only (raw Orphadata + web crawl)
        orphanet = PROJECT_ROOT / "data" / "raw" / "orphanet"
        if orphanet.exists():
            paths_to_upload.append((orphanet, f"{bucket}/data/raw/orphanet"))

    if not paths_to_upload:
        print("No local data dirs found. Run fetch_orphanet_all.py and crawl_orphanet_web.py first.", file=sys.stderr)
        return 1

    if args.dry_run:
        for local, remote in paths_to_upload:
            print(f"Would upload: {local} -> {remote}")
        return 0

    for local, remote in paths_to_upload:
        print(f"Uploading {local} -> {remote} ...", flush=True)
        if not run_gsutil([str(local), remote + "/"]):
            return 1
        print(f"  Done.", flush=True)
    print("Upload complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
