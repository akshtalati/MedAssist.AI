"""Manifest and metadata generation for data fetches."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def create_manifest(
    source: str,
    fetch_id: str,
    api_endpoint: str,
    query_params: dict,
    record_count: int,
    total_available: Optional[int],
    file_path: Path,
    status: str = "success",
    error: Optional[str] = None,
    checksum_sha256: Optional[str] = None,
    schema_version: str = "1.0",
) -> dict:
    """Create a manifest dict for a fetch operation."""
    manifest = {
        "source": source,
        "fetch_id": fetch_id,
        "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "schema_version": schema_version,
        "api_endpoint": api_endpoint,
        "query_params": query_params,
        "record_count": record_count,
        "total_available": total_available,
        "file_path": str(file_path),
        "checksum_sha256": checksum_sha256,
        "status": status,
        "error": error,
    }
    return manifest


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_manifest(manifest: dict, metadata_dir: Path) -> Path:
    """Save manifest to metadata directory as JSON."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_dir / f"{manifest['fetch_id']}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path
