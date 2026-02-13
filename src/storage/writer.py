"""Data writer for storing raw API responses with metadata."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .metadata import create_manifest, compute_sha256, save_manifest


class DataWriter:
    """Writes raw data and manifest files to disk."""

    def __init__(self, base_path: Path, metadata_path: Path):
        self.base_path = Path(base_path)
        self.metadata_path = Path(metadata_path)

    def write_raw(
        self,
        source: str,
        fetch_id: str,
        data: Any,
        api_endpoint: str,
        query_params: dict,
        record_count: int,
        total_available: Optional[int] = None,
        subdir: Optional[str] = None,
        include_header: bool = True,
    ) -> Path:
        """
        Write raw data to a JSON file and save manifest.

        Args:
            source: Data source name (e.g., pubmed, openfda)
            fetch_id: Unique fetch identifier
            data: Data to serialize (must be JSON-serializable)
            api_endpoint: API endpoint used
            query_params: Query parameters used
            record_count: Number of records
            total_available: Total available (if known)
            subdir: Subdirectory under source (e.g., "label" for openfda/label)
            include_header: Prepend manifest-like header to the file

        Returns:
            Path to the written file
        """
        out_dir = self.base_path / source
        if subdir:
            out_dir = out_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{fetch_id}.json"
        file_path = out_dir / filename

        if include_header:
            payload = {
                "_header": {
                    "source": source,
                    "fetch_id": fetch_id,
                    "fetched_at": datetime.now(timezone.utc).isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "schema_version": "1.0",
                },
                "data": data,
            }
        else:
            payload = data

        with open(file_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

        # Compute checksum after write
        checksum = compute_sha256(file_path)

        manifest = create_manifest(
            source=source,
            fetch_id=fetch_id,
            api_endpoint=api_endpoint,
            query_params=query_params,
            record_count=record_count,
            total_available=total_available,
            file_path=str(file_path.relative_to(self.base_path.parent.parent)),
            status="success",
            error=None,
            checksum_sha256=checksum,
        )
        save_manifest(manifest, self.metadata_path)

        return file_path

    def write_raw_failure(
        self,
        source: str,
        fetch_id: str,
        api_endpoint: str,
        query_params: dict,
        error: str,
    ) -> None:
        """Record a failed fetch in manifest (no data file)."""
        manifest = create_manifest(
            source=source,
            fetch_id=fetch_id,
            api_endpoint=api_endpoint,
            query_params=query_params,
            record_count=0,
            total_available=None,
            file_path="",
            status="failure",
            error=error,
        )
        save_manifest(manifest, self.metadata_path)
