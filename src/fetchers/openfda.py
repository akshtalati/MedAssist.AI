"""OpenFDA drug data fetcher. Fetches labels, adverse events, NDC."""

from pathlib import Path
from typing import Optional

from ..config import get_env
from .base import BaseFetcher


class OpenFDAFetcher(BaseFetcher):
    """Fetches OpenFDA drug labels, adverse events, and NDC data."""

    def __init__(self):
        super().__init__("openfda")
        cfg = self.config.get("sources", {}).get("openfda", {})
        self.base_url = cfg.get("base_url", "https://api.fda.gov").rstrip("/")
        self.endpoints = cfg.get("endpoints", {
            "label": "/drug/label.json",
            "event": "/drug/event.json",
            "ndc": "/drug/ndc.json",
        })
        self.batch_size = cfg.get("batch_size", 1000)

    def _request(self, endpoint: str, limit: int = 1000, skip: int = 0, search: Optional[str] = None) -> dict:
        """Make OpenFDA API request with pagination."""
        url = f"{self.base_url}{endpoint}"
        params = {"limit": min(limit, self.batch_size), "skip": skip}
        if search:
            params["search"] = search
        api_key = get_env("OPENFDA_API_KEY")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = self._get(url, params=params, headers=headers or None)
        resp.raise_for_status()
        return resp.json()

    def fetch(
        self,
        endpoint: str = "label",
        max_records: int = 5000,
        search: Optional[str] = None,
    ) -> Path:
        """
        Fetch data from an OpenFDA endpoint.

        Args:
            endpoint: One of label, event, ndc
            max_records: Maximum records to fetch
            search: Optional search query (e.g., 'reactionmeddrapt:"headache"')
        """
        path = self.endpoints.get(endpoint, "/drug/label.json")
        fetch_id = self.generate_fetch_id()

        all_results = []
        skip = 0
        total = None

        while len(all_results) < max_records:
            data = self._request(path, limit=self.batch_size, skip=skip, search=search)
            results = data.get("results", [])
            if not results:
                break
            all_results.extend(results)
            meta = data.get("meta", {})
            total = meta.get("results", {}).get("total")
            if total is not None and len(all_results) >= total:
                break
            skip += len(results)
            if len(results) < self.batch_size:
                break

        # Preserve full API response structure
        payload = {
            "meta": {"total": total, "limit": max_records, "skip": 0},
            "results": all_results,
        }

        return self.writer.write_raw(
            source=self.source,
            fetch_id=fetch_id,
            data=payload,
            api_endpoint=f"{self.base_url}{path}",
            query_params={"endpoint": endpoint, "max_records": max_records, "search": search},
            record_count=len(all_results),
            total_available=total,
            subdir=endpoint.replace("/", "_").strip("_") or "label",
            include_header=True,
        )
