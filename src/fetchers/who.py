"""WHO guidelines and documents fetcher. Uses WHO REST API."""

from pathlib import Path
from typing import Optional

from .base import BaseFetcher


class WHOFetcher(BaseFetcher):
    """Fetches WHO treatment guidelines and document metadata."""

    def __init__(self):
        super().__init__("who")
        cfg = self.config.get("sources", {}).get("who", {})
        self.base_url = (cfg.get("base_url", "https://www.who.int/api") or "https://www.who.int/api").rstrip("/")
        self.endpoints = cfg.get("endpoints", {
            "documents": "/documents",
            "guidelines": "/news/daguidelines",
        })

    def fetch(
        self,
        endpoint: str = "documents",
        limit: int = 100,
        language: str = "en",
    ) -> Path:
        """
        Fetch WHO documents or guidelines.

        Args:
            endpoint: One of documents, guidelines
            limit: Max items to fetch
            language: Language code
        """
        path = self.endpoints.get(endpoint, self.endpoints["documents"])
        url = f"{self.base_url}{path}"
        fetch_id = self.generate_fetch_id()

        params = {"language": language}
        if limit:
            params["limit"] = limit

        try:
            resp = self._get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            # WHO API may have different structure; try generic content
            try:
                alt_url = "https://www.who.int/api/hubs/publications"
                fallback_params = {}
                if limit:
                    fallback_params["limit"] = limit
                resp = self._get(alt_url, params=fallback_params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e2:
                self.writer.write_raw_failure(
                    source=self.source,
                    fetch_id=fetch_id,
                    api_endpoint=url,
                    query_params={"endpoint": endpoint, "limit": limit},
                    error=f"{str(e)}; fallback: {str(e2)}",
                )
                raise

        # Extract records from various WHO response shapes
        records = []
        if isinstance(data, list):
            records = data[:limit]
        elif isinstance(data, dict):
            records = data.get("value", data.get("data", data.get("results", [])))
            if isinstance(records, dict):
                records = [records]
            records = records[:limit] if isinstance(records, list) else []

        payload = {
            "raw_response": data,
            "records": records,
            "endpoint": endpoint,
        }

        return self.writer.write_raw(
            source=self.source,
            fetch_id=fetch_id,
            data=payload,
            api_endpoint=url,
            query_params={"endpoint": endpoint, "limit": limit, "language": language},
            record_count=len(records) if isinstance(records, list) else 1,
            total_available=None,
            subdir=endpoint,
            include_header=True,
        )
