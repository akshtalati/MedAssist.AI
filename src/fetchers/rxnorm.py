"""RxNorm drug nomenclature fetcher. Uses RxNav REST API."""

from pathlib import Path
from typing import Optional

from .base import BaseFetcher


class RxNormFetcher(BaseFetcher):
    """Fetches RxNorm drug names, classes, and NDC mappings."""

    def __init__(self):
        super().__init__("rxnorm")
        cfg = self.config.get("sources", {}).get("rxnorm", {})
        self.base_url = cfg.get("base_url", "https://rxnav.nlm.nih.gov/REST").rstrip("/")
        self.batch_size = cfg.get("batch_size", 100)

    def _request(self, path: str, params: Optional[dict] = None):
        """Make request to RxNav base URL."""
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        return super()._get(url, params=params)

    def fetch(
        self,
        query: str = "aspirin",
        query_type: str = "approximate",
        max_records: int = 100,
    ) -> Path:
        """
        Fetch RxNorm drug data by search.

        Args:
            query: Drug name or term to search
            query_type: approximate (getApproximateMatch) or exact (findRxcuiByString)
            max_records: Maximum drugs to return
        """
        fetch_id = self.generate_fetch_id()

        if query_type == "approximate":
            # getApproximateMatch - returns list of candidates
            path = "approximateTerm.json"
            params = {"term": query, "maxEntries": max_records}
        else:
            # findRxcuiByString
            path = f"rxcui.json"
            params = {"name": query}

        resp = self._request(path, params=params)
        resp.raise_for_status()
        data = resp.json()

        # Normalize response structure (API may return approximateTerm or approximateGroup)
        candidates = (
            data.get("approximateTerm", {}).get("candidate")
            or data.get("approximateGroup", {}).get("candidate")
        )
        if candidates is not None:
            if isinstance(candidates, dict):
                candidates = [candidates]
            drugs = candidates[:max_records]
        elif "idGroup" in data:
            rxcuis = data.get("idGroup", {}).get("rxnormId", [])
            if isinstance(rxcuis, str):
                rxcuis = [rxcuis]
            drugs = [{"rxcui": r} for r in rxcuis[:max_records]]
        else:
            drugs = []

        payload = {"query": query, "query_type": query_type, "drugs": drugs, "raw_response": data}

        return self.writer.write_raw(
            source=self.source,
            fetch_id=fetch_id,
            data=payload,
            api_endpoint=f"{self.base_url.rstrip('/')}/{path.lstrip('/')}",
            query_params={"query": query, "query_type": query_type},
            record_count=len(drugs),
            total_available=len(drugs),
            include_header=True,
        )
