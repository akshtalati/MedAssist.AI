"""Base fetcher with retry, backoff, and rate limiting."""

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import get_data_paths, load_config
from ..storage import DataWriter


def _gen_fetch_id(source: str) -> str:
    """Generate unique fetch ID: source_YYYYMMDD_HHMMSS."""
    now = datetime.now(timezone.utc)
    return f"{source}_{now.strftime('%Y%m%d_%H%M%S')}"


class BaseFetcher(ABC):
    """Base class for all data fetchers. Handles retry, rate limit, and storage."""

    def __init__(self, source: str, config_key: Optional[str] = None):
        self.source = source
        self.config_key = config_key or source
        self.config = load_config()
        paths = get_data_paths()
        self.writer = DataWriter(paths["raw"], paths["metadata"])
        self._last_request_time: float = 0.0
        self._rate_limit_delay = self._get_rate_limit()

    def _get_rate_limit(self) -> float:
        """Get rate limit delay in seconds from config."""
        sources = self.config.get("sources", {})
        src_config = sources.get(self.config_key, {})
        per_sec = src_config.get("rate_limit_per_sec", 3)
        return 1.0 / per_sec if per_sec else 0.34

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.monotonic()

    @retry(
        retry=retry_if_exception_type((requests.RequestException, ConnectionError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
    )
    def _get(self, url: str, params: Optional[dict] = None, **kwargs) -> requests.Response:
        """GET request with retry and rate limiting."""
        self._rate_limit()
        return requests.get(url, params=params, timeout=120, **kwargs)

    @retry(
        retry=retry_if_exception_type((requests.RequestException, ConnectionError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
    )
    def _get_stream(self, url: str, **kwargs) -> requests.Response:
        """GET request with streaming (for large downloads)."""
        self._rate_limit()
        return requests.get(url, stream=True, timeout=120, **kwargs)

    def generate_fetch_id(self) -> str:
        """Generate a unique fetch ID for this run."""
        return _gen_fetch_id(self.source)

    @abstractmethod
    def fetch(self, **kwargs) -> Path:
        """
        Fetch data and store locally.

        Returns:
            Path to the written data file.
        """
        pass
