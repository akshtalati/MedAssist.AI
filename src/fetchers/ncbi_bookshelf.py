"""NCBI Bookshelf fetcher. Uses E-Utilities (ESearch + ESummary) for StatPearls, guidelines, etc."""

import re
from pathlib import Path

from ..config import get_env
from .base import BaseFetcher


class NCBIBookshelfFetcher(BaseFetcher):
    """Fetches NCBI Bookshelf content (StatPearls, clinical guidelines, pharmacology)."""

    def __init__(self):
        super().__init__("ncbi_bookshelf")
        cfg = self.config.get("sources", {}).get("ncbi_bookshelf", self.config.get("sources", {}).get("pubmed", {}))
        self.base_url = cfg.get("base_url", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
        self.batch_size = cfg.get("batch_size", 50)

    def _common_params(self) -> dict:
        params = {
            "tool": "MedAssist.AI",
            "email": get_env("EMAIL", "medassist@example.com"),
            "retmode": "json",
        }
        api_key = get_env("NCBI_API_KEY")
        if api_key:
            params["api_key"] = api_key
        return params

    def _esearch(self, term: str, retmax: int = 100, retstart: int = 0) -> dict:
        url = f"{self.base_url}/esearch.fcgi"
        params = {
            **self._common_params(),
            "db": "books",
            "term": term,
            "retmax": retmax,
            "retstart": retstart,
        }
        resp = self._get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def _esummary(self, book_ids: list[str]) -> dict:
        """ESummary: get book metadata and BookTeaser (abstract) for each ID."""
        if not book_ids:
            return {}
        url = f"{self.base_url}/esummary.fcgi"
        params = {
            **self._common_params(),
            "db": "books",
            "id": ",".join(book_ids),
        }
        resp = self._get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def _parse_bookinfo_teaser(self, bookinfo: str) -> str:
        """Extract BookTeaser text from bookinfo XML."""
        if not bookinfo:
            return ""
        m = re.search(r"<BookTeaser>([^<]*(?:<[^/][^>]*>[^<]*)*)</BookTeaser>", bookinfo)
        if m:
            text = m.group(1)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"&[^;]+;", " ", text)
            return re.sub(r"\s+", " ", text).strip()
        return ""

    def fetch(
        self,
        term: str = "pharmacology",
        max_records: int = 100,
    ) -> Path:
        """
        Fetch NCBI Bookshelf content matching search term.
        Uses ESummary to get title, abstract (BookTeaser), and NBK IDs.

        Args:
            term: Search query (e.g., pharmacology, rare disease, StatPearls)
            max_records: Maximum book entries to fetch
        """
        fetch_id = self.generate_fetch_id()

        search_res = self._esearch(term, retmax=min(max_records, 500))
        if "error" in search_res:
            self.writer.write_raw_failure(
                source=self.source,
                fetch_id=fetch_id,
                api_endpoint=f"{self.base_url}/esearch.fcgi",
                query_params={"term": term, "max_records": max_records},
                error=search_res.get("error", "ESearch failed"),
            )
            raise RuntimeError(search_res.get("error", "ESearch failed"))

        esearch_res = search_res.get("esearchresult", {})
        total = int(esearch_res.get("count", 0))
        id_list = esearch_res.get("idlist", [])[:max_records]

        if not id_list:
            return self.writer.write_raw(
                source=self.source,
                fetch_id=fetch_id,
                data={"books": [], "total_available": total},
                api_endpoint=f"{self.base_url}/esearch.fcgi",
                query_params={"term": term, "max_records": max_records},
                record_count=0,
                total_available=total,
                subdir="sections",
                include_header=True,
            )

        books = []
        for i in range(0, len(id_list), self.batch_size):
            batch_ids = id_list[i : i + self.batch_size]
            try:
                summary_res = self._esummary(batch_ids)
                result = summary_res.get("result", {})
                for uid in result.get("uids", []):
                    item = result.get(uid, {})
                    if isinstance(item, dict):
                        teaser = self._parse_bookinfo_teaser(item.get("bookinfo", ""))
                        books.append({
                            "uid": uid,
                            "nbk_id": item.get("bookaccessionid", item.get("accessionid", "")),
                            "title": item.get("title", ""),
                            "pubdate": item.get("pubdate", ""),
                            "abstract": teaser or item.get("text", ""),
                            "url": f"https://www.ncbi.nlm.nih.gov/books/{item.get('bookaccessionid', uid)}/" if item.get("bookaccessionid") else "",
                        })
            except Exception as e:
                for bid in batch_ids:
                    books.append({"uid": bid, "error": str(e)})

        payload = {
            "query_term": term,
            "total_available": total,
            "books": books,
        }

        return self.writer.write_raw(
            source=self.source,
            fetch_id=fetch_id,
            data=payload,
            api_endpoint=f"{self.base_url}/esummary.fcgi",
            query_params={"term": term, "max_records": max_records},
            record_count=len(books),
            total_available=total,
            subdir="sections",
            include_header=True,
        )
