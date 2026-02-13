"""PubMed Central fetcher. Uses E-Utilities with db=pmc for open-access subset."""

import xml.etree.ElementTree as ET
from pathlib import Path

from ..config import get_env
from .base import BaseFetcher


class PMCFetcher(BaseFetcher):
    """Fetches PubMed Central article metadata and abstracts."""

    def __init__(self):
        super().__init__("pmc")
        cfg = self.config.get("sources", {}).get("pmc", {})
        self.base_url = cfg.get("base_url", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
        self.db = cfg.get("db", "pmc")
        self.batch_size = cfg.get("batch_size", 500)

    def _common_params(self) -> dict:
        """Common params for E-Utilities."""
        params = {
            "tool": "MedAssist.AI",
            "email": get_env("EMAIL", "medassist@example.com"),
            "retmode": "json",
        }
        api_key = get_env("NCBI_API_KEY")
        if api_key:
            params["api_key"] = api_key
        return params

    def _esearch(self, term: str, retmax: int = 500, retstart: int = 0) -> dict:
        """ESearch on PMC database."""
        url = f"{self.base_url}/esearch.fcgi"
        # Optional: add "open access [filter]" for open-access subset only
        search_term = term or "rare disease"
        params = {
            **self._common_params(),
            "db": self.db,
            "term": search_term,
            "retmax": retmax,
            "retstart": retstart,
        }
        resp = self._get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def _efetch(self, pmc_ids: list[str]) -> dict:
        """EFetch PMC articles as XML."""
        if not pmc_ids:
            return {"records": []}
        url = f"{self.base_url}/efetch.fcgi"
        params = {
            **self._common_params(),
            "db": self.db,
            "id": ",".join(pmc_ids),
            "retmode": "xml",
        }
        # Override retmode for efetch
        params["retmode"] = "xml"
        resp = self._get(url, params=params)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        articles = []
        ns = {"pmc": "http://www.ncbi.nlm.nih.gov/eutils"}
        for art in root.findall(".//pmc:article", ns) or root.findall(".//article"):
            articles.append(self._parse_pmc_article(art))
        return {"records": articles}

    def _parse_pmc_article(self, article: ET.Element) -> dict:
        """Parse PMC article XML to dict."""
        ns = {"pmc": "http://www.ncbi.nlm.nih.gov/eutils"}
        result = {"pmcid": None, "title": "", "abstract": "", "authors": [], "journal": "", "pub_date": ""}

        front = article.find(".//pmc:front", ns) or article.find(".//front")
        if front is None:
            return result

        article_meta = front.find(".//pmc:article-meta", ns) or front.find(".//article-meta")
        if article_meta is None:
            return result

        pmcid_e = article_meta.find(".//pmc:article-id[@pub-id-type='pmc']", ns)
        if pmcid_e is None:
            pmcid_e = article_meta.find(".//article-id[@pub-id-type='pmc']")
        if pmcid_e is not None and pmcid_e.text:
            result["pmcid"] = pmcid_e.text

        title_group = article_meta.find(".//pmc:title-group", ns) or article_meta.find(".//title-group")
        if title_group is not None:
            title_e = title_group.find(".//pmc:article-title", ns) or title_group.find(".//article-title")
            if title_e is not None:
                result["title"] = "".join(title_e.itertext()).strip()

        abst = article_meta.find(".//pmc:abstract", ns) or article_meta.find(".//abstract")
        if abst is not None:
            result["abstract"] = " ".join(t for t in abst.itertext() if t and isinstance(t, str)).strip()

        contrib_group = article_meta.find(".//pmc:contrib-group", ns) or article_meta.find(".//contrib-group")
        if contrib_group is not None:
            for contrib in contrib_group.findall(".//pmc:contrib", ns) or contrib_group.findall(".//contrib"):
                name_e = contrib.find(".//pmc:name", ns) or contrib.find(".//name")
                if name_e is not None:
                    given = name_e.find(".//pmc:given-names", ns) or name_e.find(".//given-names")
                    surname = name_e.find(".//pmc:surname", ns) or name_e.find(".//surname")
                    parts = []
                    if given is not None and given.text:
                        parts.append(given.text)
                    if surname is not None and surname.text:
                        parts.append(surname.text)
                    if parts:
                        result["authors"].append(" ".join(parts))

        pub_date = article_meta.find(".//pmc:pub-date", ns) or article_meta.find(".//pub-date")
        if pub_date is not None:
            year = pub_date.find(".//pmc:year", ns) or pub_date.find(".//year")
            month = pub_date.find(".//pmc:month", ns) or pub_date.find(".//month")
            day = pub_date.find(".//pmc:day", ns) or pub_date.find(".//day")
            parts = []
            if year is not None and year.text:
                parts.append(year.text)
            if month is not None and month.text:
                parts.append(month.text)
            if day is not None and day.text:
                parts.append(day.text)
            result["pub_date"] = "-".join(parts) if parts else ""

        return result

    def fetch(
        self,
        term: str = "rare disease",
        max_records: int = 500,
    ) -> Path:
        """
        Fetch PMC articles (open-access subset) matching search term.

        Args:
            term: Search query
            max_records: Maximum articles to fetch
        """
        fetch_id = self.generate_fetch_id()

        search_res = self._esearch(term, retmax=min(max_records, 10000))
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
        id_list = esearch_res.get("idlist", [])

        if not id_list:
            return self.writer.write_raw(
                source=self.source,
                fetch_id=fetch_id,
                data={"articles": [], "total_available": total},
                api_endpoint=f"{self.base_url}/esearch.fcgi",
                query_params={"term": term, "max_records": max_records},
                record_count=0,
                total_available=total,
                include_header=True,
            )

        articles = []
        for i in range(0, len(id_list), self.batch_size):
            batch_ids = id_list[i : i + self.batch_size]
            fetch_res = self._efetch(batch_ids)
            articles.extend(fetch_res.get("records", []))

        return self.writer.write_raw(
            source=self.source,
            fetch_id=fetch_id,
            data={"articles": articles, "total_available": total},
            api_endpoint=f"{self.base_url}/efetch.fcgi",
            query_params={"term": term, "max_records": max_records},
            record_count=len(articles),
            total_available=total,
            include_header=True,
        )
