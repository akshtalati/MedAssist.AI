"""PubMed article fetcher. Uses NCBI E-Utilities (ESearch + EFetch)."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from ..config import get_env
from .base import BaseFetcher


class PubMedFetcher(BaseFetcher):
    """Fetches PubMed article metadata and abstracts via E-Utilities."""

    def __init__(self):
        super().__init__("pubmed")
        cfg = self.config.get("sources", {}).get("pubmed", {})
        self.base_url = cfg.get("base_url", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
        self.batch_size = cfg.get("batch_size", 500)

    def _common_params(self) -> dict:
        """Common params for E-Utilities (tool, email, optional api_key)."""
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
        """ESearch: get PMIDs matching query."""
        url = f"{self.base_url}/esearch.fcgi"
        params = {
            **self._common_params(),
            "db": "pubmed",
            "term": term,
            "retmax": retmax,
            "retstart": retstart,
        }
        resp = self._get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def _efetch(self, pmids: list[int], rettype: str = "abstract") -> dict:
        """EFetch: get full records for PMIDs."""
        if not pmids:
            return {"records": []}
        url = f"{self.base_url}/efetch.fcgi"
        params = {
            **self._common_params(),
            "db": "pubmed",
            "id": ",".join(str(p) for p in pmids),
            "rettype": rettype,
            "retmode": "xml",
        }
        resp = self._get(url, params=params)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        ns = {"pm": "http://www.ncbi.nlm.nih.gov/eutils"}
        articles = []
        for article in root.findall(".//pm:PubmedArticle", ns) or root.findall(".//PubmedArticle"):
            articles.append(self._parse_article(article))
        return {"records": articles}

    def _parse_article(self, article: ET.Element) -> dict:
        """Parse PubmedArticle XML to dict."""
        ns = {"pm": "http://www.ncbi.nlm.nih.gov/eutils"}
        result = {"pmid": None, "title": "", "abstract": "", "authors": [], "journal": "", "pub_date": ""}

        med = article.find(".//pm:MedlineCitation", ns) or article.find(".//MedlineCitation")
        if med is None:
            return result

        pmid_elem = med.find(".//pm:PMID", ns) or med.find(".//PMID")
        if pmid_elem is not None and pmid_elem.text:
            result["pmid"] = int(pmid_elem.text)

        article_elem = med.find(".//pm:Article", ns) or med.find(".//Article")
        if article_elem is not None:
            title_e = article_elem.find(".//pm:ArticleTitle", ns) or article_elem.find(".//ArticleTitle")
            if title_e is not None and title_e.text:
                result["title"] = "".join(title_e.itertext()).strip()

            abst_e = article_elem.find(".//pm:Abstract", ns) or article_elem.find(".//Abstract")
            if abst_e is not None:
                texts = []
                for pt in abst_e.findall(".//pm:AbstractText", ns) or abst_e.findall(".//AbstractText"):
                    if pt.text:
                        texts.append(pt.text)
                    texts.extend(pt.itertext())
                result["abstract"] = " ".join(t for t in texts if t and isinstance(t, str)).strip()

            auth_list = article_elem.find(".//pm:AuthorList", ns) or article_elem.find(".//AuthorList")
            if auth_list is not None:
                for auth in auth_list.findall(".//pm:Author", ns) or auth_list.findall(".//Author"):
                    last = auth.find(".//pm:LastName", ns) or auth.find(".//LastName")
                    fore = auth.find(".//pm:ForeName", ns) or auth.find(".//ForeName")
                    if last is not None and last.text:
                        name = last.text
                        if fore is not None and fore.text:
                            name = f"{fore.text} {name}"
                        result["authors"].append(name)

            journal_e = article_elem.find(".//pm:Journal", ns) or article_elem.find(".//Journal")
            if journal_e is not None:
                title_j = journal_e.find(".//pm:Title", ns) or journal_e.find(".//Title")
                if title_j is not None and title_j.text:
                    result["journal"] = title_j.text

        pub_date = med.find(".//pm:PubDate", ns) or med.find(".//PubDate")
        if pub_date is not None and pub_date.text:
            result["pub_date"] = pub_date.text

        return result

    def fetch(
        self,
        term: str = "rare disease",
        max_records: int = 500,
    ) -> Path:
        """
        Fetch PubMed articles matching search term.

        Args:
            term: PubMed search query
            max_records: Maximum number of articles to fetch
        """
        fetch_id = self.generate_fetch_id()

        # ESearch to get count and first batch of PMIDs
        search_res = self._esearch(term, retmax=min(max_records, 10000))
        if "error" in search_res:
            self.writer.write_raw_failure(
                source=self.source,
                fetch_id=fetch_id,
                api_endpoint=f"{self.base_url}/esearch.fcgi",
                query_params={"term": term, "retmax": max_records},
                error=search_res.get("error", "Unknown error"),
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
                query_params={"term": term, "retmax": max_records},
                record_count=0,
                total_available=total,
                include_header=True,
            )

        # EFetch in batches
        articles = []
        for i in range(0, len(id_list), self.batch_size):
            batch_ids = [int(pid) for pid in id_list[i : i + self.batch_size]]
            fetch_res = self._efetch(batch_ids)
            articles.extend(fetch_res.get("records", []))

        return self.writer.write_raw(
            source=self.source,
            fetch_id=fetch_id,
            data={"articles": articles, "total_available": total},
            api_endpoint=f"{self.base_url}/efetch.fcgi",
            query_params={"term": term, "retmax": max_records},
            record_count=len(articles),
            total_available=total,
            include_header=True,
        )
