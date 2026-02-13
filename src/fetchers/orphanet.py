"""Orphanet rare disease data fetcher. Downloads XML and converts to JSON."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .base import BaseFetcher


def _xml_to_dict(element: ET.Element) -> Any:
    """Recursively convert XML element to dict/list structure."""
    if len(element) == 0 and (element.text is None or not element.text.strip()):
        return element.text or ""
    if len(element) == 0:
        return element.text.strip() if element.text else ""
    result: dict[str, Any] = {}
    for child in element:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        val = _xml_to_dict(child)
        if tag in result:
            if not isinstance(result[tag], list):
                result[tag] = [result[tag]]
            result[tag].append(val)
        else:
            result[tag] = val
    if element.text and element.text.strip():
        result["_text"] = element.text.strip()
    return result


def parse_orphanet_xml(xml_content: bytes) -> dict:
    """Parse Orphanet product XML into a structured dict."""
    root = ET.fromstring(xml_content)
    # Handle default namespace
    ns = {}
    if root.tag.startswith("{"):
        ns["orphadata"] = root.tag.split("}")[0].strip("{")
    result: dict[str, Any] = {"_root_tag": root.tag.split("}")[-1]}
    for child in root:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        val = _xml_to_dict(child)
        if tag in result:
            if not isinstance(result[tag], list):
                result[tag] = [result[tag]]
            result[tag].append(val)
        else:
            result[tag] = val
    return result


class OrphanetFetcher(BaseFetcher):
    """Fetches Orphanet rare disease data from Orphadata XML files."""

    def __init__(self):
        super().__init__("orphanet")
        self.config = self.config.get("sources", {}).get("orphanet", {})

    def fetch(
        self,
        dataset: str = "phenotypes",
        language: str = "en",
        use_official: bool = True,
    ) -> Path:
        """
        Fetch Orphanet data. Supports: phenotypes, diseases, genes.

        Args:
            dataset: One of phenotypes (product4), diseases (product6), genes (product1)
            language: Language code (en, fr, de, es, it, nl, pt)
            use_official: Use official orphadata.com (default) vs GitHub
        """
        url_map = {
            "phenotypes": self.config.get("phenotypes_url", "https://www.orphadata.com/data/xml/en_product4.xml"),
            "diseases": self.config.get("diseases_url", "https://www.orphadata.com/data/xml/en_product6.xml"),
            "genes": self.config.get("genes_url", "https://www.orphadata.com/data/xml/en_product1.xml"),
        }
        url = url_map.get(dataset, url_map["phenotypes"])
        # Replace language in URL if pattern exists
        if "en_product" in url and language != "en":
            url = url.replace("en_product", f"{language}_product")
        elif "en_" not in url and dataset == "phenotypes":
            url = f"https://www.orphadata.com/data/xml/{language}_product4.xml"

        fetch_id = self.generate_fetch_id()

        resp = self._get_stream(url)
        resp.raise_for_status()
        content = resp.content

        try:
            parsed = parse_orphanet_xml(content)
        except ET.ParseError as e:
            self.writer.write_raw_failure(
                source=self.source,
                fetch_id=fetch_id,
                api_endpoint=url,
                query_params={"dataset": dataset, "language": language},
                error=str(e),
            )
            raise

        # Count records heuristically (Orphanet XML structure varies)
        count = 0
        for v in parsed.values():
            if isinstance(v, list):
                count += len(v)
            elif isinstance(v, dict) and v:
                count += 1

        return self.writer.write_raw(
            source=self.source,
            fetch_id=fetch_id,
            data=parsed,
            api_endpoint=url,
            query_params={"dataset": dataset, "language": language},
            record_count=count if count > 0 else 1,
            total_available=count if count > 0 else None,
            subdir=dataset,
            include_header=True,
        )
