"""Orphanet rare disease data fetcher. Downloads XML and converts to JSON."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator

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

    # Legacy dataset name -> product_id for backward compatibility
    _DATASET_TO_PRODUCT = {
        "phenotypes": "product4",
        "diseases": "product6",
        "genes": "product1",
    }

    def __init__(self):
        super().__init__("orphanet")
        self.config = self.config.get("sources", {}).get("orphanet", {})

    def get_product_registry(self) -> dict[str, dict]:
        """Return product_id -> { name, url_pattern } for all configured products."""
        products = self.config.get("products") or {}
        # Fallback if no products in config: use legacy URLs
        if not products:
            base = self.config.get("base_url", "https://www.orphadata.com/data/xml")
            products = {
                "product1": {"name": "Genes associated with rare diseases", "url_pattern": f"{base}/en_product1.xml"},
                "product4": {"name": "Rare diseases with associated phenotypes", "url_pattern": f"{base}/en_product4.xml"},
                "product6": {"name": "Rare diseases and genes", "url_pattern": f"{base}/en_product6.xml"},
            }
        return products

    def fetch_product(
        self,
        product_id: str,
        language: str = "en",
    ) -> Path | None:
        """
        Fetch a single Orphadata product by product_id (e.g. product1, product4).
        Returns path to written JSON, or None if product_id not in registry or fetch failed.
        """
        registry = self.get_product_registry()
        if product_id not in registry:
            return None
        info = registry[product_id]
        url = info.get("url_pattern", "").strip()
        if not url or not url.startswith("http"):
            return None
        if language != "en" and "en_product" in url:
            url = url.replace("en_product", f"{language}_product")

        fetch_id = self.generate_fetch_id()

        try:
            resp = self._get_stream(url)
            resp.raise_for_status()
            content = resp.content
        except Exception as e:
            self.writer.write_raw_failure(
                source=self.source,
                fetch_id=fetch_id,
                api_endpoint=url,
                query_params={"product_id": product_id, "language": language},
                error=str(e),
            )
            return None

        try:
            parsed = parse_orphanet_xml(content)
        except ET.ParseError as e:
            self.writer.write_raw_failure(
                source=self.source,
                fetch_id=fetch_id,
                api_endpoint=url,
                query_params={"product_id": product_id, "language": language},
                error=str(e),
            )
            return None

        count = 0
        for v in parsed.values():
            if isinstance(v, list):
                count += len(v)
            elif isinstance(v, dict) and v:
                count += 1

        path = self.writer.write_raw(
            source=self.source,
            fetch_id=fetch_id,
            data=parsed,
            api_endpoint=url,
            query_params={"product_id": product_id, "language": language},
            record_count=count if count > 0 else 1,
            total_available=count if count > 0 else None,
            subdir=product_id,
            include_header=True,
        )
        return path

    def fetch_all_products(self, language: str = "en") -> Iterator[tuple[str, Path | None]]:
        """Fetch every product in the registry. Yields (product_id, path or None) for each."""
        for product_id in self.get_product_registry():
            path = self.fetch_product(product_id, language=language)
            yield product_id, path

    def fetch(
        self,
        dataset: str = "phenotypes",
        language: str = "en",
        use_official: bool = True,
    ) -> Path:
        """
        Fetch Orphanet data. Supports: phenotypes (product4), diseases (product6), genes (product1),
        or any product_id from the registry (e.g. product2, product5).
        """
        product_id = self._DATASET_TO_PRODUCT.get(dataset, dataset)
        if not product_id.startswith("product"):
            product_id = "product4"
        path = self.fetch_product(product_id, language=language)
        if path is not None:
            return path
        # Legacy fallback for the three classic datasets if product fetch failed (e.g. 404)
        url_map = {
            "phenotypes": self.config.get("phenotypes_url", "https://www.orphadata.com/data/xml/en_product4.xml"),
            "diseases": self.config.get("diseases_url", "https://www.orphadata.com/data/xml/en_product6.xml"),
            "genes": self.config.get("genes_url", "https://www.orphadata.com/data/xml/en_product1.xml"),
        }
        if dataset not in url_map:
            raise FileNotFoundError(f"Orphadata product {product_id} failed and dataset {dataset} has no legacy URL")
        url = url_map[dataset]
        if "en_product" in url and language != "en":
            url = url.replace("en_product", f"{language}_product")
        fetch_id = self.generate_fetch_id()
        resp = self._get_stream(url)
        resp.raise_for_status()
        parsed = parse_orphanet_xml(resp.content)
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
