"""OpenStax textbook fetcher. Downloads PDFs and extracts text (CC BY 4.0)."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .base import BaseFetcher


# Medical/science books from OpenStax - direct PDF URLs
OPENSTAX_BOOKS = {
    "anatomy-physiology-2e": {
        "title": "Anatomy and Physiology 2e",
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/Anatomy_and_Physiology_2e_-_WEB_c9nD9QL.pdf",
        "category": "general_medicine",
    },
    "biology-2e": {
        "title": "Biology 2e",
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/Biology2e-WEB.pdf",
        "category": "general_science",
    },
    "microbiology": {
        "title": "Microbiology",
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/Microbiology-WEB.pdf",
        "category": "infectious_disease",
    },
    "pharmacology": {
        "title": "Pharmacology for Nurses",
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/Pharmacology-WEB.pdf",
        "category": "pharmacology",
    },
    "medical-surgical-nursing": {
        "title": "Medical-Surgical Nursing",
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/Medical-Surgical_Nursing-WEB.pdf",
        "category": "clinical_guidelines",
    },
    "fundamentals-nursing": {
        "title": "Fundamentals of Nursing",
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/Fundamentals_of_Nursing_-_WEB.pdf",
        "category": "clinical_practice",
    },
    "clinical-nursing-skills": {
        "title": "Clinical Nursing Skills",
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/Clinical-Nursing-Skills-WEB.pdf",
        "category": "clinical_practice",
    },
}


class OpenStaxFetcher(BaseFetcher):
    """Fetches OpenStax textbooks (PDF), extracts text, stores raw + extracted."""

    def __init__(self):
        super().__init__("openstax")
        cfg = self.config.get("sources", {}).get("openstax", {})
        self.books = dict(OPENSTAX_BOOKS)
        for slug, info in cfg.get("books", {}).items():
            if isinstance(info, str):
                self.books[slug] = {"title": slug.replace("-", " ").title(), "url": info, "category": "general"}
            elif isinstance(info, dict) and "url" in info:
                self.books[slug] = {**self.books.get(slug, {}), **info}

    def fetch(
        self,
        book: str = "pharmacology",
        extract_text: bool = True,
    ) -> Path:
        """
        Fetch an OpenStax textbook: download PDF and extract text.

        Args:
            book: Book slug (e.g., pharmacology, anatomy-physiology-2e)
            extract_text: Whether to extract and store plain text from PDF
        """
        if book not in self.books:
            raise ValueError(f"Unknown book '{book}'. Available: {list(self.books.keys())}")

        info = self.books[book]
        url = info.get("url", info) if isinstance(info, dict) else info
        if isinstance(url, dict):
            url = url.get("url", "")
        title = info.get("title", book) if isinstance(info, dict) else book

        fetch_id = self.generate_fetch_id()
        raw_dir = self.writer.base_path / self.source / "raw"
        extracted_dir = self.writer.base_path / self.source / "extracted"
        raw_dir.mkdir(parents=True, exist_ok=True)
        extracted_dir.mkdir(parents=True, exist_ok=True)

        # Download PDF
        raw_path = raw_dir / f"{book}.pdf"
        resp = self._get_stream(url)
        resp.raise_for_status()
        with open(raw_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

        # Extract text
        chapters = []
        char_count = 0
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(raw_path)
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    chapters.append({"page": i + 1, "content": text})
                    char_count += len(text)
            doc.close()
        except ImportError:
            chapters = [{"page": 0, "content": "(PyMuPDF not installed - run: pip install PyMuPDF)"}]
        except Exception as e:
            chapters = [{"page": 0, "content": f"(Extraction failed: {e})"}]

        # Build payload
        payload = {
            "metadata": {
                "title": title,
                "book_slug": book,
                "source_url": url,
                "license": "CC BY 4.0",
                "raw_pdf_path": str(raw_path.relative_to(self.writer.base_path.parent.parent)),
                "page_count": len(chapters),
                "char_count": char_count,
            },
            "chapters": chapters,
        }

        extracted_path = extracted_dir / f"{book}_{fetch_id}.json"
        full_payload = {
            "_header": {
                "source": self.source,
                "fetch_id": fetch_id,
                "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "schema_version": "1.0",
            },
            "data": payload,
        }
        with open(extracted_path, "w") as f:
            json.dump(full_payload, f, indent=2, default=str)

        # Create manifest
        from ..storage.metadata import create_manifest, compute_sha256, save_manifest
        checksum = compute_sha256(extracted_path)
        manifest = create_manifest(
            source=self.source,
            fetch_id=fetch_id,
            api_endpoint=url,
            query_params={"book": book, "extract_text": extract_text},
            record_count=len(chapters),
            total_available=len(chapters),
            file_path=str(extracted_path.relative_to(self.writer.base_path.parent.parent)),
            status="success",
            error=None,
            checksum_sha256=checksum,
        )
        save_manifest(manifest, self.writer.metadata_path)

        return extracted_path
