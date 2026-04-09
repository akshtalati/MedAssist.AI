"""Grounding validator for markdown references."""

from __future__ import annotations

import re


LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def extract_urls_from_context(context: str) -> set[str]:
    return set(m.group(2).strip() for m in LINK_RE.finditer(context or ""))


def validate_grounding(answer_md: str, allowed_urls: set[str]) -> tuple[str, int]:
    """Drop markdown links not present in allowed set."""
    violations = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal violations
        title, url = match.group(1), match.group(2).strip()
        if url in allowed_urls:
            return match.group(0)
        violations += 1
        return title

    return LINK_RE.sub(_replace, answer_md or ""), violations
