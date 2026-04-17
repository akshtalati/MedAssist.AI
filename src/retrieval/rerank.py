"""Optional cross-encoder rerank for merged literature lists (feature-flagged)."""

from __future__ import annotations

import os
from typing import Any

RERANK_ENABLED = os.environ.get("RAG_RERANKER_ENABLED", "").strip().lower() in ("1", "true", "yes")
RERANK_MODEL = os.environ.get("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_cross_encoder = None


def maybe_rerank_literature(query: str, entries: list[dict[str, Any]], *, top_k: int | None = None) -> list[dict[str, Any]]:
    """
    Re-score merged RAG/ILIKE rows with a cross-encoder. No-op if disabled or import fails.
    """
    if not RERANK_ENABLED or not entries:
        return entries
    global _cross_encoder
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return entries
    try:
        if _cross_encoder is None:
            _cross_encoder = CrossEncoder(RERANK_MODEL)
    except Exception:
        return entries
    texts = []
    for e in entries:
        t = f"{e.get('title') or ''}\n{(e.get('text') or '')[:800]}"
        texts.append(t)
    pairs = [[(query or "")[:512], t] for t in texts]
    try:
        scores = _cross_encoder.predict(pairs)
    except Exception:
        return entries
    ranked = sorted(zip(entries, scores), key=lambda x: float(x[1]), reverse=True)
    out = [e for e, _ in ranked]
    if top_k is not None:
        return out[:top_k]
    return out
