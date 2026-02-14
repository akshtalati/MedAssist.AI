"""Index builders for Phase 2 - symptom→disease, RAG, etc."""

from .orphanet_index import build_symptom_index, SymptomIndex

__all__ = ["build_symptom_index", "SymptomIndex"]
