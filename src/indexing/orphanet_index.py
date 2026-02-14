"""
Orphanet symptom→disease index builder.

Parses raw Orphanet phenotypes JSON and builds an inverted index:
  symptom (HPO term) → list of (orpha_code, disease_name, frequency)

Enables multi-symptom queries: "fever and vomiting" → intersect disease sets.
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

from ..config import get_data_paths, PROJECT_ROOT


def _normalize_symptom(term: str) -> str:
    """Normalize symptom for indexing: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", term.strip().lower()) if term else ""


def _extract_disorders(data: dict) -> list[dict]:
    """Extract disorder list from Orphanet phenotypes JSON structure."""
    try:
        inner = data.get("data", data)
        status_list = inner.get("HPODisorderSetStatusList", {})
        disorders = status_list.get("HPODisorderSetStatus", [])
        if isinstance(disorders, dict):
            disorders = [disorders]
        return disorders
    except (TypeError, AttributeError):
        return []


def _extract_disease_and_symptoms(disorder: dict) -> tuple[Optional[str], Optional[str], list[tuple[str, str, str]]]:
    """
    Extract (orpha_code, disease_name, [(hpo_id, hpo_term, frequency), ...]) from a disorder.
    """
    disorder_obj = disorder.get("Disorder", {}) if isinstance(disorder, dict) else {}
    if not disorder_obj:
        return None, None, []

    orpha_code = disorder_obj.get("OrphaCode") or disorder_obj.get("OrphaCode")
    name = disorder_obj.get("Name", "")
    if not orpha_code or not name:
        return None, None, []

    assoc_list = disorder_obj.get("HPODisorderAssociationList")
    if not isinstance(assoc_list, dict):
        return str(orpha_code), name, []
    raw_assoc = assoc_list.get("HPODisorderAssociation")
    associations = (
        raw_assoc if isinstance(raw_assoc, list)
        else ([raw_assoc] if isinstance(raw_assoc, dict) else [])
    )

    symptoms = []
    for assoc in associations:
        hpo = assoc.get("HPO", {}) if isinstance(assoc, dict) else {}
        freq_obj = assoc.get("HPOFrequency", {}) if isinstance(assoc, dict) else {}
        hpo_id = hpo.get("HPOId", "") if isinstance(hpo, dict) else ""
        hpo_term = hpo.get("HPOTerm", "") if isinstance(hpo, dict) else ""
        freq = freq_obj.get("Name", "") if isinstance(freq_obj, dict) else ""
        if hpo_term:
            symptoms.append((hpo_id, hpo_term, freq))

    return str(orpha_code), name, symptoms


def build_symptom_index(
    orphanet_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> "SymptomIndex":
    """
    Build symptom→disease index from raw Orphanet phenotypes JSON.

    Args:
        orphanet_path: Path to raw Orphanet phenotypes JSON. If None, uses
                      most recent file in data/raw/orphanet/phenotypes/
        output_path: Where to save the index JSON. If None, uses
                     data/normalized/symptom_index.json

    Returns:
        SymptomIndex instance
    """
    paths = get_data_paths()
    if orphanet_path is None:
        raw_dir = paths["raw"] / "orphanet" / "phenotypes"
        if not raw_dir.exists():
            raise FileNotFoundError(f"Orphanet phenotypes dir not found: {raw_dir}")
        files = sorted(raw_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise FileNotFoundError(f"No Orphanet JSON in {raw_dir}")
        orphanet_path = files[0]

    if output_path is None:
        paths["normalized"].mkdir(parents=True, exist_ok=True)
        output_path = paths["normalized"] / "symptom_index.json"

    with open(orphanet_path) as f:
        raw = json.load(f)

    # symptom_normalized -> [(orpha_code, disease_name, frequency, hpo_id), ...]
    index: dict[str, list[tuple[str, str, str, str]]] = {}
    disease_info: dict[str, dict] = {}  # orpha_code -> {name, symptoms[]}

    disorders = _extract_disorders(raw)
    for disorder in disorders:
        orpha_code, disease_name, symptoms = _extract_disease_and_symptoms(disorder)
        if not orpha_code:
            continue

        disease_info[orpha_code] = {"name": disease_name, "orpha_code": orpha_code}

        for hpo_id, hpo_term, freq in symptoms:
            norm = _normalize_symptom(hpo_term)
            if not norm:
                continue
            entry = (orpha_code, disease_name, freq, hpo_id)
            if norm not in index:
                index[norm] = []
            if entry not in index[norm]:
                index[norm].append(entry)

    payload = {
        "source": str(orphanet_path),
        "symptom_to_diseases": {k: list(v) for k, v in index.items()},
        "disease_count": len(disease_info),
        "symptom_count": len(index),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    return SymptomIndex(index_path=output_path)


class SymptomIndex:
    """Query interface for symptom→disease index."""

    def __init__(self, index_path: Optional[Path] = None):
        if index_path is None:
            index_path = PROJECT_ROOT / "data" / "normalized" / "symptom_index.json"
        self.index_path = Path(index_path)
        self._symptom_to_diseases: dict[str, list[tuple[str, str, str, str]]] = {}
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        with open(self.index_path) as f:
            data = json.load(f)
        raw = data.get("symptom_to_diseases", {})
        self._symptom_to_diseases = {
            k: [tuple(e) for e in v] for k, v in raw.items()
        }

    def _diseases_for_symptom(self, symptom: str) -> set[tuple[str, str, str]]:
        """Get (orpha_code, disease_name, frequency) set for a symptom."""
        norm = _normalize_symptom(symptom)
        if not norm:
            return set()
        direct = self._symptom_to_diseases.get(norm, [])
        result = set()
        for orpha, name, freq, _ in direct:
            result.add((orpha, name, freq))
        if not result:
            for key, entries in self._symptom_to_diseases.items():
                if norm in key or key in norm:
                    for orpha, name, freq, _ in entries:
                        result.add((orpha, name, freq))
        return result

    def query(
        self,
        symptoms: list[str],
        match_all: bool = True,
    ) -> list[dict]:
        """
        Query diseases by symptoms.

        Args:
            symptoms: List of symptom terms (e.g. ["fever", "vomiting"])
            match_all: If True, return diseases that have ALL symptoms (intersection).
                       If False, return diseases that have ANY symptom (union).

        Returns:
            List of {"orpha_code", "disease_name", "matched_symptoms", "frequency"}
        """
        if not symptoms:
            return []

        sets: list[set[tuple[str, str, str]]] = []
        for s in symptoms:
            ds = self._diseases_for_symptom(s)
            if ds:
                sets.append(ds)

        if not sets:
            return []

        if match_all:
            common = set.intersection(*sets) if len(sets) > 1 else sets[0]
        else:
            common = set.union(*sets)

        results = []
        for orpha, name, freq in common:
            results.append({
                "orpha_code": orpha,
                "disease_name": name,
                "matched_symptoms": symptoms,
                "frequency": freq,
                "orpha_url": f"https://www.orpha.net/consor/cgi-bin/OC_Exp.php?lng=en&Expert={orpha}",
            })
        return results
