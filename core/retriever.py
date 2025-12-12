"""Glossary-backed retrieval for definition/process intents."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None  # type: ignore


_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{1,}")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "have",
    "has",
    "had",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "it's",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "too",
    "up",
    "was",
    "we",
    "what",
    "when",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


_DEFAULT_GLOSSARY: List[Dict[str, str]] = [
    {
        "term": "Asaoka method",
        "definition": (
            "A graphical technique that extrapolates settlement readings to estimate the final consolidation level."
        ),
    },
    {
        "term": "Settlement plate",
        "definition": (
            "Instrumentation installed on reclaimed land to measure vertical displacement during surcharge loading."
        ),
    },
    {
        "term": "Degree of consolidation",
        "definition": (
            "Percentage indicator describing how close the soil is to finishing primary consolidation under surcharge."
        ),
    },
    {
        "term": "Holding period",
        "definition": (
            "Elapsed days between surcharge completion and the latest reading, used to confirm that settlement rates taper off."
        ),
    },
    {
        "term": "7-day settlement rate",
        "definition": (
            "Change in settlement over the past week expressed in millimetres; compliance requires it to stay at or below 4 mm."
        ),
    },
    {
        "term": "Ground level (mCD)",
        "definition": (
            "Elevation of the plate location relative to chart datum. Compliance for Phase 2 requires it to stay at or above 16.9 mCD."
        ),
    },
    {
        "term": "Surcharge load",
        "definition": (
            "Temporary sand fill placed to accelerate soil consolidation before permanent structures are built."
        ),
    },
    {
        "term": "Surcharge complete date",
        "definition": (
            "The date the surcharge placement finished; it marks the start point for the holding period and consolidation tracking."
        ),
    },
    {
        "term": "Compliance criteria",
        "definition": (
            "Project checks requiring Asaoka DOC ≥ 90%, ground level ≥ 16.9 mCD, and the 7-day settlement rate ≤ 4 mm."
        ),
    },
    {
        "term": "Tuas Terminal Phase 2",
        "definition": (
            "Singapore mega-port reclamation programme: 365 hectares of land, 9 km wharf, and ~1900 settlement plates monitoring soil improvement."
        ),
    },
]


def _tokenise(text: str) -> Counter:
    tokens = [token.lower() for token in _TOKEN_PATTERN.findall(text)]
    filtered = [token for token in tokens if token not in _STOPWORDS]
    if not filtered:
        filtered = tokens
    return Counter(filtered)


def _normalise_phrase(text: str) -> str:
    stripped = re.sub(r"[^A-Za-z0-9]+", " ", text.lower())
    return stripped.strip()


def _ensure_vector(counter: Counter, fallback_text: str) -> Counter:
    if counter:
        return counter
    baseline_tokens = [token for token in fallback_text.lower().split() if token]
    return Counter(baseline_tokens)


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    intersection = set(a) & set(b)
    numerator = sum(a[t] * b[t] for t in intersection)
    norm_a = sum(v * v for v in a.values()) ** 0.5
    norm_b = sum(v * v for v in b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return numerator / (norm_a * norm_b)


def _default_glossary_path() -> str:
    base_dir = Path(__file__).resolve().parent.parent
    return str(base_dir / "config" / "industry" / "glossary.yaml")


class GlossaryRetriever:
    def __init__(self, glossary_path: Optional[str] = None) -> None:
        self.glossary = self._load_glossary(glossary_path)
        self._definition_vectors = [
            _ensure_vector(_tokenise(entry["definition"]), entry["definition"])
            for entry in self.glossary
        ]
        self._term_vectors = [
            _ensure_vector(_tokenise(entry["term"]), entry["term"])
            for entry in self.glossary
        ]
        self._term_lookup = {entry["term"].lower(): entry for entry in self.glossary}
        self._term_weight = 1.4
        self._direct_bonus = 0.6
        self._substring_bonus = 0.2

    def _load_glossary(self, glossary_path: Optional[str]) -> List[Dict[str, str]]:
        if glossary_path is None:
            glossary_path = _default_glossary_path()
        path = Path(str(glossary_path))
        if not path.exists():
            return list(_DEFAULT_GLOSSARY)
        data: Optional[object]
        try:
            if path.suffix in {".yaml", ".yml"} and yaml is not None:
                with path.open("r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                data = loaded.get("glossary") if isinstance(loaded, dict) else loaded
            else:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
        except Exception:
            return list(_DEFAULT_GLOSSARY)
        if not isinstance(data, list):
            return list(_DEFAULT_GLOSSARY)
        normalised: List[Dict[str, str]] = []
        for entry in data:
            if isinstance(entry, dict) and "term" in entry and "definition" in entry:
                normalised.append({"term": str(entry["term"]), "definition": str(entry["definition"])})
        return normalised or list(_DEFAULT_GLOSSARY)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        normalised_query = _normalise_phrase(query)
        if not normalised_query:
            return []
        query_vec = _tokenise(query)
        query_term_vec = _ensure_vector(_tokenise(normalised_query), normalised_query)

        scores: List[Tuple[float, Dict[str, Any]]] = []
        for entry, def_vec, term_vec in zip(self.glossary, self._definition_vectors, self._term_vectors):
            score_def = _cosine(query_vec, def_vec)
            score_term = _cosine(query_term_vec, term_vec)
            composite = score_def
            if score_term > 0:
                composite += self._term_weight * score_term
            entry_term = entry["term"].lower()
            if entry_term == normalised_query:
                composite += self._direct_bonus
            elif normalised_query in entry_term:
                composite += self._substring_bonus
            if composite > 0:
                enriched: Dict[str, Any] = dict(entry)
                enriched["score"] = float(composite)
                scores.append((composite, enriched))

        direct = self._term_lookup.get(normalised_query)
        if direct and not any(item[1]["term"].lower() == normalised_query for item in scores):
            enriched_direct: Dict[str, Any] = dict(direct)
            enriched_direct["score"] = float(self._direct_bonus + self._term_weight)
            scores.append((enriched_direct["score"], enriched_direct))

        if not scores:
            return []
        scores.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scores[:top_k]]

    def format_definitions(self, query: str, top_k: int = 3) -> str:
        matches = self.search(query, top_k=top_k)
        if not matches:
            return ""
        lines = [f"{match['term']}: {match['definition']}" for match in matches]
        return "\n".join(lines)


__all__ = ["GlossaryRetriever"]
