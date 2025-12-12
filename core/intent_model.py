"""Universal grammar-driven intent and dialog state inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, cast

import math
import copy
import re

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_']+")


@dataclass
class PatternSpec:
    weight: float
    all_tokens: Tuple[str, ...]
    any_tokens: Tuple[str, ...]
    any_followups: Tuple[str, ...]
    start_tokens: Tuple[str, ...]
    negate_tokens: Tuple[str, ...]
    question: Optional[bool]
    window: int
    length_lt: Optional[int]

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "PatternSpec":
        return cls(
            weight=_coerce_float(payload.get("weight"), 0.0),
            all_tokens=tuple(_normalise_iter(payload.get("all_tokens", []))),
            any_tokens=tuple(_normalise_iter(payload.get("any_tokens", []))),
            any_followups=tuple(_normalise_iter(payload.get("any_followups", []))),
            start_tokens=tuple(_normalise_iter(payload.get("start_tokens", []))),
            negate_tokens=tuple(_normalise_iter(payload.get("negate_tokens", []))),
            question=_coerce_optional_bool(payload.get("question")),
            window=_coerce_int(payload.get("window"), 5),
            length_lt=_coerce_optional_int(payload.get("length_lt")),
        )


@dataclass
class IntentAnalysis:
    intent: str
    state: str
    intent_scores: Dict[str, float]
    state_scores: Dict[str, float]
    tokens: List[str]


def _normalise_iter(raw: object) -> Iterable[str]:
    if isinstance(raw, (list, tuple, set)):
        for item in raw:
            if isinstance(item, str) and item.strip():
                yield item.strip().lower()
    elif isinstance(raw, str) and raw.strip():
        yield raw.strip().lower()


def _coerce_optional_int(raw: object) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            return int(float(raw.strip()))
        except ValueError:
            return None
    return None


def _coerce_float(raw: object, default: float = 0.0) -> float:
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            return float(raw.strip())
        except ValueError:
            return default
    return default


def _coerce_int(raw: object, default: int = 0) -> int:
    value = _coerce_optional_int(raw)
    return value if value is not None else default


def _coerce_optional_bool(raw: object) -> Optional[bool]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


class GrammarIntentModel:
    """Scores intents and dialog states using declarative grammar patterns."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = self._resolve_config_path(config_path)
        self.config = self._load_config(self.config_path)
        intents_raw = cast(Dict[str, Dict[str, Any]], self.config.get("intents", {}))
        states_raw = cast(Dict[str, Dict[str, Any]], self.config.get("states", {}))
        self.intent_specs = self._build_intent_specs(intents_raw)
        self.state_specs = self._build_intent_specs(states_raw)
        transitions_raw = cast(Dict[str, Dict[str, Any]], self.config.get("transitions", {}))
        self.transitions = {
            state: {key: float(value) for key, value in mapping.items()}
            for state, mapping in transitions_raw.items()
        }
        self.initial_state = self.config.get("initial_state", "clarify")
        self._previous_state = self.initial_state

    # ------------------------------------------------------------------
    def analyse(self, utterance: str) -> IntentAnalysis:
        tokens = self._tokenise(utterance)
        features = self._extract_features(utterance, tokens)
        intent_scores = self._score_intents(tokens, features)
        chosen_intent = max(intent_scores.items(), key=lambda item: item[1])[0]
        state_scores = self._score_states(tokens, features)
        chosen_state = max(state_scores.items(), key=lambda item: item[1])[0]
        self._previous_state = chosen_state
        return IntentAnalysis(
            intent=chosen_intent,
            state=chosen_state,
            intent_scores=intent_scores,
            state_scores=state_scores,
            tokens=tokens,
        )

    # Builders ---------------------------------------------------------
    def _build_intent_specs(self, spec_payload: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        built: Dict[str, Dict[str, Any]] = {}
        for name, payload in spec_payload.items():
            base = float(payload.get("base", 0.0) or 0.0)
            patterns_raw = payload.get("patterns", []) or []
            patterns = [PatternSpec.from_dict(item) for item in cast(Iterable[Dict[str, Any]], patterns_raw)]
            built[name] = {"base": base, "patterns": patterns}
        return built

    # Scoring ----------------------------------------------------------
    def _score_intents(self, tokens: List[str], features: Dict[str, Any]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for intent_name, payload in self.intent_specs.items():
            base = cast(float, payload["base"])
            patterns: List[PatternSpec] = cast(List[PatternSpec], payload["patterns"])
            score = base
            for pattern in patterns:
                score += self._score_pattern(pattern, tokens, features)
            scores[intent_name] = score
        # Normalise by subtracting min for stability
        minimum = min(scores.values()) if scores else 0.0
        if minimum:
            for key in scores:
                scores[key] -= minimum
        return scores

    def _score_states(self, tokens: List[str], features: Dict[str, Any]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for state_name, payload in self.state_specs.items():
            base = cast(float, payload["base"])
            patterns: List[PatternSpec] = cast(List[PatternSpec], payload["patterns"])
            score = base
            for pattern in patterns:
                score += self._score_pattern(pattern, tokens, features)
            prev_state = cast(str, self._previous_state)
            transition_weight = self._transition_weight(prev_state, state_name)
            score += transition_weight
            scores[state_name] = score
        minimum = min(scores.values()) if scores else 0.0
        if minimum:
            for key in scores:
                scores[key] -= minimum
        return scores

    def _score_pattern(self, pattern: PatternSpec, tokens: List[str], features: Dict[str, Any]) -> float:
        if not pattern.weight:
            return 0.0
        token_len = cast(int, features["token_len"])
        token_set = cast(Set[str], features["token_set"])
        has_question = cast(bool, features["has_question"])
        if pattern.length_lt is not None and token_len >= pattern.length_lt:
            return 0.0
        if pattern.question is True and not has_question:
            return 0.0
        if pattern.question is False and has_question:
            return 0.0
        if pattern.negate_tokens and any(token in token_set for token in pattern.negate_tokens):
            return 0.0
        if pattern.all_tokens and not all(token in token_set for token in pattern.all_tokens):
            return 0.0
        if pattern.start_tokens and tokens:
            if tokens[0] not in pattern.start_tokens:
                return 0.0
        if pattern.any_tokens and not any(token in token_set for token in pattern.any_tokens):
            return 0.0
        if pattern.any_followups:
            if not pattern.any_tokens:
                has_follow = any(token in token_set for token in pattern.any_followups)
            else:
                has_follow = self._has_followup(tokens, pattern.any_tokens, pattern.any_followups, pattern.window)
            if not has_follow:
                return 0.0
        return pattern.weight

    def _transition_weight(self, previous_state: str, candidate_state: str) -> float:
        table = self.transitions.get(previous_state, {})
        probability = table.get(candidate_state)
        if probability is None:
            probability = table.get("default", 0.1)
        probability = max(probability, 1e-4)
        return math.log(probability)

    # Token helpers ----------------------------------------------------
    def _tokenise(self, utterance: str) -> List[str]:
        return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(utterance)]

    def _extract_features(self, utterance: str, tokens: List[str]) -> Dict[str, Any]:
        return {
            "has_question": "?" in utterance,
            "token_len": len(tokens),
            "token_set": set(tokens),
        }

    def _has_followup(
        self,
        tokens: List[str],
        anchors: Tuple[str, ...],
        followups: Tuple[str, ...],
        window: int,
    ) -> bool:
        positions = [idx for idx, token in enumerate(tokens) if token in anchors]
        if not positions:
            return False
        for pos in positions:
            end = min(len(tokens), pos + window + 1)
            for idx in range(pos + 1, end):
                if tokens[idx] in followups:
                    return True
        return False

    # Config helpers ---------------------------------------------------
    def _resolve_config_path(self, config_path: Optional[str]) -> Path:
        if config_path:
            return Path(config_path)
        base_dir = Path(__file__).resolve().parent.parent
        default_path = base_dir / "config" / "universal" / "intent_patterns.yaml"
        return default_path

    def _load_config(self, path: Path) -> Dict[str, object]:
        data: Dict[str, object] = {}
        if yaml and path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
            except Exception:
                data = {}

        if not data:
            data = {
                "intents": {
                    "conversation": {"base": 0.1, "patterns": []},
                    "definition": {"base": 0.1, "patterns": []},
                    "data_insight": {"base": 0.1, "patterns": []},
                    "resource": {"base": 0.1, "patterns": []},
                    "fallback": {"base": 0.1, "patterns": []},
                },
                "states": {
                    "clarify": {"base": 0.1, "patterns": []},
                },
                "transitions": {},
                "initial_state": "clarify",
            }

        if yaml:
            industry_path = path.parent.parent / "industry" / path.name
            if industry_path.exists():
                try:
                    with industry_path.open("r", encoding="utf-8") as handle:
                        industry_data = yaml.safe_load(handle) or {}
                    data = self._merge_configs(data, industry_data)
                except Exception:
                    pass

        return data

    @staticmethod
    def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = copy.deepcopy(base)

        base_intents = cast(Dict[str, Dict[str, Any]], merged.setdefault("intents", {}))
        for intent_name, payload in cast(Dict[str, Dict[str, Any]], override.get("intents", {})).items():
            target = base_intents.setdefault(intent_name, {})
            if "base" in payload:
                target["base"] = payload["base"]
            if "patterns" in payload:
                existing = list(target.get("patterns", []))
                existing.extend(payload.get("patterns", []))
                target["patterns"] = existing

        base_states = cast(Dict[str, Dict[str, Any]], merged.setdefault("states", {}))
        for state_name, payload in cast(Dict[str, Dict[str, Any]], override.get("states", {})).items():
            target = base_states.setdefault(state_name, {})
            if "base" in payload:
                target["base"] = payload["base"]
            if "patterns" in payload:
                existing = list(target.get("patterns", []))
                existing.extend(payload.get("patterns", []))
                target["patterns"] = existing

        base_transitions = cast(Dict[str, Dict[str, Any]], merged.setdefault("transitions", {}))
        for state_name, mapping in cast(Dict[str, Dict[str, Any]], override.get("transitions", {})).items():
            base_mapping = base_transitions.setdefault(state_name, {})
            base_mapping.update(mapping)

        if "initial_state" in override:
            merged["initial_state"] = override["initial_state"]

        return merged


__all__ = ["GrammarIntentModel", "IntentAnalysis"]
