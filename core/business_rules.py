"""Business rule mapping for contextualising statistical insights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass
class BusinessRule:
    name: str
    threshold: float
    comparison: str
    description: str
    recommendation: str


_DEFAULT_RULES: Dict[str, List[BusinessRule]] = {
    "plates_data": [
    ]
}


class BusinessRuleEngine:
    _COLUMN_ALIASES: Dict[str, Iterable[str]] = {
        "plate": ("PointID", "point_id", "plate", "plateid", "grid", "full_name", "id", "name"),
        "berth": ("berth",),
        "area": ("area", "region", "zone"),
        "status": ("status", "state", "flag"),
    }

    def __init__(self, rule_map: Optional[Dict[str, List[BusinessRule]]] = None, config_path: Optional[str] = None) -> None:
        if rule_map is not None:
            self.rule_map = rule_map
        else:
            loaded = self._load_rules_from_config(config_path)
            self.rule_map = loaded or _DEFAULT_RULES

    def apply_business_context(
        self,
        insights: List[Dict[str, Any]],
        data_type: str,
        raw_data: Any,
    ) -> List[Dict[str, Any]]:
        df = self._normalise(raw_data)
        contextualised = [self._attach_recommendation(insight, df, data_type) for insight in insights]
        rule_breaches = self._evaluate_rules(df, data_type)
        return contextualised + rule_breaches

    def _attach_recommendation(
        self,
        insight: Dict[str, Any],
        df: pd.DataFrame,
        data_type: str,
    ) -> Dict[str, Any]:
        payload = insight.get("payload") or {}
        focus_points = self._extract_focus_points(insight)
        focus_label = self._format_focus_points(focus_points)
        score = float(insight.get("score", 0) or 0)
        prefix = self._severity_prefix(score)

        pattern_type = insight.get("pattern_type")
        recommendation = ""
        if pattern_type == "trend":
            column = payload.get("column", "the metric")
            change = self._safe_float(insight.get("delta", payload.get("change")))
            slope = self._safe_float(payload.get("slope")) or 0.0
            direction = "rising" if slope > 0 else "falling"
            change_text = f" by {change:+.2f}" if change is not None else ""
            recommendation = (
                f"{prefix} track {column} for {focus_label}; the trend is {direction}{change_text}. "
                "Capture a confirmation reading and prep mitigation if the drift continues."
            )
        elif pattern_type == "outlier":
            value = self._safe_float(payload.get("value"))
            z_score = abs(self._safe_float(payload.get("z_score")) or 0.0)
            value_text = f"{value:.2f}" if value is not None else "an outlying value"
            sigma_text = f" ({z_score:.1f}σ)" if z_score else ""
            recommendation = (
                f"{prefix} re-run instrumentation checks for {focus_label}; reading {value_text}{sigma_text} sits off baseline. "
                "Validate sensor calibration and site conditions."
            )
        elif pattern_type == "correlation":
            columns = payload.get("columns") or ()
            paired = " and ".join(str(col) for col in columns if col)
            if not paired:
                paired = "the paired metrics"
            recommendation = (
                f"{prefix} review activities impacting {paired}; coordinate monitoring for {focus_label} given the coupled movement."
            )
        else:
            recommendation = (
                f"{prefix} keep {focus_label} on the watchlist and capture another update before taking action."
            )

        if focus_points and not insight.get("focus_points"):
            insight["focus_points"] = focus_points
        insight["business_context"] = recommendation
        insight["recommendation"] = recommendation
        insight["data_type"] = data_type
        return insight

    def _evaluate_rules(self, df: pd.DataFrame, data_type: str) -> List[Dict[str, Any]]:
        rules = self.rule_map.get(data_type, [])
        if df.empty or not rules:
            return []
        highlights: List[Dict[str, Any]] = []
        columns = self._detect_columns(df)
        plate_col = columns.get("plate")
        total_rows = int(len(df))
        for rule in rules:
            metric_col = self._match_rule_column(df, rule.name)
            if not metric_col:
                continue
            series = pd.to_numeric(df[metric_col], errors="coerce")
            if series.empty:
                continue
            mask, direction = self._rule_mask(series, rule)
            if mask is None or not mask.any():
                continue
            offending = df.loc[mask]
            affected_count = int(mask.sum())
            delta_series = series[mask] - rule.threshold
            if rule.comparison.strip() in {">=", ">"}:
                worst_delta = float(delta_series.min())
            else:
                worst_delta = float(delta_series.max())
            focus_points: List[str] = []
            if plate_col and plate_col in offending.columns:
                focus_points = offending[plate_col].astype(str).head(5).tolist()
            cohorts = self._summarise_cohorts(offending, columns)
            description = (
                f"{affected_count} of {total_rows} records show {metric_col} {direction} the target {rule.threshold:.2f}."
            )
            if worst_delta:
                description += f" Worst gap ≈ {worst_delta:+.2f}."
            if cohorts:
                cohort_lines = [f"{label}: {', '.join(values)}" for label, values in cohorts.items() if values]
                if cohort_lines:
                    description += " Key clusters: " + "; ".join(cohort_lines) + "."

            payload = {
                "metric": metric_col,
                "threshold": rule.threshold,
                "comparison": rule.comparison,
                "focus_points": focus_points,
                "worst_delta": worst_delta,
                "count": affected_count,
                "sample_size": total_rows,
            }
            focus_label = self._format_focus_points(focus_points)
            gap_text = f"gap of {abs(worst_delta):.2f}" if worst_delta else "gap"
            recommendation = f"Escalate: {rule.recommendation} Focus on {focus_label}; {gap_text} remains.".strip()

            highlights.append(
                {
                    "pattern_type": "rule_breach",
                    "description": description.strip(),
                    "rows": offending.to_dict(orient="records"),
                    "business_context": recommendation,
                    "recommendation": recommendation,
                    "data_type": data_type,
                    "focus_points": focus_points,
                    "cohorts": cohorts,
                    "delta": worst_delta,
                    "threshold": rule.threshold,
                    "comparison": rule.comparison,
                    "payload": payload,
                }
            )
        return highlights

    @staticmethod
    def _normalise(raw_data: Any) -> pd.DataFrame:
        if raw_data is None:
            return pd.DataFrame()
        if isinstance(raw_data, pd.DataFrame):
            return raw_data.copy()
        if isinstance(raw_data, list):
            return pd.DataFrame(raw_data)
        if isinstance(raw_data, dict):
            candidate = raw_data.get("data") if "data" in raw_data else raw_data
            if isinstance(candidate, list):
                return pd.DataFrame(candidate)
        return pd.DataFrame()

    def _load_rules_from_config(self, config_path: Optional[str]) -> Optional[Dict[str, List[BusinessRule]]]:
        path = Path(config_path) if config_path else self._default_rule_path()
        if not path or not path.exists():
            return None
        if yaml is None:
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        rule_map: Dict[str, List[BusinessRule]] = {}
        for data_type, entries in data.items():
            if not isinstance(entries, list):
                continue
            bucket: List[BusinessRule] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                try:
                    bucket.append(
                        BusinessRule(
                            name=str(entry["name"]),
                            threshold=float(entry["threshold"]),
                            comparison=str(entry.get("comparison", ">=")).strip(),
                            description=str(entry.get("description", "")),
                            recommendation=str(entry.get("recommendation", "")),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            if bucket:
                rule_map[data_type] = bucket
        return rule_map or None

    @staticmethod
    def _default_rule_path() -> Optional[Path]:
        base_dir = Path(__file__).resolve().parent.parent
        path = base_dir / "config" / "industry" / "business_rules.yaml"
        return path if path.exists() else None

    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        mapping: Dict[str, Optional[str]] = {}
        sanitized = {self._normalise_key(col): col for col in df.columns}
        for key, aliases in self._COLUMN_ALIASES.items():
            mapping[key] = self._resolve_from_map(sanitized, aliases)
        return mapping

    def _resolve_from_map(self, mapping: Dict[str, str], aliases: Iterable[str]) -> Optional[str]:
        for alias in aliases:
            target = mapping.get(self._normalise_key(alias))
            if target:
                return target
        return None

    def _match_rule_column(self, df: pd.DataFrame, name: str) -> Optional[str]:
        sanitized = {self._normalise_key(col): col for col in df.columns}
        key = self._normalise_key(name)
        if key in sanitized:
            return sanitized[key]
        for norm, original in sanitized.items():
            if key in norm:
                return original
        return None

    @staticmethod
    def _normalise_key(value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())

    def _rule_mask(self, series: pd.Series, rule: BusinessRule) -> Tuple[Optional[pd.Series], str]:
        comparison = rule.comparison.strip()
        if comparison == ">=":
            return (series.lt(rule.threshold) & series.notna(), "below")
        if comparison == "<=":
            return (series.gt(rule.threshold) & series.notna(), "above")
        if comparison == ">":
            return (series.le(rule.threshold) & series.notna(), "at or below")
        if comparison == "<":
            return (series.ge(rule.threshold) & series.notna(), "at or above")
        return None, ""

    def _summarise_cohorts(self, df: pd.DataFrame, columns: Dict[str, Optional[str]]) -> Dict[str, List[str]]:
        summary: Dict[str, List[str]] = {}
        for key, label in (("berth", "berth"), ("area", "area"), ("status", "status")):
            col = columns.get(key)
            if not col or col not in df.columns:
                continue
            values = df[col].dropna()
            if values.empty:
                continue
            top = values.astype(str).value_counts().head(3)
            summary[label] = [f"{idx} ({int(cnt)})" for idx, cnt in top.items()]
        return summary

    def _extract_focus_points(self, insight: Dict[str, Any]) -> List[str]:
        focus = insight.get("focus_points")
        if not focus and isinstance(insight.get("payload"), dict):
            focus = insight["payload"].get("focus_points")
        if isinstance(focus, list):
            return [str(item) for item in focus if item not in (None, "")]
        if focus:
            return [str(focus)]
        return []

    @staticmethod
    def _format_focus_points(points: Iterable[str]) -> str:
        items = [str(point) for point in points if point]
        if not items:
            return "the affected plates"
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return f"{items[0]}, {items[1]}, and {items[2]}"

    @staticmethod
    def _severity_prefix(score: float) -> str:
        if score >= 0.75:
            return "Escalate:"
        if score >= 0.5:
            return "Action:"
        return "Monitor:"

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


__all__ = ["BusinessRuleEngine", "BusinessRule"]
