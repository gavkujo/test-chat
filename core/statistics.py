"""Statistical insight utilities for settlement and similar tabular data."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class StatisticalPattern:
    pattern_type: str
    description: str
    score: float
    payload: Dict[str, Any]
    focus_points: List[str] = field(default_factory=list)
    cohorts: Dict[str, Any] = field(default_factory=dict)
    delta: Optional[float] = None
    sample_size: int = 0


class StatisticalInsightEngine:
    """Detect simple statistical patterns that are still explainable."""

    def __init__(self, min_points: int = 5) -> None:
        self.min_points = min_points

    def generate_insights(self, raw_data: Any, data_type: str) -> List[StatisticalPattern]:
        df = self._normalise(raw_data)
        if df.empty:
            return []

        context = self._build_context(df)

        patterns: List[StatisticalPattern] = []
        patterns.extend(self._detect_trends(df, data_type, context))
        patterns.extend(self._detect_outliers(df, context))
        patterns.extend(self._detect_correlations(df, context))
        return sorted(patterns, key=lambda p: p.score, reverse=True)

    # ---------------------------------------------------------------------
    def _normalise(self, raw_data: Any) -> pd.DataFrame:
        if raw_data is None:
            return pd.DataFrame()
        if isinstance(raw_data, pd.DataFrame):
            return raw_data.copy()
        if isinstance(raw_data, list):
            return pd.DataFrame(raw_data)
        if isinstance(raw_data, dict):
            candidate = raw_data.get("data") if "data" in raw_data else raw_data
            return pd.DataFrame(candidate)
        return pd.DataFrame()

    def _detect_trends(
        self,
        df: pd.DataFrame,
        data_type: str,
        context: Dict[str, Any],
    ) -> List[StatisticalPattern]:
        del data_type  # context carries enough signal for messaging tweaks
        numeric_cols = self._numeric_columns(df)
        if not numeric_cols:
            return []

        ordered_df = df.copy()
        order_col = context.get("date_col")
        if order_col and order_col in ordered_df.columns:
            ordered_df["__order__"] = pd.to_datetime(ordered_df[order_col], errors="coerce")
            ordered_df = ordered_df.sort_values("__order__", kind="mergesort")
            ordered_df = ordered_df.drop(columns=["__order__"])
        elif "Latest_Date" in ordered_df.columns:
            ordered_df = ordered_df.sort_values("Latest_Date")
        elif "DateTime" in ordered_df.columns:
            ordered_df = ordered_df.sort_values("DateTime")

        index = np.arange(len(ordered_df))
        patterns: List[StatisticalPattern] = []
        for col in numeric_cols:
            numeric_series = pd.to_numeric(ordered_df[col], errors="coerce")
            series = numeric_series.to_numpy(dtype=float)
            mask = ~np.isnan(series)
            sample_size = int(mask.sum())
            if sample_size < self.min_points:
                continue
            valid_series = series[mask]
            if valid_series.size == 0:
                continue
            valid_index = index[mask]
            slope = self._slope(valid_index, valid_series)
            change = float(valid_series[-1] - valid_series[0])
            magnitude = abs(slope) * (sample_size / max(self.min_points, 10))
            if abs(change) < 1e-6 and magnitude < 0.05:
                continue
            direction = "upward" if slope > 0 else "downward"
            label = self._prettify_label(col)
            primary = f"{label} shows an {direction} drift (Δ≈{change:+.2f}) across {sample_size} readings"
            plate_count = context.get("unique_counts", {}).get("plate")
            if plate_count:
                primary += f" spanning {plate_count} plate{'s' if plate_count != 1 else ''}"
            segments = [primary]
            date_range = context.get("date_range")
            if date_range:
                segments.append(f"Window {date_range[0]} → {date_range[1]}")
            berth_summary = context.get("cohort_summary", {}).get("berth")
            if berth_summary:
                segments.append(f"Top berths: {', '.join(berth_summary)}")
            description = "; ".join(segments) + "."
            focus_points = self._rank_points(ordered_df, col, context, ascending=(slope < 0))
            payload = {
                "column": col,
                "slope": slope,
                "change": change,
                "sample_size": sample_size,
                "focus_points": focus_points,
            }
            patterns.append(
                StatisticalPattern(
                    pattern_type="trend",
                    description=description,
                    score=float(min(1.0, abs(magnitude))),
                    payload=payload,
                    focus_points=focus_points,
                    cohorts=context.get("cohort_summary", {}),
                    delta=change,
                    sample_size=sample_size,
                )
            )
        return patterns

    def _detect_outliers(self, df: pd.DataFrame, context: Dict[str, Any]) -> List[StatisticalPattern]:
        patterns: List[StatisticalPattern] = []
        numeric_cols = self._numeric_columns(df)
        for col in numeric_cols:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            series = numeric_series.to_numpy(dtype=float)
            if series.size < self.min_points:
                continue
            if np.isnan(series).all():
                continue
            mean = np.nanmean(series)
            std = np.nanstd(series)
            if std == 0 or math.isnan(std):
                continue
            z_scores = (series - mean) / std
            if np.isnan(z_scores).all():
                continue
            idx = int(np.nanargmax(np.abs(z_scores)))
            z = float(z_scores[idx])
            if abs(z) < 2.0 or math.isnan(z):
                continue
            value = float(series[idx])
            row = df.iloc[idx]
            plate_id = self._value_as_str(row, context.get("plate_col"))
            berth_value = self._value_as_str(row, context.get("berth_col"))
            label = self._prettify_label(col)
            location_bits: List[str] = []
            if plate_id:
                location_bits.append(plate_id)
            if berth_value:
                location_bits.append(f"berth {berth_value}")
            location_text = " on " + ", ".join(location_bits) if location_bits else ""
            description = f"{label} spikes to {value:.2f}{location_text}, about {abs(z):.1f}σ off the cohort mean."
            row_cohorts = self._row_cohorts(row, context)
            focus_points = [plate_id] if plate_id else []
            payload = {
                "column": col,
                "index": idx,
                "value": value,
                "mean": float(mean),
                "std": float(std),
                "z_score": z,
                "focus_points": focus_points,
            }
            patterns.append(
                StatisticalPattern(
                    pattern_type="outlier",
                    description=description,
                    score=float(min(1.0, abs(z) / 5.0)),
                    payload=payload,
                    focus_points=focus_points,
                    cohorts=row_cohorts or context.get("cohort_summary", {}),
                    delta=float(value - mean),
                    sample_size=int(series.size),
                )
            )
        return patterns

    def _detect_correlations(self, df: pd.DataFrame, context: Dict[str, Any]) -> List[StatisticalPattern]:
        patterns: List[StatisticalPattern] = []
        numeric_cols = self._numeric_columns(df)
        if len(numeric_cols) < 2:
            return patterns
        subset = df[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
        if subset.empty:
            return patterns
        corr = subset.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        plate_col = context.get("plate_col")
        plate_count = context.get("unique_counts", {}).get("plate")
        sample_size = int(len(subset))
        for col in upper.columns:
            for idx, value in upper[col].dropna().items():
                if abs(value) < 0.6:
                    continue
                direction = "positive" if value > 0 else "negative"
                label_a = self._prettify_label(idx)
                label_b = self._prettify_label(col)
                segments = [f"{label_a} and {label_b} move in {direction} lockstep (ρ≈{value:.2f})"]
                if plate_count:
                    segments.append(f"Across {plate_count} plate{'s' if plate_count != 1 else ''}")
                segments.append(f"Sample size {sample_size}")
                description = "; ".join(segments) + "."
                focus_points = self._correlation_focus_points(df, idx, col, plate_col)
                payload = {
                    "columns": (idx, col),
                    "correlation": float(value),
                    "sample_size": sample_size,
                    "focus_points": focus_points,
                }
                patterns.append(
                    StatisticalPattern(
                        pattern_type="correlation",
                        description=description,
                        score=float(min(1.0, abs(value))),
                        payload=payload,
                        focus_points=focus_points,
                        cohorts=context.get("cohort_summary", {}),
                        delta=None,
                        sample_size=sample_size,
                    )
                )
        return patterns

    # Utilities -----------------------------------------------------------------
    @staticmethod
    def _numeric_columns(df: pd.DataFrame) -> List[str]:
        numeric = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric.append(col)
        return numeric

    @staticmethod
    def _slope(x: np.ndarray, y: np.ndarray) -> float:
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return 0.0
        x_f = x[mask]
        y_f = y[mask]
        slope, _ = np.polyfit(x_f, y_f, 1)
        return float(slope)

    def _build_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        context: Dict[str, Any] = {"row_count": int(len(df))}
        sanitized = {self._normalise_key(col): col for col in df.columns}
        context["sanitized"] = sanitized

        def resolve(aliases: Iterable[str]) -> Optional[str]:
            for alias in aliases:
                target = sanitized.get(self._normalise_key(alias))
                if target:
                    return target
            return None

        context["plate_col"] = resolve(["PointID", "point_id", "plate", "plateid", "grid", "full_name", "id", "name"])
        context["berth_col"] = resolve(["berth"])
        context["area_col"] = resolve(["area", "region", "zone"])
        context["status_col"] = resolve(["status", "state", "flag"])
        context["date_col"] = resolve(["Latest_Date", "latest_date", "Date", "date", "datetime", "timestamp", "reading_date"])

        label_map = {
            "plate_col": "plate",
            "berth_col": "berth",
            "area_col": "area",
            "status_col": "status",
        }
        unique_counts: Dict[str, int] = {}
        for key, label in label_map.items():
            col = context.get(key)
            if col and col in df.columns:
                unique_counts[label] = int(df[col].dropna().nunique())
        context["unique_counts"] = unique_counts

        cohort_summary: Dict[str, List[str]] = {}
        for key, label in (("berth_col", "berth"), ("area_col", "area"), ("status_col", "status")):
            col = context.get(key)
            if not col or col not in df.columns:
                continue
            values = df[col].dropna()
            if values.empty:
                continue
            top = values.astype(str).value_counts().head(3)
            cohort_summary[label] = [f"{idx} ({int(cnt)})" for idx, cnt in top.items()]
        context["cohort_summary"] = cohort_summary

        date_col = context.get("date_col")
        context["date_range"] = None
        if date_col and date_col in df.columns:
            converted = pd.to_datetime(df[date_col], errors="coerce")
            converted = converted.dropna()
            if not converted.empty:
                context["date_range"] = (
                    self._format_date(converted.min()),
                    self._format_date(converted.max()),
                )
        return context

    def _rank_points(
        self,
        df: pd.DataFrame,
        value_col: str,
        context: Dict[str, Any],
        ascending: bool,
        limit: int = 3,
    ) -> List[str]:
        plate_col = context.get("plate_col")
        if not plate_col or plate_col not in df.columns or value_col not in df.columns:
            return []
        subset = df[[plate_col, value_col]].copy()
        subset[value_col] = pd.to_numeric(subset[value_col], errors="coerce")
        subset = subset.dropna(subset=[value_col])
        if subset.empty:
            return []
        ordered = subset.sort_values(value_col, ascending=ascending)
        return ordered[plate_col].astype(str).head(limit).tolist()

    def _row_cohorts(self, row: pd.Series, context: Dict[str, Any]) -> Dict[str, List[str]]:
        cohorts: Dict[str, List[str]] = {}
        for key, label in (("berth_col", "berth"), ("area_col", "area"), ("status_col", "status")):
            col = context.get(key)
            if col and col in row and not pd.isna(row[col]):
                cohorts[label] = [str(row[col])]
        return cohorts

    def _correlation_focus_points(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
        plate_col: Optional[str],
    ) -> List[str]:
        if not plate_col or plate_col not in df.columns:
            return []
        subset = df[[plate_col, col_a, col_b]].copy()
        subset[col_a] = pd.to_numeric(subset[col_a], errors="coerce")
        subset[col_b] = pd.to_numeric(subset[col_b], errors="coerce")
        subset = subset.dropna(subset=[col_a, col_b])
        if subset.empty:
            return []
        mean_a = subset[col_a].mean()
        mean_b = subset[col_b].mean()
        subset["__strength__"] = (subset[col_a] - mean_a).abs() + (subset[col_b] - mean_b).abs()
        ordered = subset.sort_values("__strength__", ascending=False)
        return ordered[plate_col].astype(str).head(3).tolist()

    @staticmethod
    def _normalise_key(value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())

    @staticmethod
    def _prettify_label(name: str) -> str:
        cleaned = name.replace("_", " ").replace("-", " ").strip()
        if not cleaned:
            return name
        return " ".join(part.capitalize() for part in cleaned.split())

    @staticmethod
    def _format_date(value: Any) -> str:
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        try:
            return pd.to_datetime(value).strftime("%Y-%m-%d")
        except Exception:
            return str(value)

    @staticmethod
    def _value_as_str(row: pd.Series, column: Optional[str]) -> Optional[str]:
        if not column or column not in row:
            return None
        value = row[column]
        if pd.isna(value):
            return None
        return str(value)


__all__ = ["StatisticalInsightEngine", "StatisticalPattern"]
