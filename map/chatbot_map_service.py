"""Utilities to parse chatbot prompts, call the GeoChat API, and render map responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests

from map.map_builder import SITE_PREFIX, STATUS_ORDER, create_map_response

API_BASE_URL = "http://172.16.181.2:8887/geochat"

PLATE_ID_PATTERN = re.compile(
    rf"(?i){SITE_PREFIX}-R(?P<num>\d{{2}})(?P<suffix>[A-Za-z]*)-SM-(?P<plate>\d{{2}})"
)
REGION_CODE_PATTERN = re.compile(r"(?i)R(?P<num>\d{2})(?P<suffix>[A-Za-z]*)")
AREA_TOKEN_PATTERN = re.compile(r"\b([IVXLCDM]+)([a-dA-D])\b")
BERTH_TOKEN_PATTERN = re.compile(r"\b([AB]\d{1,2})\b", re.IGNORECASE)

PROBLEM_KEYWORDS = [
    "problem",
    "problematic",
    "issue",
    "issues",
    "not ok",
    "not okay",
    "non compliant",
    "non-compliant",
    "underperform",
    "under-performing",
    "bad",
    "fail",
    "failing",
    "threshold",
]
OK_KEYWORDS = ["ok", "okay", "ready", "satisfied", "completed", "good"]
PRESSURE_KEYWORDS = ["pressure", "kpa"]
SETTLEMENT_KEYWORDS = ["settlement", "mm"]

FULL_MAP_PATTERNS = (
    re.compile(r"\b(entire|whole|full)\s+(map|site)\b", re.IGNORECASE),
    re.compile(r"\bshow(?:\s+me)?\s+the\s+(?:entire|whole|full)\s+map\b", re.IGNORECASE),
    re.compile(r"\bmap\s+the\s+whole\s+site\b", re.IGNORECASE),
    re.compile(r"\bmap\s+everything\b", re.IGNORECASE),
    re.compile(r"\ball\s+plates\s+(?:on|across)\s+the\s+map\b", re.IGNORECASE),
)


class QueryParsingError(Exception):
    """Raised when the prompt cannot be deterministically interpreted."""


class LookupExecutionError(Exception):
    """Raised when the GeoChat lookup fails or returns an unexpected payload."""


@dataclass
class QueryIntent:
    raw_prompt: str
    lookup: str
    features: List[str]
    status_filter: Optional[Set[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.raw_prompt,
            "lookup": self.lookup,
            "features": self.features,
            "status_filter": sorted(self.status_filter) if self.status_filter else [],
        }

    @property
    def features_param(self) -> str:
        return ",".join(self.features)


@dataclass
class MapDataResult:
    intent: QueryIntent
    api_url: Optional[str]
    records: List[Dict[str, Any]]
    raw_records: List[Dict[str, Any]]
    highlight_ids: Set[str]
    total_records: int

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "intent": self.intent.to_dict(),
            "records_fetched": self.total_records,
            "records_after_filter": len(self.records),
            "highlight_ids": sorted(self.highlight_ids),
        }
        if self.api_url:
            payload["api_url"] = self.api_url
        return payload


@dataclass
class MapResult:
    response: Dict[str, Any]
    data: MapDataResult

    def to_dict(self) -> Dict[str, Any]:
        return self.data.to_dict()

    @property
    def intent(self) -> QueryIntent:
        return self.data.intent

    @property
    def api_url(self) -> Optional[str]:
        return self.data.api_url

    @property
    def records_fetched(self) -> int:
        return self.data.total_records

    @property
    def records_after_filter(self) -> int:
        return len(self.data.records)

    @property
    def highlight_ids(self) -> Set[str]:
        return self.data.highlight_ids

    @property
    def records(self) -> List[Dict[str, Any]]:
        return self.data.records

    @property
    def raw_records(self) -> List[Dict[str, Any]]:
        return self.data.raw_records


def normalize_plate_id(value: str) -> Optional[str]:
    if not value:
        return None
    match = PLATE_ID_PATTERN.search(value.strip())
    if not match:
        return None
    num = match.group("num")
    suffix = (match.group("suffix") or "").lower()
    plate = match.group("plate")
    return f"{SITE_PREFIX}-R{num}{suffix}-SM-{plate}"


def normalize_region_code(token: str) -> Optional[str]:
    if not token:
        return None
    match = REGION_CODE_PATTERN.fullmatch(token.strip())
    if not match:
        return None
    num = match.group("num")
    suffix = (match.group("suffix") or "").lower()
    return f"R{num}{suffix}"


def normalize_area_token(token: str) -> Optional[str]:
    if not token:
        return None
    match = AREA_TOKEN_PATTERN.fullmatch(token.strip())
    if not match:
        return None
    roman, suffix = match.groups()
    return f"{roman.upper()}{suffix.lower()}"


def _extract_plate_ids(prompt: str) -> Set[str]:
    return {
        normalize_plate_id(match.group(0))
        for match in PLATE_ID_PATTERN.finditer(prompt)
        if normalize_plate_id(match.group(0))
    }


def _extract_region_codes(prompt: str) -> Set[str]:
    return {
        normalize_region_code(match.group(0))
        for match in REGION_CODE_PATTERN.finditer(prompt)
        if normalize_region_code(match.group(0))
    }


def _extract_area_tokens(prompt: str) -> Set[str]:
    return {
        normalize_area_token(match.group(0))
        for match in AREA_TOKEN_PATTERN.finditer(prompt)
        if normalize_area_token(match.group(0))
    }


def _extract_berth_tokens(prompt: str) -> Set[str]:
    return {match.group(1).upper() for match in BERTH_TOKEN_PATTERN.finditer(prompt)}


def _detect_explicit_statuses(prompt: str) -> Set[str]:
    matches: Set[str] = set()
    for section_match in re.finditer(r"status(?:es)?[^:]*[:=]?([^\n.;]+)", prompt, flags=re.IGNORECASE):
        section = section_match.group(1).upper()
        matches.update(re.findall(r"[OPXS]", section))
    return {status for status in matches if status in STATUS_ORDER}


def detect_status_filter(prompt: str) -> Optional[Set[str]]:
    lowered = prompt.lower()
    explicit = _detect_explicit_statuses(prompt)
    if explicit:
        return explicit

    problematic = any(keyword in lowered for keyword in PROBLEM_KEYWORDS)
    ok_request = any(keyword in lowered for keyword in OK_KEYWORDS)
    contains_not_ok = "not ok" in lowered or "not okay" in lowered

    statuses: Set[str] = set()
    if problematic:
        statuses.update({"S", "P", "X"})

    if any(term in lowered for term in PRESSURE_KEYWORDS) and (
        "issue" in lowered or "problem" in lowered or "under" in lowered or "below" in lowered
    ):
        statuses.add("P")

    if any(term in lowered for term in SETTLEMENT_KEYWORDS) and (
        "issue" in lowered or "problem" in lowered or "over" in lowered or "above" in lowered
    ):
        statuses.add("S")

    if "critical" in lowered or "alarm" in lowered:
        statuses.add("X")

    if ok_request and not problematic and not contains_not_ok:
        statuses = {"O"}

    return statuses or None


def wants_full_map(prompt: str) -> bool:
    lowered = prompt.lower()
    if "whole map" in lowered or "entire map" in lowered or "full map" in lowered:
        return True
    if "whole site" in lowered or "entire site" in lowered or "full site" in lowered:
        return True
    for pattern in FULL_MAP_PATTERNS:
        if pattern.search(prompt):
            return True
    return False


def parse_query_intent(prompt: str) -> QueryIntent:
    plate_ids = _extract_plate_ids(prompt)
    area_tokens = _extract_area_tokens(prompt)
    region_codes = _extract_region_codes(prompt)
    berth_tokens = _extract_berth_tokens(prompt)

    lowered = prompt.lower()
    status_filter = detect_status_filter(prompt)

    if wants_full_map(prompt):
        return QueryIntent(
            raw_prompt=prompt,
            lookup="all",
            features=["all"],
            status_filter=status_filter,
        )

    if plate_ids:
        return QueryIntent(
            raw_prompt=prompt,
            lookup="SM",
            features=sorted(plate_ids),
            status_filter=status_filter,
        )

    if area_tokens or region_codes:
        features = sorted(area_tokens or region_codes)
        return QueryIntent(
            raw_prompt=prompt,
            lookup="area",
            features=features,
            status_filter=status_filter,
        )

    if "berth" in lowered and berth_tokens:
        features = sorted(berth_tokens)
        return QueryIntent(
            raw_prompt=prompt,
            lookup="berth",
            features=features,
            status_filter=status_filter,
        )

    if berth_tokens:
        return QueryIntent(
            raw_prompt=prompt,
            lookup="berth",
            features=sorted(berth_tokens),
            status_filter=status_filter,
        )

    raise QueryParsingError("Unable to determine lookup scope from prompt. Consider naming berths, areas, or plate IDs explicitly.")


def build_api_url(intent: QueryIntent) -> str:
    if not intent.features:
        raise QueryParsingError("At least one lookup feature is required to call the API.")
    return f"{API_BASE_URL}/{intent.lookup}&{intent.features_param}"


def fetch_lookup_records(intent: QueryIntent, timeout: float = 10.0) -> List[Dict[str, Any]]:
    url = build_api_url(intent)
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise LookupExecutionError(f"GeoChat API request failed: {exc}") from exc

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise LookupExecutionError("GeoChat API did not return valid JSON.") from exc

    if not isinstance(payload, list):
        raise LookupExecutionError("GeoChat API response must be a JSON list of plate records.")

    return payload


def filter_records_by_status(
    records: Iterable[Dict[str, Any]],
    status_filter: Optional[Set[str]],
) -> Tuple[List[Dict[str, Any]], Set[str], int]:
    filtered: List[Dict[str, Any]] = []
    highlight_ids: Set[str] = set()
    total = 0

    for record in records:
        total += 1
        normalized = dict(record)

        point_id_raw = record.get("PointID") or record.get("point_id") or record.get("PointId")
        point_id = normalize_plate_id(point_id_raw) if isinstance(point_id_raw, str) else None
        if point_id:
            normalized["PointID"] = point_id

        def _clean_value(value: Any) -> Any:
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped or stripped.lower() in {"null", "none", "n/a", "na"}:
                    return None
                return stripped
            return value

        def _assign(canonical: str, *aliases: str) -> None:
            for key in (canonical,) + aliases:
                if key in record:
                    candidate = _clean_value(record.get(key))
                    if candidate is not None:
                        normalized[canonical] = candidate
                        return

        _assign("7day_rate", "seven_day_rate", "weekly_rate")
        _assign("Surcharge_Pressure", "surcharge_pressure", "surcharge_pressure_kpa")
        _assign("Asaoka_DOC", "asaoka_doc", "degreeofconsolidation", "asaokaDoc")

        status_value = record.get("Status") or record.get("status")
        status = status_value.strip().upper() if isinstance(status_value, str) else None
        if status:
            normalized["Status"] = status

        matches_filter = True
        if status_filter:
            matches_filter = bool(status and status in status_filter)

        if matches_filter:
            filtered.append(normalized)
            if point_id:
                highlight_ids.add(point_id)
        elif not status_filter and point_id:
            highlight_ids.add(point_id)

    if not status_filter:
        highlight_ids = {record.get("PointID") for record in filtered if record.get("PointID")}

    return filtered, highlight_ids, total


def lookup_prompt_records(
    prompt: str,
    *,
    records_override: Optional[List[Dict[str, Any]]] = None,
    timeout: float = 10.0,
) -> MapDataResult:
    intent = parse_query_intent(prompt)

    if records_override is not None and not isinstance(records_override, list):
        raise LookupExecutionError("records_override must be a list of plate record dictionaries.")

    raw_records_input: List[Dict[str, Any]]
    if records_override is not None:
        raw_records_input = list(records_override)
        api_url = None
    else:
        raw_records_input = fetch_lookup_records(intent, timeout=timeout)
        api_url = build_api_url(intent)

    filtered_records, highlight_ids, total_records = filter_records_by_status(
        raw_records_input,
        intent.status_filter,
    )

    return MapDataResult(
        intent=intent,
        api_url=api_url,
        records=filtered_records,
        raw_records=list(raw_records_input),
        highlight_ids=highlight_ids,
        total_records=total_records,
    )


def build_map_for_prompt(
    prompt: str,
    *,
    map_layout: Optional[Dict[str, Dict[str, Any]]] = None,
    records_override: Optional[List[Dict[str, Any]]] = None,
    timeout: float = 10.0,
) -> MapResult:
    data_result = lookup_prompt_records(
        prompt,
        records_override=records_override,
        timeout=timeout,
    )

    highlight_arg: Optional[Set[str]] = data_result.highlight_ids if data_result.intent.status_filter else None

    response = create_map_response(
        query=prompt,
        map_layout=map_layout,
        plate_records=data_result.records,
        highlight_grids=highlight_arg,
    )

    if not data_result.records and data_result.total_records:
        response["response_message"] = (
            "No records satisfied the current filter; non-matching plates are shown in grey."
        )
    elif not data_result.records:
        response["response_message"] = "No plate records were returned for this lookup."

    response_metadata = {
        "lookup": data_result.intent.lookup,
        "lookup_features": data_result.intent.features,
        "status_filter": sorted(data_result.intent.status_filter) if data_result.intent.status_filter else [],
        "records_fetched": data_result.total_records,
        "records_after_filter": len(data_result.records),
        "records_filtered_out": max(data_result.total_records - len(data_result.records), 0),
    }

    if data_result.api_url:
        response_metadata["api_url"] = data_result.api_url

    response["metadata"].update(response_metadata)

    return MapResult(
        response=response,
        data=data_result,
    )