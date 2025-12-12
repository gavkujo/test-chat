# map_builder.py - Reusable map construction utilities for chatbot integration
import plotly.graph_objects as go
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import random
from datetime import datetime, timedelta
import re
from collections import Counter

SITE_PREFIX = "F3"

STATUS_STYLES: Dict[str, Dict[str, str]] = {
    "O": {"color": "#4CAF50", "label": "OK"},
    "S": {"color": "#F8BBD0", "label": "Settlement Issue"},
    "P": {"color": "#FF9800", "label": "Pressure Issue"},
    "X": {"color": "#F44336", "label": "Critical Issue"},
    "NA": {"color": "#B0BEC5", "label": "No Data"},
}

STATUS_ORDER: List[str] = ["O", "S", "P", "X"]

ROMAN_SYMBOLS = {
    "M": 1000,
    "D": 500,
    "C": 100,
    "L": 50,
    "X": 10,
    "V": 5,
    "I": 1,
}

AREA_PATTERN = re.compile(r"^(?P<roman>[IVXLCDM]+)(?P<suffix>[A-Za-z0-9]*)$", re.IGNORECASE)


def roman_to_int(value: str) -> Optional[int]:
    """Convert a Roman numeral string to an integer."""
    if not value:
        return None
    total = 0
    prev_value = 0
    for char in value.upper():
        if char not in ROMAN_SYMBOLS:
            return None
        current = ROMAN_SYMBOLS[char]
        if current > prev_value:
            total += current - 2 * prev_value
        else:
            total += current
        prev_value = current
    return total


_fallback_area_index = 1
_fallback_area_map: Dict[str, str] = {}


def area_to_region_code(area_name: str) -> str:
    """Convert a berth area label (Roman numerals) into the canonical region code."""
    global _fallback_area_index
    area_name = area_name.strip()
    match = AREA_PATTERN.match(area_name)
    if match:
        roman_part = match.group("roman") or ""
        suffix_raw = match.group("suffix") or ""
        roman_value = roman_to_int(roman_part)
        if roman_value is not None:
            suffix_letters = "".join(ch.lower() for ch in suffix_raw if ch.isalpha())
            return f"R{roman_value:02d}{suffix_letters}"

    if area_name not in _fallback_area_map:
        _fallback_area_map[area_name] = f"R{_fallback_area_index:02d}"
        _fallback_area_index += 1
    return _fallback_area_map[area_name]


def status_label(status: str) -> str:
    style = STATUS_STYLES.get(status)
    if style:
        return style["label"]
    return status


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_value(value: Any, precision: Optional[int] = None, fallback: str = "—") -> str:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        if precision is not None:
            return f"{value:.{precision}f}"
        return f"{value}"
    return str(value)

# ============================================================================
# 1. DATA STRUCTURES
# ============================================================================

class GridCell:
    """Represents a single grid cell/plate"""

    def __init__(
        self,
        berth: str,
        area: str,
        grid_number: str,
        row: int,
        col: int,
        settlement: float = 0.0,
        water: float = 0.0,
        soil: str = "",
        doc: str = "",
        has_data: bool = True,
        site_prefix: str = SITE_PREFIX,
        region_code: Optional[str] = None,
    ):
        self.berth = berth
        self.area = area
        self.grid_number = grid_number
        self.row = row
        self.col = col
        self.settlement = settlement
        self.water = water
        self.soil = soil
        self.doc = doc
        self.has_data = has_data
        self.site_prefix = site_prefix
        self.region_code = region_code or area_to_region_code(area)
        self.full_name = f"{self.site_prefix}-{self.region_code}-SM-{grid_number}"
        self.status: Optional[str] = None
        self.latest_settlement: Optional[float] = None
        self.latest_gl: Optional[float] = None
        self.latest_gwl: Optional[float] = None
        self.surcharge_pressure: Optional[float] = None
        self.weekly_rate: Optional[float] = None
        self.asaoka_doc: Optional[float] = None
        self.surcharge_complete: Optional[str] = None
        self.timestamp: Optional[str] = None
        self.record: Dict[str, Any] = {}
        self.original_label = f"{berth}-{area}-{grid_number}"
        
    @property
    def category(self) -> str:
        """Calculate category based on settlement and water thresholds"""
        if not self.has_data:
            return "NA"

        if self.status:
            normalized = self.status.strip().upper()
            if normalized in STATUS_STYLES:
                return normalized
            return "NA"

        # Fallback classification using settlement/water when explicit status is absent.
        settlement_issue = self.settlement <= 40
        water_issue = self.water >= 110

        if not settlement_issue and not water_issue:
            return "O"
        if settlement_issue and not water_issue:
            return "S"
        if water_issue and not settlement_issue:
            return "P"
        if settlement_issue and water_issue:
            return "X"
        return "NA"
    
    @property
    def color(self) -> str:
        """Get color based on category"""
        style = STATUS_STYLES.get(self.category)
        return style["color"] if style else "#9E9E9E"
    
    @property
    def tooltip_text(self) -> str:
        """Generate hover tooltip text"""
        details: List[str] = [f"<b>{self.full_name}</b>"]
        details.append(f"Berth/Area: {self.berth}-{self.area}")
        details.append(f"Grid Label: {self.original_label}")

        status_code: Optional[str] = None
        if self.status and isinstance(self.status, str):
            candidate = self.status.strip().upper()
            if candidate in STATUS_STYLES:
                status_code = candidate

        if not status_code:
            status_code = self.category

        status_text = status_label(status_code)
        if self.status and status_code == self.status.strip().upper():
            details.append(f"Status: {status_text} ({status_code})")
        else:
            details.append(f"Status: {status_text}")

        if not self.has_data:
            details.append("Data: No measurement records available.")
        else:
            if self.timestamp:
                details.append(f"Latest Reading: {self.timestamp}")

            numeric_fields: List[Tuple[str, Optional[float], Optional[int]]] = [
                ("Latest Settlement (mm)", self.latest_settlement, 3),
                ("7-day Rate (mm)", self.weekly_rate, 3),
                ("Surcharge Pressure (kPa)", self.surcharge_pressure, 2),
                ("Asaoka DOC (%)", self.asaoka_doc, 1),
                ("Latest GL", self.latest_gl, 3),
                ("Latest GWL", self.latest_gwl, 3),
            ]

            for label, value, precision in numeric_fields:
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    if precision is not None:
                        formatted_value = f"{value:.{precision}f}"
                    else:
                        formatted_value = f"{value}"
                else:
                    formatted_value = str(value)
                details.append(f"{label}: {formatted_value}")

            if self.surcharge_complete:
                details.append(f"Surcharge Complete: {self.surcharge_complete}")
            if self.soil:
                details.append(f"Soil Type: {self.soil}")

        details.append(f"Position: Row {self.row}, Col {self.col}")
        return "<br>".join(details)


class BerthArea:
    """Represents a berth area with multiple grids"""
    def __init__(self, berth: str, area: str, rows: int, cols: int, region_code: Optional[str] = None):
        self.berth = berth
        self.area = area
        self.rows = rows
        self.cols = cols
        self.region_code = region_code or area_to_region_code(area)
        self.grids: Dict[str, GridCell] = {}
        
    def add_grid(self, grid: GridCell):
        """Add a grid to this area"""
        self.grids[grid.grid_number] = grid


class Berth:
    """Represents a berth with multiple areas"""
    def __init__(self, name: str, x_offset: int = 0, y_offset: int = 0):
        self.name = name
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.areas: Dict[str, BerthArea] = {}
        self.area_order: List[str] = []

    def add_area(self, area: BerthArea):
        """Add an area to this berth"""
        self.areas[area.area] = area
        self.area_order.append(area.area)

    @property
    def column_count(self) -> int:
        """Total columns for this berth"""
        return max((area.cols for area in self.areas.values()), default=0)

    @property
    def total_rows(self) -> int:
        """Total rows across all areas (stacked vertically)"""
        return sum(area.rows for area in self.areas.values())

    @property
    def start_row_for_area(self) -> Dict[str, int]:
        """Get starting row for each area (0-indexed)"""
        start_rows = {}
        current_row = 0
        for area_name in self.area_order:
            start_rows[area_name] = current_row
            current_row += self.areas[area_name].rows
        return start_rows

    @property
    def max_y(self) -> int:
        """Bottom-most row coordinate considering offset"""
        return self.y_offset + self.total_rows

    @property
    def max_x(self) -> int:
        """Right-most column coordinate considering offset"""
        return self.x_offset + self.column_count


class ConstructionMap:
    """Represents the entire construction site map"""
    def __init__(self):
        self.berths: Dict[str, Berth] = {}
        self.berth_order: List[str] = []
    
    def add_berth(self, berth: Berth):
        """Add a berth to the map"""
        self.berths[berth.name] = berth
        self.berth_order.append(berth.name)
    
    @property
    def max_rows(self) -> int:
        """Maximum rows across all berths including offsets"""
        return max((berth.max_y for berth in self.berths.values()), default=0)

    @property
    def max_columns(self) -> int:
        """Maximum columns across all berths including offsets"""
        return max((berth.max_x for berth in self.berths.values()), default=0)

    @property
    def min_x(self) -> int:
        return min((berth.x_offset for berth in self.berths.values()), default=0)

    @property
    def min_y(self) -> int:
        return min((berth.y_offset for berth in self.berths.values()), default=0)

    def iter_grids(self) -> Iterable[GridCell]:
        for berth in self.berths.values():
            for area in berth.areas.values():
                for grid in area.grids.values():
                    yield grid


# ============================================================================
# 2. MAP BUILDING UTILITIES
# ============================================================================

def create_sample_map_structure() -> Dict[str, Dict[str, Any]]:
    """Create sample map structure with explicit berth offsets and area sizes."""
    return {
        "B9": {
            "offset": {"x": 0, "y": 0},
            "areas": {
                "IF2": {"rows": 3, "cols": 7},
                "XXIa": {"rows": 3, "cols": 7},
                "XXIb": {"rows": 6, "cols": 7},
                "XXIc": {"rows": 3, "cols": 7},
                "XXId": {"rows": 3, "cols": 7},
            },
        },
        "B8": {
            "offset": {"x": 7, "y": 4},
            "areas": {
                "XXa": {"rows": 2, "cols": 10},
                "XXb": {"rows": 6, "cols": 10},
                "XXc": {"rows": 3, "cols": 10},
                "XXd": {"rows": 3, "cols": 10},
            },
        },
        "B7": {
            "offset": {"x": 17, "y": 4},
            "areas": {
                "Ia": {"rows": 8, "cols": 10},
                "Ib": {"rows": 3, "cols": 10},
                "XVIa": {"rows": 3, "cols": 10},
            },
        },
        "B6": {
            "offset": {"x": 27, "y": 4},
            "areas": {
                "IIa": {"rows": 8, "cols": 10},
                "IIb": {"rows": 3, "cols": 10},
                "XVa": {"rows": 3, "cols": 10},
            },
        },
        "B5": {
            "offset": {"x": 37, "y": 4},
            "areas": {
                "IIIa": {"rows": 8, "cols": 10},
                "IIIb": {"rows": 3, "cols": 10},
                "XIVa": {"rows": 3, "cols": 10},
            },
        },
        "B4": {
            "offset": {"x": 47, "y": 4},
            "areas": {
                "IVa": {"rows": 8, "cols": 10},
                "IVb": {"rows": 3, "cols": 10},
                "XIIIa": {"rows": 3, "cols": 10},
            },
        },
        "B3": {
            "offset": {"x": 57, "y": 4},
            "areas": {
                "Va": {"rows": 8, "cols": 10},
                "Vb": {"rows": 3, "cols": 10},
                "XIIa": {"rows": 3, "cols": 10},
            },
        },
        "B2": {
            "offset": {"x": 67, "y": 4},
            "areas": {
                "VIa": {"rows": 8, "cols": 10},
                "VIb": {"rows": 3, "cols": 10},
                "XIa": {"rows": 3, "cols": 10},
            },
        },
        "B1": {
            "offset": {"x": 77, "y": 4},
            "areas": {
                "VIIa": {"rows": 8, "cols": 10},
                "VIIb": {"rows": 3, "cols": 10},
                "Xa": {"rows": 3, "cols": 10},
            },
        },
        "A24": {
            "offset": {"x": 87, "y": 4},
            "areas": {
                "VIIIa": {"rows": 8, "cols": 10},
                "VIIIb": {"rows": 3, "cols": 10},
                "IXa": {"rows": 3, "cols": 10},
            },
        },
        "A22": {
            "offset": {"x": 87, "y": 18},
            "areas": {
                "IXb": {"rows": 8, "cols": 10},
            },
        },
        "A21": {
            "offset": {"x": 77, "y": 18},
            "areas": {
                "Xb": {"rows": 8, "cols": 10},
            },
        },
        "A20": {
            "offset": {"x": 67, "y": 18},
            "areas": {
                "XIb": {"rows": 8, "cols": 10},
            },
        },
        "A19": {
            "offset": {"x": 57, "y": 18},
            "areas": {
                "XIIb": {"rows": 8, "cols": 10},
            },
        },
        "A18": {
            "offset": {"x": 47, "y": 18},
            "areas": {
                "XIIIb": {"rows": 8, "cols": 10},
            },
        },
        "A17": {
            "offset": {"x": 37, "y": 18},
            "areas": {
                "XIVb": {"rows": 8, "cols": 10},
            },
        },
        "A16": {
            "offset": {"x": 27, "y": 18},
            "areas": {
                "XVb": {"rows": 8, "cols": 10},
            },
        },
        "A15": {
            "offset": {"x": 17, "y": 18},
            "areas": {
                "XVIb": {"rows": 8, "cols": 10},
            },
        },
        "A14": {
            "offset": {"x": 7, "y": 18},
            "areas": {
                "XVIIb": {"rows": 8, "cols": 10},
            },
        },
        "A13": {
            "offset": {"x": 0, "y": 18},
            "areas": {
                "XVIIb": {"rows": 15, "cols": 7},
            },
        },
    }


def _normalize_area_dimensions(area_def: Any) -> Tuple[int, int]:
    """Extract rows/cols from layout definitions."""
    if isinstance(area_def, dict):
        rows = area_def.get("rows") or area_def.get("length") or area_def.get("height")
        cols = area_def.get("cols") or area_def.get("columns") or area_def.get("width")
        if rows is None or cols is None:
            raise ValueError("Area definitions must include rows/cols (or width/length).")
        return int(rows), int(cols)
    if isinstance(area_def, (list, tuple)) and len(area_def) >= 2:
        return int(area_def[0]), int(area_def[1])
    if isinstance(area_def, int):
        return int(area_def), int(area_def)
    raise ValueError("Unsupported area definition format.")


def build_construction_map(map_structure: Dict[str, Dict[str, Any]]) -> ConstructionMap:
    """Create a ConstructionMap instance from a layout definition."""
    construction_map = ConstructionMap()

    for berth_name, berth_data in map_structure.items():
        offset_info = berth_data.get("offset", {})
        berth = Berth(
            name=berth_name,
            x_offset=int(offset_info.get("x", 0)),
            y_offset=int(offset_info.get("y", 0)),
        )

        areas = berth_data.get("areas", {})
        if not areas:
            continue

        area_order = berth_data.get("area_order") or list(areas.keys())
        for area_name in area_order:
            area_def = areas[area_name]
            rows, cols = _normalize_area_dimensions(area_def)
            region_code = area_to_region_code(area_name)
            area = BerthArea(berth_name, area_name, rows, cols, region_code=region_code)

            plate_index = 1
            for row in range(1, rows + 1):
                for col in range(1, cols + 1):
                    grid_number = f"{plate_index:02d}"
                    plate_index += 1
                    area.add_grid(
                        GridCell(
                            berth=berth_name,
                            area=area_name,
                            grid_number=grid_number,
                            row=row,
                            col=col,
                            has_data=False,
                            site_prefix=SITE_PREFIX,
                            region_code=region_code,
                        )
                    )

            berth.add_area(area)

        construction_map.add_berth(berth)

    return construction_map


def generate_sample_data(map_structure: Dict[str, Dict[str, Any]]) -> ConstructionMap:
    """Generate sample grid data for testing using layout."""
    construction_map = build_construction_map(map_structure)
    soil_types = ["Clay", "Sand", "Silt", "Loam"]

    for grid in construction_map.iter_grids():
        settlement = random.uniform(20, 60)
        water = random.uniform(90, 130)
        doc_date = (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")
        grid.settlement = round(settlement, 1)
        grid.water = round(water, 1)
        grid.soil = random.choice(soil_types)
        grid.doc = doc_date
        grid.has_data = True

    return construction_map


def extract_plate_records(construction_map: ConstructionMap) -> List[Dict[str, Any]]:
    """Convert construction map data into plate record dictionaries."""
    records: List[Dict[str, Any]] = []
    for grid in construction_map.iter_grids():
        if not grid.has_data:
            continue
        records.append(
            {
                "full_name": grid.full_name,
                "berth": grid.berth,
                "area": grid.area,
                "grid_number": grid.grid_number,
                "settlement": grid.settlement,
                "water": grid.water,
                "soil": grid.soil,
                "doc": grid.doc,
            }
        )
    return records


def summarize_grid_coverage(grids: Iterable[GridCell]) -> Dict[str, Any]:
    """Collect coverage and status statistics for a grid collection."""
    grid_list = list(grids)
    grids_with_data = [grid for grid in grid_list if grid.has_data]
    status_counts = Counter(grid.category for grid in grids_with_data)

    total = len(grid_list)
    with_data = len(grids_with_data)
    missing = total - with_data
    coverage_pct = (with_data / total * 100.0) if total else 0.0

    status_counts_struct = {status: status_counts.get(status, 0) for status in STATUS_ORDER}
    na_count = status_counts.get("NA", 0)

    return {
        "total": total,
        "with_data": with_data,
        "missing": missing,
        "coverage_pct": coverage_pct,
        "status_counts": {**status_counts_struct, "NA": na_count},
        "status_counts_with_na": {
            **status_counts_struct,
            "NA": na_count + missing,
        },
    }


def resolve_requested_grid_names(
    grids: Iterable[GridCell],
    highlight_grids: Optional[Set[str]],
    highlight_areas: Optional[Set[str]],
    highlight_berths: Optional[Set[str]],
) -> Set[str]:
    """Expand requested items into a concrete set of grid full-name identifiers."""
    requested: Set[str] = set(highlight_grids or set())
    area_set = set(highlight_areas or set())
    berth_set = set(highlight_berths or set())

    if not area_set and not berth_set:
        return requested

    for grid in grids:
        area_key = f"{grid.berth}-{grid.area}"
        if area_key in area_set or grid.berth in berth_set:
            requested.add(grid.full_name)

    return requested


def _resolve_plate_identifier(record: Dict[str, Any]) -> Optional[str]:
    """Try to resolve a unique grid identifier from a plate record."""
    direct_keys = ["PointID", "point_id", "full_name", "grid", "plate_id", "id"]
    for key in direct_keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    region_code = record.get("region_code") or record.get("RegionCode")
    grid_num = record.get("grid_number") or record.get("plate") or record.get("grid")
    if region_code and grid_num is not None:
        return f"{SITE_PREFIX}-{str(region_code).strip()}-SM-{str(grid_num).zfill(2)}"

    berth = record.get("berth")
    area = record.get("area") or record.get("berth_area")
    if berth and area and grid_num is not None:
        return f"{str(berth).upper()}-{str(area).title()}-{str(grid_num).zfill(2)}"

    if area and grid_num is not None:
        region = area_to_region_code(str(area))
        return f"{SITE_PREFIX}-{region}-SM-{str(grid_num).zfill(2)}"

    return None


def apply_plate_records(
    construction_map: ConstructionMap, plate_records: Iterable[Dict[str, Any]]
) -> List[GridCell]:
    """Apply measurement data onto an existing construction map."""
    index = {grid.full_name: grid for grid in construction_map.iter_grids()}
    updated: List[GridCell] = []

    for record in plate_records:
        identifier = _resolve_plate_identifier(record)
        if not identifier:
            continue

        grid = index.get(identifier)
        if not grid:
            continue

        grid.record = record
        grid.status = "NA"

        status_value = record.get("Status") or record.get("status")
        if isinstance(status_value, str) and status_value.strip():
            normalized_status = status_value.strip().upper()
            if normalized_status in STATUS_STYLES:
                grid.status = normalized_status
            elif normalized_status in {"NULL", "NONE", "N/A", "NA"}:
                grid.status = "NA"

        settlement = record.get("Latest_Settlement", record.get("settlement"))
        settlement_value = _to_float(settlement)
        if settlement_value is not None:
            grid.latest_settlement = settlement_value
            grid.settlement = settlement_value

        weekly = record.get("7day_rate") or record.get("weekly_rate")
        weekly_value = _to_float(weekly)
        if weekly_value is not None:
            grid.weekly_rate = weekly_value

        surcharge_pressure = record.get("Surcharge_Pressure") or record.get("surcharge_pressure")
        surcharge_value = _to_float(surcharge_pressure)
        if surcharge_value is not None:
            grid.surcharge_pressure = surcharge_value

        asaoka = record.get("Asaoka_DOC") or record.get("asaoka_doc")
        asaoka_value = _to_float(asaoka)
        if asaoka_value is not None:
            grid.asaoka_doc = asaoka_value

        latest_gl = record.get("Latest_GL") or record.get("latest_gl")
        latest_gl_value = _to_float(latest_gl)
        if latest_gl_value is not None:
            grid.latest_gl = latest_gl_value

        latest_gwl = record.get("Latest_GWL") or record.get("latest_gwl")
        latest_gwl_value = _to_float(latest_gwl)
        if latest_gwl_value is not None:
            grid.latest_gwl = latest_gwl_value
            grid.water = latest_gwl_value

        if latest_gl_value is not None and latest_gwl_value is None:
            grid.water = latest_gl_value

        doc = record.get("Datetime") or record.get("datetime") or record.get("timestamp") or record.get("doc")
        if doc is not None:
            grid.doc = str(doc)
            grid.timestamp = str(doc)

        surcharge_complete = record.get("Surcharge_Complete_date") or record.get("surcharge_complete_date")
        if surcharge_complete is not None:
            grid.surcharge_complete = str(surcharge_complete)

        soil = record.get("soil") or record.get("soil_type")
        if soil is not None:
            grid.soil = str(soil)

        grid.has_data = True
        updated.append(grid)

    return updated


# ============================================================================
# 3. MAP VISUALIZATION
# ============================================================================

def create_continuous_map_figure(
    construction_map: ConstructionMap,
    highlight_grids: Optional[Set[str]] = None,
    highlight_areas: Optional[Set[str]] = None,
    highlight_berths: Optional[Set[str]] = None,
) -> go.Figure:
    """Render the construction map with optional highlighting."""
    highlight_grids = highlight_grids or set()
    highlight_areas = highlight_areas or set()
    highlight_berths = highlight_berths or set()
    has_highlights = bool(highlight_grids or highlight_areas or highlight_berths)

    min_x = construction_map.min_x
    min_y = construction_map.min_y
    max_x = construction_map.max_columns
    max_y = construction_map.max_rows

    x_shift = -min_x
    y_shift = -min_y
    plot_min_x = x_shift + min_x
    plot_max_x = x_shift + max_x
    plot_min_y = y_shift + min_y
    plot_max_y = y_shift + max_y

    fig = go.Figure()
    annotations = []
    shapes = []

    for berth_name in construction_map.berth_order:
        berth = construction_map.berths[berth_name]
        is_berth_highlighted = berth_name in highlight_berths
        area_start_rows = berth.start_row_for_area

        # Calculate berth boundaries
        berth_left = x_shift + berth.x_offset
        berth_right = berth_left + berth.column_count
        berth_top = y_shift + berth.y_offset
        berth_bottom = berth_top + berth.total_rows

        for area_name in berth.area_order:
            area = berth.areas[area_name]
            row_offset = area_start_rows[area_name]
            area_full_name = f"{berth_name}-{area_name}"
            is_area_highlighted = area_full_name in highlight_areas or is_berth_highlighted

            # Calculate area boundaries
            area_left = x_shift + berth.x_offset
            area_right = area_left + area.cols
            area_top = y_shift + berth.y_offset + row_offset
            area_bottom = area_top + area.rows

            for grid in area.grids.values():
                grid_highlighted = grid.full_name in highlight_grids

                # Grid opacity logic
                if grid.has_data:
                    fillcolor = grid.color
                    if has_highlights:
                        emphasize = grid_highlighted or is_area_highlighted or is_berth_highlighted
                        opacity_value = 1.0 if emphasize else 0.2
                    else:
                        opacity_value = 1.0
                    line_color = "#2f3640" if opacity_value == 1.0 else "#999999"
                else:
                    fillcolor = "#CFD8DC"
                    opacity_value = 0.15 if has_highlights else 0.6
                    line_color = "#90A4AE"

                grid_left = area_left + (grid.col - 1)
                grid_right = grid_left + 1
                grid_top = area_top + (grid.row - 1)
                grid_bottom = grid_top + 1

                # ALWAYS create hover for grids with data
                if grid.has_data:
                    # Get status for hover
                    status_code: Optional[str] = None
                    if grid.status and isinstance(grid.status, str):
                        candidate = grid.status.strip().upper()
                        if candidate in STATUS_STYLES:
                            status_code = candidate

                    if not status_code:
                        status_code = grid.category

                    status_display = status_label(status_code)
                    
                    # Create hover text
                    hover_text = (
                        f"<b>{grid.full_name}</b><br>"
                        f"Status: {status_display}<br>"
                        f"7-day Rate: {_format_value(grid.weekly_rate, 3, '—')} mm<br>"
                        f"Asaoka DOC: {_format_value(grid.asaoka_doc, 1, '—')}%<br>"
                        f"Surcharge Pressure: {_format_value(grid.surcharge_pressure, 2, '—')} kPa"
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=[grid_left, grid_right, grid_right, grid_left, grid_left],
                        y=[grid_top, grid_top, grid_bottom, grid_bottom, grid_top],
                        mode="lines",
                        fill="toself",
                        fillcolor=fillcolor,
                        line=dict(color=line_color, width=1),
                        opacity=opacity_value,
                        hoverinfo="text",
                        text=hover_text,
                        hoveron="fills",
                        showlegend=False,
                        name=grid.full_name,
                    ))
                else:
                    # Grids without data - no hover
                    fig.add_trace(go.Scatter(
                        x=[grid_left, grid_right, grid_right, grid_left, grid_left],
                        y=[grid_top, grid_top, grid_bottom, grid_bottom, grid_top],
                        mode="lines",
                        fill="toself",
                        fillcolor=fillcolor,
                        line=dict(color=line_color, width=1),
                        opacity=opacity_value,
                        hoverinfo="skip",
                        showlegend=False,
                        name=grid.full_name,
                    ))

            # Area rectangle and label (AFTER drawing all grids in this area)
            area_label_opacity = 0.6 if (not has_highlights or is_area_highlighted) else 0.25
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=area_left,
                    y0=area_top,
                    x1=area_right,
                    y1=area_bottom,
                    line=dict(color=f"rgba(33, 33, 33, {area_label_opacity})", width=1.5),
                    fillcolor="rgba(0,0,0,0)",
                    layer="below",
                )
            )
            annotations.append(
                dict(
                    x=(area_left + area_right) / 2,
                    y=(area_top + area_bottom) / 2,
                    text=area_name,
                    showarrow=False,
                    font=dict(size=10, color="rgba(55,65,81,0.8)"),
                    bgcolor=f"rgba(255,255,255,{area_label_opacity * 0.5})",
                    bordercolor=f"rgba(31,41,51,{area_label_opacity * 0.4})",
                    borderwidth=0.5,
                    borderpad=2,
                )
            )

        # Berth rectangle and label (AFTER drawing all areas in this berth)
        berth_label_opacity = 0.65 if (not has_highlights or is_berth_highlighted) else 0.3
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=berth_left,
                y0=berth_top,
                x1=berth_right,
                y1=berth_bottom,
                line=dict(color=f"rgba(0,0,0,{berth_label_opacity})", width=2),
                fillcolor="rgba(0,0,0,0)",
                layer="below",
            )
        )
        annotations.append(
            dict(
                x=(berth_left + berth_right) / 2,
                y=berth_top - 0.8,
                text=berth_name,
                showarrow=False,
                font=dict(size=8, color="rgba(31,41,55,0.75)", family="Arial"),
                bgcolor=f"rgba(248,249,250,{berth_label_opacity * 0.5})",
                bordercolor=f"rgba(55,65,81,{berth_label_opacity * 0.4})",
                borderwidth=1,
                borderpad=4,
            )
        )

    fig.update_layout(
        title=dict(text="<b>CONSTRUCTION SITE MAP</b>", x=0.5, font=dict(size=24, family="Arial Black")),
        showlegend=True,
        height=760,
        width=max(960, int((plot_max_x - plot_min_x + 6) * 45)),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#f8f9fa",
        hovermode="x unified",
        dragmode="pan",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1,
            range=[plot_min_x - 1, plot_max_x + 1],
            constrain="domain",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[plot_min_y - 1, plot_max_y + 1],
            autorange="reversed",
            constrain="domain",
        ),
        shapes=shapes,
        annotations=annotations,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="black",
            borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=80, b=20),
    )

    categories = [
        (code, status_label(code), STATUS_STYLES.get(code, {}).get("color", "#9E9E9E"))
        for code in ["O", "S", "P", "X", "NA"]
    ]

    for code, desc, color in categories:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=14, color=color, line=dict(width=1, color="black")),
                name=f"<b>{code}</b>: {desc}",
                showlegend=True,
            )
        )

    if has_highlights:
        annotations.append(
            dict(
                x=plot_min_x + (plot_max_x - plot_min_x) / 2,
                y=y_shift - 1,
                text="<i>Non-highlighted regions are dimmed.</i>",
                showarrow=False,
                font=dict(size=10, color="gray"),
                xref="x",
                yref="y",
            )
        )
        fig.update_layout(annotations=annotations)

    return fig

def create_map_response(
    query: str,
    map_layout: Optional[Dict[str, Dict[str, Any]]] = None,
    plate_records: Optional[Iterable[Dict[str, Any]]] = None,
    highlight_grids: Optional[Set[str]] = None,
    highlight_areas: Optional[Set[str]] = None,
    highlight_berths: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Create a map response for chat integration."""
    layout = map_layout or create_sample_map_structure()
    construction_map = build_construction_map(layout)

    plate_records_list: List[Dict[str, Any]] = list(plate_records or [])
    applied_grids: List[GridCell] = []
    if plate_records_list:
        applied_grids = apply_plate_records(construction_map, plate_records_list)

    all_grids = list(construction_map.iter_grids())
    grids_with_data = [grid for grid in all_grids if grid.has_data]
    data_grid_names = {grid.full_name for grid in grids_with_data}
    coverage_stats = summarize_grid_coverage(all_grids)

    final_highlight_grids: Optional[Set[str]] = None
    if highlight_grids:
        final_highlight_grids = set(highlight_grids) & data_grid_names
    elif not highlight_grids and not highlight_areas and not highlight_berths and plate_records_list:
        final_highlight_grids = data_grid_names

    fig = create_continuous_map_figure(
        construction_map,
        highlight_grids=final_highlight_grids,
        highlight_areas=highlight_areas,
        highlight_berths=highlight_berths,
    )

    response = {
        "type": "map",
        "query": query,
        "response_message": (
            "Showing construction site map with plate data" if plate_records_list else "Showing empty construction site map"
        ),
        "figure_json": fig.to_json(),
        "highlighted_items": {
            "grids": sorted(final_highlight_grids) if final_highlight_grids else [],
            "areas": sorted(highlight_areas) if highlight_areas else [],
            "berths": sorted(highlight_berths) if highlight_berths else [],
        },
        "metadata": {
            "total_berths": len(construction_map.berths),
            "total_grids": coverage_stats["total"],
            "grids_with_data": coverage_stats["with_data"],
            "grids_without_data": coverage_stats["missing"],
            "data_coverage_percent": round(coverage_stats["coverage_pct"], 1),
            "status_counts": coverage_stats["status_counts"],
            "timestamp": datetime.now().isoformat(),
        },
    }

    if plate_records_list:
        response["metadata"].update(
            {
                "plates_returned": len(plate_records_list),
                "plates_applied": len(applied_grids),
            }
        )

    return response
