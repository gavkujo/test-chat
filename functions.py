from helpers.reporter import reporter_Asaoka
from helpers.asaoka import Asaoka_data
from helpers.settlement_data import reporter_Settlement
from helpers.datasources import SM_overview


def Func1(id, SCD, ASD, max_date):
    """Return structured Asaoka data for downstream insight generation."""

    try:
        data = Asaoka_data(id, SCD, ASD, max_date=max_date, asaoka_days=7, period=0, n=4)
        return {
            "status": "ok",
            "type": "Asaoka_data",
            "intent": "plates_data",
            "data": data,
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        return {
            "status": "error",
            "type": "Asaoka_data",
            "error": str(exc),
        }


def Func2(ids, SCD, ASD, max_date):
    """Generate Asaoka PDF and signal availability to the UI."""

    try:
        with open("static/asaoka_report.pdf", "wb") as f:
            data_bytes = reporter_Asaoka(ids, SCD, ASD, max_date, n=4, asaoka_days=7, dtick=500)
            if not data_bytes:
                raise ValueError("No bytes returned for Asaoka report")
            f.write(data_bytes)
        return {
            "status": "ok",
            "type": "reporter_Asaoka",
            "intent": "resource",
            "path": "static/asaoka_report.pdf",
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        return {
            "status": "error",
            "type": "reporter_Asaoka",
            "error": str(exc),
        }


def Func3(ids, max_date):
    """Create combined settlement plot and expose file path."""

    try:
        with open("static/Combined_settlement_plot.pdf", "wb") as f:
            data_bytes = reporter_Settlement(ids, max_date)
            if not data_bytes:
                raise ValueError("No bytes returned for settlement plot")
            f.write(data_bytes)
        return {
            "status": "ok",
            "type": "plot_combi_S",
            "intent": "resource",
            "path": "static/Combined_settlement_plot.pdf",
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        return {
            "status": "error",
            "type": "plot_combi_S",
            "error": str(exc),
        }


def Func4(ids):
    try:
        data = SM_overview(ids)
        return {
            "status": "ok",
            "type": "SM_overview",
            "intent": "plates_data",
            "data": data,
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        return {
            "status": "error",
            "type": "SM_overview",
            "error": str(exc),
        }


