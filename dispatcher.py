# dispatcher.py

"""Thin helpers that bridge parsed parameters to concrete functions."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import functions
from data.parser_test import parse_and_build


class Dispatcher:
    """Provide pure parsing and execution utilities for the Streamlit app."""

    def __init__(self, func_map: Optional[Dict[str, Callable[..., Any]]] = None) -> None:
        self.func_map = func_map or {
            "Asaoka_data": functions.Func1,
            "reporter_Asaoka": functions.Func2,
            "plot_combi_S": functions.Func3,
            "SM_overview": functions.Func4,
        }

    def pure_parse(self, raw_query: str, func_name: str) -> Dict[str, Any]:
        """Return structured parameters or raise ``MissingSlot`` for the UI to handle."""

        return parse_and_build(raw_query, func_name)

    def run_function(self, func_name: str, params: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Execute a mapped function when we hold a complete parameter set."""

        if not params:
            return None
        func = self.func_map.get(func_name)
        if func is None:
            raise KeyError(f"Unsupported function requested: {func_name}")
        return func(**params)