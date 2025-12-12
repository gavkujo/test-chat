#!/usr/bin/env python3
"""Logic-driven response generator facade.

This module replaces the former Ollama-backed router with a fully
rule/templates/statistics based implementation defined in
``core.response_generator``. The public API intentionally mirrors the old
``LLMRouter`` interface so the rest of the application can remain unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.response_generator import ResponseGenerator, ResponsePayload


class LLMRouter(ResponseGenerator):
    """Backwards compatible shim for the updated response generator."""

    def handle_user(
        self,
        user_input: str,
        func_name: Optional[str] = None,
        classifier_data: Optional[Dict[str, Any]] = None,
        func_output: Optional[Any] = None,
        stream: bool = False,
    ) -> ResponsePayload:
        return super().handle_user(
            user_input=user_input,
            func_name=func_name,
            classifier_data=classifier_data,
            func_output=func_output,
            stream=stream,
        )
