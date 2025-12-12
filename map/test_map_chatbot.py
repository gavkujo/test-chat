"""Streamlit sandbox to exercise the production prompt → map pipeline."""

import json
from typing import Dict, List, Optional

import streamlit as st
from plotly.io import from_json

from chatbot_map_service import (
    API_BASE_URL,
    LookupExecutionError,
    QueryParsingError,
    build_map_for_prompt,
    parse_query_intent,
)


def render_intent_preview(prompt: str) -> None:
    """Show the deterministic lookup derived from the current prompt."""
    if not prompt.strip():
        st.info("Enter a prompt to preview the lookup request.")
        return

    try:
        intent = parse_query_intent(prompt)
    except QueryParsingError as exc:
        st.warning(f"Parsing issue: {exc}")
        return

    st.json(intent.to_dict(), expanded=False)


def parse_override_payload(raw_json: str) -> List[Dict[str, object]]:
    if not raw_json.strip():
        raise LookupExecutionError("Provide a JSON list of plate records when using the override mode.")
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise LookupExecutionError(f"Plate data JSON is invalid: {exc}") from exc
    if not isinstance(payload, list):
        raise LookupExecutionError("Plate data override must be a JSON list of record objects.")
    return payload


def main() -> None:
    st.set_page_config(page_title="Chatbot Map Sandbox", page_icon="🗺️", layout="wide")
    st.title("Chatbot Map Production Sandbox")
    st.write("Simulate the deterministic prompt parsing, GeoChat lookup, and map rendering flow.")

    with st.sidebar:
        st.header("Data source")
        data_source = st.radio("Plate data source", ("GeoChat API", "Paste JSON"), index=0)
        st.caption(f"GeoChat endpoint base: {API_BASE_URL}")

    prompt = st.text_area(
        "Chatbot prompt",
        height=120,
        placeholder="e.g. show me all the problematic plates in berths A1 and B2",
    )

    with st.expander("Parsed intent", expanded=False):
        render_intent_preview(prompt)

    records_override: Optional[List[Dict[str, object]]] = None
    manual_json = ""
    if data_source == "Paste JSON":
        manual_json = st.text_area(
            "Plate data JSON override",
            height=260,
            placeholder="[ {\"PointID\": \"F3-R07a-SM-24\", ...} ]",
        )

    if st.button("Generate Map", type="primary"):
        if not prompt.strip():
            st.error("Enter a prompt before generating the map.")
            return

        if data_source == "Paste JSON":
            try:
                records_override = parse_override_payload(manual_json)
            except LookupExecutionError as exc:
                st.error(f"Override payload error: {exc}")
                return

        try:
            result = build_map_for_prompt(
                prompt=prompt,
                map_layout=None,
                records_override=records_override,
            )
        except QueryParsingError as exc:
            st.error(f"Could not interpret the prompt: {exc}")
            return
        except LookupExecutionError as exc:
            st.error(f"Lookup failed: {exc}")
            return
        except Exception as exc:  # pragma: no cover - surface unexpected issues
            st.exception(exc)
            return

        st.success("Map generated successfully.")
        st.subheader("Response message")
        st.write(result.response["response_message"])

        figure = from_json(result.response["figure_json"])
        st.plotly_chart(figure, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Metadata")
            st.json(result.response["metadata"], expanded=False)
        with col2:
            st.caption("Highlighted items")
            st.json(result.response["highlighted_items"], expanded=False)

        with st.expander("Filtered plate records", expanded=False):
            if result.records:
                st.dataframe(result.records)
            else:
                st.write("No records matched the active filter.")

        with st.expander("Raw lookup records", expanded=False):
            if result.raw_records:
                st.dataframe(result.raw_records)
            else:
                st.write("The lookup returned no records.")

        with st.expander("Debug info", expanded=False):
            st.json(result.to_dict(), expanded=False)
            if result.api_url:
                st.code(result.api_url, language="text")


if __name__ == "__main__":
    main()
