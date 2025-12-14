# app.py

'''
Boskalis GeoChat Streamlit Frontend

This module implements the interactive Streamlit interface for the Boskalis GeoChat NL2Func pipeline.

Features:
1. Chat UI with role-based message rendering for user and assistant.
2. Slot-filling support with guided prompts for missing function parameters.
3. Function clash detection and resolution workflow.
4. PDF generation download buttons for settlement resources.
5. Automatic map rendering when the user explicitly requests a map view.
6. Session state management for chat history, slot state, dispatcher, classifier, and LLM router.
7. Debug logging for input handling, slot-filling, clash resolution, and function execution.
'''

import os
from typing import Any, Dict, Iterable, Optional

import plotly.io as pio
import streamlit as st  # type: ignore

from core.response_generator import ChartPayload, ResponsePayload
from dispatcher import Dispatcher
from llm_main import LLMRouter
from main import Classifier, choose_function
from data.parser_test import FunctionClash, MissingSlot
from map.chatbot_map_service import (
    LookupExecutionError,
    QueryParsingError,
    build_map_for_prompt,
    lookup_prompt_records,
)

@st.cache_resource
def load_classifier():
    return Classifier()

FUNCTION_DESCRIPTIONS = {
    "reporter_Asaoka": "Asaoka report generator",
    "plot_combi_S": "Settlement plotter", 
    "Asaoka_data": "Asaoka assessment information retriever",
    "SM_overview": "Settlement overview processor"
}

def get_function_description(func_name):
    """Get user-friendly description for function"""
    text = FUNCTION_DESCRIPTIONS.get(func_name)
    print("[DEBUG] ", text)
    return text

MAP_KEYWORDS = (
    " map",
    "map ",
    " map.",
    "map.",
    "map?",
    "map!",
    "map!)",
    "show on a map",
    "map this",
    "map it",
    "mapping",
)

GEOCHAT_REDIRECT_FUNCS = {"Asaoka_data", "SM_overview"}


def wants_map(prompt: str) -> bool:
    lowered = prompt.lower()
    if "map" not in lowered:
        return False
    return any(keyword in lowered for keyword in MAP_KEYWORDS) or lowered.strip().startswith("map")


def attach_map_payload(prompt: str, payload: ResponsePayload) -> bool:
    existing = payload.metadata.get("map")
    if existing:
        return existing.get("status") == "ok"
    try:
        result = build_map_for_prompt(prompt)
    except QueryParsingError as exc:
        payload.metadata["map"] = {"status": "parse_error", "error": str(exc)}
        return False
    except LookupExecutionError as exc:
        payload.metadata["map"] = {"status": "lookup_error", "error": str(exc)}
        return False
    except Exception as exc:  # pragma: no cover - safety net
        payload.metadata["map"] = {"status": "error", "error": str(exc)}
        return False

    response_message = result.response.get("response_message") or "Construction site map"
    payload.charts.append(
        ChartPayload(
            title=response_message,
            figure_json=result.response["figure_json"],
        )
    )
    payload.metadata["map"] = {
        "status": "ok",
        "result": result.to_dict(),
        "response_message": response_message,
        "map_metadata": result.response.get("metadata", {}),
    }
    return True


def derive_lookup_context(prompt: str) -> Optional[Dict[str, Any]]:
    """Return structured GeoChat dataset context when prompt targets map lookups."""

    try:
        lookup_result = lookup_prompt_records(prompt)
    except QueryParsingError:
        return None
    except LookupExecutionError as exc:
        error_text = str(exc)
        return {
            "func_name": "SM_overview",
            "func_output": {"status": "error", "error": error_text, "source": "geochat_map_api"},
            "classifier_data": {
                "Function": "SM_overview",
                "Source": "geochat_map_api",
                "Status": "error",
                "Error": error_text,
            },
            "params": {"lookup": None},
        }
    except Exception as exc:  # pragma: no cover - defensive guard
        error_text = str(exc)
        return {
            "func_name": "SM_overview",
            "func_output": {"status": "error", "error": error_text, "source": "geochat_map_api"},
            "classifier_data": {
                "Function": "SM_overview",
                "Source": "geochat_map_api",
                "Status": "error",
                "Error": error_text,
            },
            "params": {"lookup": None},
        }

    payload = {
        "data": lookup_result.records,
        "lookup": lookup_result.to_dict(),
        "source": "geochat_map_api",
        "records_fetched": lookup_result.total_records,
        "highlight_ids": sorted(lookup_result.highlight_ids),
    }
    if not lookup_result.records:
        payload["status"] = "no_data"
        payload["message"] = "GeoChat returned no plate records for this request."

    classifier_data = {
        "Function": "SM_overview",
        "Source": "geochat_map_api",
        "Lookup": lookup_result.intent.to_dict(),
        "Records": len(lookup_result.records),
        "RecordsFetched": lookup_result.total_records,
    }
    if lookup_result.api_url:
        classifier_data["ApiUrl"] = lookup_result.api_url

    params = {"lookup": lookup_result.intent.to_dict()}

    return {
        "func_name": "SM_overview",
        "func_output": payload,
        "classifier_data": classifier_data,
        "params": params,
    }


def build_geo_lookup_payload(prompt: str, target_func: str) -> Dict[str, Any]:
    """Fetch plate details from the GeoChat API for deterministic responses."""

    def _error_payload(message: str) -> Dict[str, Any]:
        error_output = {
            "status": "error",
            "type": target_func,
            "intent": "plates_data",
            "error": message,
            "source": "geochat_map_api",
        }
        classifier_meta = {
            "Function": target_func,
            "Source": "geochat_map_api",
            "Status": "error",
            "Error": message,
        }
        return {
            "func_name": target_func,
            "func_output": error_output,
            "classifier_data": classifier_meta,
            "params": {"lookup": None},
        }

    try:
        lookup_result = lookup_prompt_records(prompt)
    except QueryParsingError as exc:
        return _error_payload(str(exc))
    except LookupExecutionError as exc:
        return _error_payload(str(exc))
    except Exception as exc:  # pragma: no cover - defensive guard
        return _error_payload(str(exc))

    output_payload: Dict[str, Any] = {
        "status": "ok",
        "type": target_func,
        "intent": "plates_data",
        "data": lookup_result.records,
        "lookup": lookup_result.to_dict(),
        "source": "geochat_map_api",
        "records_fetched": lookup_result.total_records,
        "highlight_ids": sorted(lookup_result.highlight_ids),
    }
    if not lookup_result.records:
        output_payload["status"] = "no_data"
        output_payload["message"] = "GeoChat returned no plate records for this request."

    classifier_meta = {
        "Function": target_func,
        "Source": "geochat_map_api",
        "Lookup": lookup_result.intent.to_dict(),
        "Records": len(lookup_result.records),
        "RecordsFetched": lookup_result.total_records,
    }
    if lookup_result.api_url:
        classifier_meta["ApiUrl"] = lookup_result.api_url

    return {
        "func_name": target_func,
        "func_output": output_payload,
        "classifier_data": classifier_meta,
        "params": {"lookup": lookup_result.intent.to_dict()},
    }


def render_assistant_payload(payload: ResponsePayload, func_output=None):
    """Render the structured response returned by the logic generator."""

    if payload.message:
        st.markdown(payload.message)

    if payload.insights:
        st.markdown("**Key takeaways:**")
        for insight in payload.insights:
            st.markdown(f"\n - {insight}")

    for table in payload.tables:
        st.markdown(f"\n **{table.title}**")
        st.dataframe(table.dataframe, width = 'stretch')

    for chart in payload.charts:
        try:
            fig = pio.from_json(chart.figure_json)
            st.plotly_chart(fig, width = 'stretch')
            if chart.title:
                st.caption(chart.title)
        except Exception as exc:
            st.caption(f"[chart rendering failed: {exc}]")

    map_meta = payload.metadata.get("map")
    if map_meta and map_meta.get("status") != "ok":
        st.warning(f"Map unavailable: {map_meta.get('error')}")

    if func_output is None:
        func_output = payload.metadata.get("payload") if payload.metadata else None

    path = None
    if isinstance(func_output, dict):
        path = func_output.get("path")
        func_type = func_output.get("type")
    else:
        func_type = None
    if path and os.path.exists(path):
        label = "Download resource"
        if func_type == "reporter_Asaoka":
            label = "Download Asaoka Report PDF"
        elif func_type == "plot_combi_S":
            label = "Download Combined Settlement Plot"
        with open(path, "rb") as f:
            st.download_button(label=label, data=f.read(), file_name=os.path.basename(path), mime="application/pdf")


# --- Streamlit Chat UI for NL2Func Pipeline ---
st.set_page_config(page_title="Boskalis GeoChat", layout="wide")
st.title("Boskalis GeoChat Assistant")

# --- Sidebar Controls ---
st.sidebar.markdown("### Controls")
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.slot_state = None

    # Clear clash resolution state
    if "clash_state" in st.session_state:
        del st.session_state.clash_state

    st.rerun()

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "slot_state" not in st.session_state:
    st.session_state.slot_state = None
if "dispatcher" not in st.session_state:
    st.session_state.classifier = load_classifier()
    st.session_state.llm_router = LLMRouter()
    st.session_state.dispatcher = Dispatcher()

# --- Helper Functions ---
def add_message(role, message):
    st.session_state.chat_history.append((role, message))
    if len(st.session_state.chat_history) > 200:
        st.session_state.chat_history.pop(0)

def display_chat():
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            if isinstance(msg, ResponsePayload):
                render_assistant_payload(msg)
            elif isinstance(msg, str):
                st.markdown(msg)
            elif isinstance(msg, Iterable):
                for chunk in msg:
                    st.markdown(chunk, unsafe_allow_html=True)
            else:
                st.markdown(str(msg))

# --- LLM Streaming Response ---
def stream_response(user_input, func_name=None, params=None, func_output=None, classifier_metadata=None):
    # echo user
    add_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    request_wants_map = wants_map(user_input)

    with st.chat_message("assistant"), st.spinner("Assistant is composing..."):
        effective_func_name = func_name
        effective_output = func_output
        classifier_meta = classifier_metadata

        if classifier_meta is None:
            if func_name:
                classifier_meta = {
                    "Function": func_name,
                    "Params": params,
                    "Output": func_output,
                }
            elif not request_wants_map:
                lookup_context = derive_lookup_context(user_input)
                if lookup_context:
                    effective_func_name = lookup_context["func_name"]
                    effective_output = lookup_context["func_output"]
                    classifier_meta = lookup_context["classifier_data"]
                    params = lookup_context.get("params")

        payload = st.session_state.llm_router.handle_user(
            user_input,
            func_name=effective_func_name,
            classifier_data=classifier_meta,
            func_output=effective_output,
        )
        if request_wants_map:
            map_success = attach_map_payload(user_input, payload)
            if map_success:
                # Map prompts should suppress the conversational answer and any non-map visuals.
                payload.message = ""
                payload.insights.clear()
                payload.tables.clear()
                if payload.charts:
                    payload.charts = [payload.charts[-1]]
        render_assistant_payload(payload, func_output=effective_output)
    add_message("assistant", payload)

# --- Main Flow ---
display_chat()

given_input = st.chat_input("Type your message...", key="input")
if given_input:
    input_text = given_input.strip()
    disp = st.session_state.dispatcher
    print("[DEBUG] Inp Recieved")

    # --- Slot-filling active ---
    if st.session_state.slot_state:
        print("[DEBUG] Slot State active")
        slot_info = st.session_state.slot_state
        slot = slot_info["slots_needed"][0]
        answer = input_text  # DO NOT inject tags here!
        print(f"[DEBUG] Slot-filling context before parse: {slot_info['aux_ctx']}")
        add_message("user", answer)
        with st.chat_message("user"):
            st.markdown(answer)

        if answer.lower() in {"skip", "never mind", "nvm", "nah"}:
            skip_msg = "Skipping function; sending to LLM."
            add_message("assistant", skip_msg)
            with st.chat_message("assistant"):
                st.markdown(skip_msg)
            st.session_state.slot_state = None
            stream_response(slot_info["orig_query"])
        else:
            slot_info["aux_ctx"] += f"\n{slot}: {answer}"
            if slot_info["slots_needed"]:
                slot_info["slots_needed"].pop(0)
            try:
                description = get_function_description(slot_info["func_name"])
                with st.spinner(f"Running {description}..."):
                    params = disp.pure_parse(slot_info["aux_ctx"], slot_info["func_name"])
                    out = disp.run_function(slot_info["func_name"], params)
                st.session_state.slot_state = None
                
                stream_response(slot_info["orig_query"], slot_info["func_name"], params, out)
            except MissingSlot as ms:
                print(f"[DEBUG] MissingSlot: {ms.slot}")
                slot_state = st.session_state.slot_state or {}
                slot_state["slots_needed"] = [ms.slot]
                st.session_state.slot_state = slot_state
                prompt = f"What's your {ms.slot}?"
                add_message("assistant", prompt)
                func = slot_info["func_name"]
                with st.chat_message("assistant"):
                    st.markdown(
                        f"<div style='margin-bottom:0.5em; padding:0.5em; background:#f6f6f6; border-radius:6px;'>"
                        f"<b>NOTE:</b><br>"
                        f"<span style='font-size:0.92em; color:#888; font-style:italic;'>The system is trying to call a function: {func}. If you believe that this is not intended, please type 'skip' or similar.</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(prompt)
            except Exception as e:
                err_msg = f"Error: {e}"
                add_message("assistant", err_msg)
                with st.chat_message("assistant"):
                    st.markdown(err_msg)
                st.session_state.slot_state = None

    # --- Handle clash resolution ---
    elif "clash_state" in st.session_state:
        print("[DEBUG] Clash State active")
        clash_info = st.session_state.clash_state
        answer = input_text.lower().strip()

        add_message("user", input_text)
        with st.chat_message("user"):
            st.markdown(input_text)

        if answer in {"1", "2"}:
            chosen_func = clash_info["classifier_func"] if answer == "1" else clash_info["rule_func"]
            del st.session_state.clash_state

            add_message("assistant", f"Using: {chosen_func}")
            with st.chat_message("assistant"):
                st.markdown(f"Using: {chosen_func}")

            try:
                description = get_function_description(chosen_func)
                if chosen_func in GEOCHAT_REDIRECT_FUNCS:
                    with st.spinner(f"Running {description}..."):
                        geo_payload = build_geo_lookup_payload(clash_info["original_input"], chosen_func)
                    stream_response(
                        clash_info["original_input"],
                        geo_payload["func_name"],
                        geo_payload.get("params"),
                        geo_payload["func_output"],
                        classifier_metadata=geo_payload["classifier_data"],
                    )
                else:
                    with st.spinner(f"Running {description}..."):
                        params = disp.pure_parse(clash_info["tagged_input"], chosen_func)
                        out = disp.run_function(chosen_func, params)
                    stream_response(clash_info["original_input"], chosen_func, params, out)
            except MissingSlot as ms:
                st.session_state.slot_state = {
                    "func_name": chosen_func,
                    "aux_ctx": clash_info["tagged_input"],
                    "slots_needed": [ms.slot],
                    "orig_query": clash_info["original_input"],
                }
                prompt = f"What's your {ms.slot}?"
                add_message("assistant", prompt)
                with st.chat_message("assistant"):
                    st.markdown(prompt)
            except Exception as exc:
                err_msg = f"Error: {exc}"
                add_message("assistant", err_msg)
                with st.chat_message("assistant"):
                    st.markdown(err_msg)

        elif answer in {"skip"}:
            del st.session_state.clash_state
            add_message("assistant", "Skipping function; sending to LLM.")
            with st.chat_message("assistant"):
                st.markdown("Skipping function; sending to LLM.")
            stream_response(clash_info["original_input"])

        else:
            add_message("assistant", "Please type '1', '2', or 'skip'.")
            with st.chat_message("assistant"):
                st.markdown("Please type '1', '2', or 'skip'.")

    # --- Initial classify (inject tags here only) ---
    else:
        print("[DEBUG] Tags case active")
        try: 
            func_name = choose_function(input_text, st.session_state.classifier)
        # Continue with normal function execution logic
            print("[DEBUG] Final Function: ", func_name)
            if func_name:
                if func_name in GEOCHAT_REDIRECT_FUNCS:
                    description = get_function_description(func_name)
                    with st.spinner(f"Running {description}..."):
                        geo_payload = build_geo_lookup_payload(input_text, func_name)
                    stream_response(
                        input_text,
                        geo_payload["func_name"],
                        geo_payload.get("params"),
                        geo_payload["func_output"],
                        classifier_metadata=geo_payload["classifier_data"],
                    )
                else:
                    try:
                        description = get_function_description(func_name)
                        with st.spinner(f"Running {description}..."):
                            params = disp.pure_parse(input_text, func_name)
                            out = disp.run_function(func_name, params)
                        print("[DEBUG] OUTPUT after running: ", out)
                        stream_response(input_text, func_name, params, out)
                    except MissingSlot as ms:
                        st.session_state.slot_state = {
                            "func_name": func_name,
                            "aux_ctx": input_text,
                            "slots_needed": [ms.slot],
                            "orig_query": input_text
                        }
                        prompt = f"What's your {ms.slot}?"
                        add_message("assistant", prompt)
                        with st.chat_message("assistant"):
                            st.markdown(
                                f"<div style='margin-bottom:0.5em; padding:0.5em; background:#f6f6f6; border-radius:6px;'>"
                                f"<b>NOTE:</b><br>"
                                f"<span style='font-size:0.92em; color:#888; font-style:italic;'>The system is trying to call a function: {func_name}. If you believe that this is not intended, please type 'skip' or similar.</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            st.markdown(prompt)
                    except Exception as exc:
                        print(f"[DEBUG] Exception caught: {exc}")
                        err_msg = f"Error: {exc}"
                        add_message("assistant", err_msg)
                        with st.chat_message("assistant"):
                            st.markdown(err_msg)
            else:
                print("[DEBUG] Function calling skipped")
                stream_response(input_text)
        except FunctionClash as clash:
            # Handle function clash like missing slot
            st.session_state.clash_state = {
                "classifier_func": clash.classifier_func,
                "rule_func": clash.rule_func,
                "original_input": clash.original_input,
                "tagged_input": input_text  # Store the input with tags
            }
            
            # Show clash resolution prompt
            prompt = "I detected a function clash. Which function should I use?"
            add_message("assistant", prompt)
            description1 = get_function_description(clash.classifier_func)
            description2 = get_function_description(clash.rule_func)
            with st.chat_message("assistant"):
                st.markdown(
                    f"<div style='margin-bottom:0.5em; padding:0.5em; background:#fff3cd; border:1px solid #ffeaa7; border-radius:6px;'>"
                    f"<b>⚠️ Function Clash Detected:</b><br>"
                    f"<span style='font-size:0.92em;'>The classifier suggests <strong>{description1}</strong> but rules suggest <strong>{description2}</strong></span><br><br>"
                    f"<span style='font-size:0.9em; color:#666;'>Please type one of the following (1/2/skip):</span><br>"
                    f"<span style='font-size:0.9em;'>• <strong>'1. '</strong> - Use {description1}</span><br>"
                    f"<span style='font-size:0.9em;'>• <strong>'2. '</strong> - Use {description2}</span><br>"
                    f"<span style='font-size:0.9em;'>• <strong>'skip'</strong> - Send directly to LLM</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.markdown(prompt)