"""Facade that replaces the former LLM router with modular logic."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple, cast

import re
import random

import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.business_rules import BusinessRuleEngine
from core.conversation import ConversationManager, ConversationTurn
from core.intent_model import GrammarIntentModel, IntentAnalysis
from core.retriever import GlossaryRetriever
from core.statistics import StatisticalInsightEngine, StatisticalPattern
from core.templates import DynamicTemplateManager, TemplateContext

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


@dataclass
class TablePayload:
    title: str
    dataframe: pd.DataFrame


@dataclass
class ChartPayload:
    title: str
    figure_json: str


@dataclass
class ResponsePayload:
    message: str
    insights: List[str] = field(default_factory=list)
    tables: List[TablePayload] = field(default_factory=list)
    charts: List[ChartPayload] = field(default_factory=list)
    fallback: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResponseGenerator:
    def __init__(self) -> None:
        self.intent_model = GrammarIntentModel()
        self.function_config = self._load_function_config()
        self.function_intents = self._extract_function_intents(self.function_config)
        self._intent_keywords = self._build_keyword_index(self.function_config)
        self.domain_focus = self.function_config.get("metadata", {}).get("domain_focus", "programme focus")
        self.templates = DynamicTemplateManager()
        self.statistics = StatisticalInsightEngine()
        self.business = BusinessRuleEngine(config_path=self._default_rules_path())
        self.conversation = ConversationManager()
        self.retriever = GlossaryRetriever()
        self._knowledge: Deque[str] = deque(maxlen=6)
        self._definition_threshold = 0.25
        self._last_focus_statement = ""
        self._last_focus_topic = ""
        self._rng = random.Random()

    # ------------------------------------------------------------------
    def handle_user(
        self,
        user_input: str,
        func_name: Optional[str] = None,
        classifier_data: Optional[Dict[str, Any]] = None,
        func_output: Optional[Any] = None,
        stream: bool = False,  # kept for backwards compatibility
    ) -> ResponsePayload:
        del stream  # streaming no longer supported; response returned in one go

        cleaned_input = user_input.strip()
        analysis = self.intent_model.analyse(cleaned_input)
        glossary_matches = self.retriever.search(cleaned_input, top_k=3)
        glossary_score = glossary_matches[0]["score"] if glossary_matches else 0.0
        focus_sentence, focus_topic = self._derive_focus_from_history()
        if analysis.state != "smalltalk" and not focus_sentence:
            current_topic = self._sanitize_focus_topic(cleaned_input)
            if current_topic:
                focus_sentence = self._build_focus_sentence(current_topic, prefix="Let's zero in on")
                focus_topic = current_topic
        applied_focus_sentence = focus_sentence
        focus_repeat = False
        if focus_sentence and focus_sentence == self._last_focus_statement:
            applied_focus_sentence = ""
            focus_repeat = True
        elif not focus_sentence and focus_topic and focus_topic == self._last_focus_topic:
            focus_repeat = True
        if analysis.state == "smalltalk":
            focus_sentence = ""
            focus_topic = ""
            applied_focus_sentence = ""
            focus_repeat = False
        keyword_intent = self._match_keyword_intent(cleaned_input)
        inferred_intent = self._resolve_intent(
            func_name=func_name,
            grammar_intent=analysis.intent,
            keyword_intent=keyword_intent,
            glossary_score=glossary_score,
        )
        sentiment = self.conversation.analyse_sentiment(cleaned_input)
        dialog_state = analysis.state
        dialog_act = self._map_state_to_dialog_act(dialog_state)
        if sentiment == "negative":
            persona = "empathetic"
        elif dialog_state == "smalltalk":
            persona = "casual"
        else:
            persona = "analytical"
        knowledge_line = "" if dialog_state == "smalltalk" else self._knowledge_prompt()
        tone = getattr(self.conversation.personality, "tone", "professional")
        context = TemplateContext(
            sentiment=sentiment,
            conversation_stage=self.conversation.stage(),
            conversation_summary=applied_focus_sentence,
            focus_statement=applied_focus_sentence,
            focus_topic=focus_topic,
            engagement=self.conversation.engagement_level(),
            persona=persona,
            tone=tone,
            domain_focus=self.domain_focus,
            knowledge_line=knowledge_line,
            dialog_state=dialog_state,
            focus_repeat=focus_repeat,
        )

        if inferred_intent == "definition":
            payload = self._handle_definition(cleaned_input, context, glossary_matches)
        elif inferred_intent == "data_insight":
            payload = self._handle_data(func_name, func_output, context, classifier_data)
        elif inferred_intent == "resource":
            payload = self._handle_resource(func_name, func_output, context)
        else:
            payload = self._handle_conversation(cleaned_input, context)

        self.conversation.record_turn(
            ConversationTurn(
                user=cleaned_input,
                assistant=payload.message,
                intent=inferred_intent,
                sentiment=sentiment,
                dialog_act=dialog_act,
            )
        )
        payload.metadata.setdefault("intent", inferred_intent)
        payload.metadata.setdefault("sentiment", sentiment)
        payload.metadata.setdefault("dialog_act", dialog_act)
        payload.metadata.setdefault("dialog_state", dialog_state)
        payload.metadata.setdefault("intent_scores", analysis.intent_scores)
        payload.metadata.setdefault("state_scores", analysis.state_scores)
        payload.metadata.setdefault("glossary_score", glossary_score)
        payload.metadata.setdefault("keyword_intent", keyword_intent)
        if focus_sentence:
            self._last_focus_statement = focus_sentence
        if focus_topic:
            self._last_focus_topic = focus_topic
        return payload

    # Definition --------------------------------------------------------
    def _handle_definition(
        self,
        user_input: str,
        ctx: TemplateContext,
        matches: Optional[List[Dict[str, Any]]] = None,
    ) -> ResponsePayload:
        entries = matches or self.retriever.search(user_input, top_k=3)
        if not entries:
            return self._fallback_response()
        primary = next(
            (entry for entry in entries if entry.get("term") and entry.get("definition")),
            None,
        )
        if primary is None:
            return self._fallback_response()

        primary_term = str(primary["term"]).strip()
        primary_definition = str(primary["definition"]).strip()
        primary_score = primary.get("score")

        related_entries: List[Dict[str, Any]] = []
        for entry in entries:
            if entry is primary:
                continue
            term = entry.get("term")
            definition = entry.get("definition")
            if not term or not definition:
                continue
            related_entries.append(
                {
                    "term": str(term).strip(),
                    "definition": str(definition).strip(),
                    "score": entry.get("score"),
                }
            )

        see_also_terms = [entry["term"] for entry in related_entries]
        definition_text = primary_definition.strip()
        if see_also_terms:
            definition_text = self._ensure_sentence(definition_text)
            definition_text = f"{definition_text} See also: {', '.join(see_also_terms)}."
        definition_text = self._ensure_sentence(definition_text)

        template = self.templates.get_template("definition", ctx)
        message = self.templates.render(
            template,
            term=primary_term or user_input.strip().capitalize(),
            definition=definition_text,
        )

        insights = [f"{primary_term}: {primary_definition}"]
        for entry in related_entries:
            insights.append(f"{entry['term']}: {entry['definition']}")

        self._push_knowledge(insights[:2])
        metadata: Dict[str, Any] = {
            "source": "glossary",
            "term": primary_term,
            "matches": len(entries),
        }
        if primary_score is not None:
            metadata["primary_score"] = primary_score
        if see_also_terms:
            metadata["see_also"] = see_also_terms

        return ResponsePayload(message=message, insights=insights, metadata=metadata)

    # Data --------------------------------------------------------------
    def _handle_data(
        self,
        func_name: Optional[str],
        func_output: Optional[Any],
        ctx: TemplateContext,
        classifier_data: Optional[Dict[str, Any]],
    ) -> ResponsePayload:
        if isinstance(func_output, dict) and func_output.get("status") == "error":
            detail = str(func_output.get("error", "an unexpected issue occurred"))
            return self._function_failure_response(
                func_name,
                reason="function_error",
                detail=detail,
                status="error",
                extra_metadata={"classifier": classifier_data},
            )

        dataset = self._extract_dataset(func_output)
        if dataset.empty:
            return self._function_failure_response(
                func_name,
                reason="no_data",
                detail="no readings were returned from the data source",
                status="no_data",
                extra_metadata={"classifier": classifier_data},
            )

        data_type = self._map_function_to_data_type(func_name)
        patterns = self.statistics.generate_insights(dataset, data_type)
        insights = self.business.apply_business_context(
            [pattern.__dict__ for pattern in patterns],
            data_type,
            dataset,
        )

        selected_insights = self._select_insights(insights)
        intro_sentences = self._build_intro_sentences(selected_insights, ctx)
        detail_sentences = [self._format_insight_description(entry) for entry in selected_insights]

        template = self.templates.get_template("data_insight", ctx)
        lead = "Here is what stands out" if intro_sentences else "Here is the raw data"
        insight_sentence = "\n".join(intro_sentences) or "No clear anomalies detected in the latest readings."
        message = self.templates.render(
            template,
            lead=lead,
            insight_sentence=insight_sentence,
        )

        tables = [TablePayload(title="Latest dataset", dataframe=dataset)]
        charts = self._build_charts(dataset, data_type)
        key_takeaways = self._build_key_takeaways(selected_insights)

        metadata = {
            "function": func_name,
            "classifier": classifier_data,
            "insight_count": len(selected_insights),
        }
        self._push_knowledge(detail_sentences[:2])
        return ResponsePayload(
            message=message,
            insights=key_takeaways,
            tables=tables,
            charts=charts,
            metadata=metadata,
        )

    # Conversation ------------------------------------------------------
    def _handle_conversation(self, user_input: str, ctx: TemplateContext) -> ResponsePayload:
        if ctx.dialog_state == "smalltalk":
            template = self.templates.get_template("smalltalk", ctx)
            message = self.templates.render(
                template,
                smalltalk_opening=self._smalltalk_reply(user_input),
                smalltalk_followup=self._smalltalk_pivot(ctx),
            )
            return ResponsePayload(message=message)

        focus_line = ctx.focus_statement.strip()
        if focus_line:
            focus_line = self._ensure_sentence(focus_line)
        else:
            focus_line = "Let's decide which plate cluster or KPI deserves attention next."

        hint_line = self._conversation_hint(ctx)
        next_step_sentence = self._ensure_sentence(self._conversation_next_step(ctx))

        if ctx.dialog_state in {"clarify", "investigate"}:
            template = self.templates.get_template("clarification", ctx)
        else:
            template = self.templates.get_template("conversation", ctx)

        message = self.templates.render(
            template,
            focus_line=focus_line,
            hint_line=hint_line,
            next_step_sentence=next_step_sentence,
        )
        return ResponsePayload(message=message)

    def _handle_resource(
        self,
        func_name: Optional[str],
        func_output: Optional[Any],
        ctx: TemplateContext,
    ) -> ResponsePayload:
        if isinstance(func_output, dict) and func_output.get("status") == "error":
            detail = str(func_output.get("error", "unknown error"))
            return self._function_failure_response(
                func_name,
                reason="resource_error",
                detail=detail,
                status="error",
            )
        template = self.templates.get_template("greeting", ctx)
        base = self.templates.render(template)
        if func_name == "reporter_Asaoka":
            suffix = "The Asaoka PDF report is available for download in the sidebar."
        elif func_name == "plot_combi_S":
            suffix = "The combined settlement plot is ready to download."
        else:
            suffix = "Your requested file is ready."
        message = f"{base}\n\n{suffix}"
        metadata = {"function": func_name, "status": "resource_ready", "payload": func_output}
        return ResponsePayload(message=message, metadata=metadata)

    # Utilities ---------------------------------------------------------
    def _function_failure_response(
        self,
        func_name: Optional[str],
        reason: str,
        detail: str = "",
        status: str = "error",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ResponsePayload:
        function_label = func_name or "the requested function"
        cleaned = detail.strip()
        if cleaned.endswith("."):
            cleaned = cleaned[:-1]
        if cleaned:
            message = (
                f"Sorry, I ran into an issue while working with {function_label}: {cleaned}. "
                "Please double-check the inputs or try again shortly."
            )
        else:
            message = (
                f"Sorry, I ran into an issue while working with {function_label}. "
                "Please double-check the inputs or try again shortly."
            )
        metadata: Dict[str, Any] = {"function": func_name, "status": status, "reason": reason}
        if extra_metadata:
            metadata.update(extra_metadata)
        return ResponsePayload(message=message, fallback=True, metadata=metadata)

    def _fallback_response(self, reason: str = "generic") -> ResponsePayload:
        template = self.templates.get_template("fallback", TemplateContext(domain_focus=self.domain_focus))
        message = self.templates.render(template)
        return ResponsePayload(message=message, fallback=True, metadata={"reason": reason})

    def _extract_dataset(self, func_output: Optional[Any]) -> pd.DataFrame:
        if isinstance(func_output, dict) and "data" in func_output:
            data = func_output.get("data")
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                return pd.DataFrame([data])
            if isinstance(data, pd.DataFrame):
                return data
        if isinstance(func_output, list):
            return pd.DataFrame(func_output)
        if isinstance(func_output, pd.DataFrame):
            return func_output
        return pd.DataFrame()

    @staticmethod
    def _map_function_to_data_type(func_name: Optional[str]) -> str:
        if func_name == "SM_overview":
            return "plates_data"
        if func_name == "Asaoka_data":
            return "plates_data"
        return "conversation"

    def _select_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        for entry in insights:
            description = entry.get("description")
            if not description:
                continue
            payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else None
            selected.append(
                {
                    "pattern_type": entry.get("pattern_type"),
                    "description": str(description).strip(),
                    "recommendation": str(
                        entry.get("recommendation")
                        or entry.get("business_context")
                        or ""
                    ).strip(),
                    "payload": payload,
                    "focus_points": entry.get("focus_points"),
                    "cohorts": entry.get("cohorts"),
                    "data_type": entry.get("data_type"),
                }
            )
            if len(selected) >= 2:
                break
        return selected

    @staticmethod
    def _pattern_prefix(pattern_type: Optional[str]) -> str:
        mapping = {
            "rule_breach": "⚠️ ",
            "outlier": "🔍 ",
            "trend": "📈 ",
            "correlation": "🔗 ",
        }
        if not pattern_type:
            return ""
        return mapping.get(pattern_type, "")

    @staticmethod
    def _sanitize_column_name(name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalnum())

    def _format_insight_description(self, entry: Dict[str, Any]) -> str:
        description = entry.get("description", "").strip()
        if not description:
            return ""
        prefix = self._pattern_prefix(entry.get("pattern_type"))
        return f"{prefix}{description}".strip()

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_focus_points(entry: Dict[str, Any]) -> List[str]:
        focus: Any = entry.get("focus_points")
        if not focus and isinstance(entry.get("payload"), dict):
            focus = entry["payload"].get("focus_points")
        points: List[str] = []
        if isinstance(focus, (list, tuple, set)):
            points = [str(item) for item in focus if item not in (None, "")]
        elif focus not in (None, ""):
            points = [str(focus)]
        return points

    @staticmethod
    def _format_focus_label(points: List[str]) -> str:
        if not points:
            return "the cohort"
        if len(points) == 1:
            return points[0]
        if len(points) == 2:
            return f"{points[0]} and {points[1]}"
        if len(points) == 3:
            return f"{points[0]}, {points[1]}, and {points[2]}"
        return f"{points[0]}, {points[1]}, and others"

    @staticmethod
    def _humanise_metric(raw: Any) -> str:
        text = str(raw or "").strip()
        if not text:
            return "the metric"
        cleaned = re.sub(r"[_\-]+", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return "the metric"
        if cleaned.isupper() and len(cleaned) <= 5:
            return cleaned
        return cleaned.lower()

    @staticmethod
    def _sentence_case(value: str) -> str:
        if not value:
            return value
        if value.isupper():
            return value
        return value[0].upper() + value[1:]

    def _build_intro_sentences(
        self,
        entries: List[Dict[str, Any]],
        ctx: Optional[TemplateContext] = None,
    ) -> List[str]:
        sentences: List[str] = []
        persona = getattr(ctx, "persona", "analytical") if ctx else "analytical"
        tone = (getattr(ctx, "tone", "professional") if ctx else "professional").lower()
        rng = self._rng

        def merge_sequences(*seqs: Optional[Iterable[str]]) -> List[str]:
            merged: List[str] = []
            for seq in seqs:
                if not seq:
                    continue
                for item in seq:
                    if item not in merged:
                        merged.append(item)
            return merged

        persona_followups: Dict[str, List[str]] = {
            "analytical": [
                "I'll keep the programme cadence aligned while we monitor the readings.",
                "I'll keep the compliance dashboards synced while we track the movement.",
                "I'll keep the instrumentation views reconciled while this settles.",
            ],
            "empathetic": [
                "I'll keep a gentle eye on the readings so nothing catches us off guard.",
                "I'll ride alongside the cohort and nudge you if it drifts again.",
                "I'll keep the alerts soft but ready in case this flares back up.",
            ],
            "casual": [
                "I'll keep tabs so nothing slips while we juggle the rest.",
                "I'll keep an eye out and shout if the dial swings harder.",
                "I'll keep the dials steady till we decide the next move.",
            ],
        }
        tone_followups: Dict[str, List[str]] = {
            "professional": [
                "I'll log this in the status tracker while we monitor it.",
                "I'll keep the compliance memo updated in the background.",
            ],
            "empathetic": [
                "I'll keep a calm watch so you don't have to worry about it.",
                "I'll look after the alerts and nudge you gently if it shifts.",
            ],
            "casual": [
                "I'll keep it on the watchlist while you juggle the rest.",
                "I'll keep the radar light on until we call the next move.",
            ],
        }
        shared_followups: List[str] = [
            "Ping me if you want me to surface the raw tables or thresholds.",
            "Happy to pivot to another KPI whenever you're ready.",
            "Let me know if you want to drill into the raw dataset.",
            "We can zoom into the focus plates whenever you need.",
        ]
        watch_templates = [
            "{focus} stay on my radar while this pattern beds in.",
            "{focus} keep a slot on the compliance board for now.",
            "{focus} remain pinned for follow-up while the numbers stabilise.",
            "{focus} stay under quiet observation while the signal matures.",
        ]
        tone_watch_templates: Dict[str, List[str]] = {
            "professional": [
                "{focus} hold a reserved lane in the programme tracker for now.",
                "{focus} stay highlighted on the compliance dashboard while we reassess.",
            ],
            "empathetic": [
                "{focus} get a softer spotlight while the trend settles.",
                "{focus} stay under a gentle watch until the signal calms down.",
            ],
            "casual": [
                "{focus} hang out on the watchlist while we feel this out.",
                "{focus} keep a placeholder on the radar till we see the next move.",
            ],
        }
        cohort_templates = [
            "I'm keeping the wider cohort aligned while we monitor the drift.",
            "I'll hold the broader cohort steady so the context stays intact.",
            "I'll keep the wider cohort balanced so the narrative stays grounded.",
        ]
        tone_cohort_templates: Dict[str, List[str]] = {
            "professional": [
                "I'll keep the wider cohort benchmarked so context stays sharp.",
            ],
            "empathetic": [
                "I'll watch the broader cohort gently so nothing feels sudden.",
            ],
            "casual": [
                "I'll keep the rest of the cohort mellow while this plays out.",
            ],
        }

        for entry in entries:
            pattern = entry.get("pattern_type")
            payload = entry.get("payload") or {}
            metric = payload.get("column") or payload.get("metric") or entry.get("metric") or "the metric"
            focus_points = self._extract_focus_points(entry)
            focus_label = self._format_focus_label(focus_points)
            metric_label = self._humanise_metric(metric)

            if pattern == "trend":
                slope = self._to_float(payload.get("slope") or entry.get("slope")) or 0.0
                if slope > 0:
                    direction = "upward"
                elif slope < 0:
                    direction = "downward"
                else:
                    direction = "shifting"
                sentences.append(f"{self._sentence_case(metric_label)} is trending {direction} for {focus_label}.")
            elif pattern == "outlier":
                sentences.append(f"{self._sentence_case(metric_label)} looks anomalous on {focus_label}.")
            elif pattern == "rule_breach":
                sentences.append(f"{self._sentence_case(metric_label)} is slipping outside the compliance band for {focus_label}.")
            elif pattern == "correlation":
                columns = payload.get("columns") if isinstance(payload, dict) else None
                if isinstance(columns, (list, tuple, set)) and columns:
                    column_text = " and ".join(self._humanise_metric(col) for col in columns if col)
                    if column_text:
                        sentences.append(f"{self._sentence_case(column_text)} appear coupled right now.")
                        continue
                sentences.append("A coupled movement is emerging across the monitored metrics.")
            else:
                sentences.append(f"{self._sentence_case(metric_label)} deserves an extra look across {focus_label}.")

            if focus_points:
                focus_text = self._sentence_case(focus_label)
                watch_options = merge_sequences(
                    [tpl.format(focus=focus_text) for tpl in tone_watch_templates.get(tone, [])],
                    [tpl.format(focus=focus_text) for tpl in watch_templates],
                )
                sentences.append(rng.choice(watch_options))
            elif entry.get("cohorts"):
                cohort_options = merge_sequences(tone_cohort_templates.get(tone), cohort_templates)
                sentences.append(rng.choice(cohort_options))

            if len(sentences) >= 4:
                break

        if not sentences:
            sentences.append("Nothing dramatic popping up across the latest cohort just yet.")

        followup_pool = merge_sequences(
            persona_followups.get(persona),
            tone_followups.get(tone),
            shared_followups,
        )
        used: Set[str] = set(sentences)
        while len(sentences) < 4 and followup_pool:
            line = rng.choice(followup_pool)
            if line in used:
                followup_pool = [item for item in followup_pool if item != line]
                continue
            sentences.append(line)
            used.add(line)

        if len(sentences) > 4:
            sentences = sentences[:4]

        return sentences

    def _build_key_takeaways(self, entries: List[Dict[str, Any]]) -> List[str]:
        takeaways: List[str] = []
        for entry in entries:
            description_text = self._format_insight_description(entry)
            if description_text:
                takeaways.append(f"Insight: {description_text}")
            recommendation = (entry.get("recommendation") or entry.get("business_context") or "").strip()
            if recommendation:
                takeaways.append(f"Recommendation: {recommendation}")
        return takeaways

    def _build_charts(self, df: pd.DataFrame, data_type: str) -> List[ChartPayload]:
        charts: List[ChartPayload] = []
        if df.empty:
            return charts

        if data_type != "plates_data":
            return charts

        frame = df.copy()
        sanitized = {self._sanitize_column_name(col): col for col in frame.columns}

        def resolve(*aliases: str) -> Optional[str]:
            for alias in aliases:
                key = self._sanitize_column_name(alias)
                if key in sanitized:
                    return sanitized[key]
            return None

        plate_col = resolve("PointID", "point_id", "plate", "plateid", "full_name", "id", "name")
        rate_col = resolve("7day_rate", "seven_day_rate", "weekly_rate")
        pressure_col = resolve("surcharge_pressure", "surcharge_pressure_kpa", "surchargepressure", "Surcharge_Pressure")
        doc_col = resolve("Asaoka_DOC", "asaoka_doc", "degreeofconsolidation", "doc")

        rules = self.business.rule_map.get("plates_data", []) if hasattr(self, "business") else []
        rule_lookup = {self._sanitize_column_name(rule.name): rule for rule in rules}

        def get_rule(*metric_aliases: str) -> Optional[Any]:
            for alias in metric_aliases:
                alias_key = self._sanitize_column_name(alias)
                if alias_key in rule_lookup:
                    return rule_lookup[alias_key]
            return None

        def collect_labels(index: pd.Index) -> List[str]:
            if plate_col and plate_col in frame.columns:
                return frame.loc[index, plate_col].astype(str).tolist()
            return [f"Plate {i}" for i in range(1, len(index) + 1)]

        box_sources: List[Tuple[str, pd.Series, List[str], Optional[Any]]] = []

        if rate_col in frame.columns:
            rate_series = pd.to_numeric(frame[rate_col], errors="coerce").dropna()
            if not rate_series.empty:
                box_sources.append((
                    "7-day Rate (mm)",
                    rate_series,
                    collect_labels(rate_series.index),
                    get_rule("7day_rate", "seven_day_rate", "weekly_rate"),
                ))

        if pressure_col in frame.columns:
            pressure_series = pd.to_numeric(frame[pressure_col], errors="coerce").dropna()
            if not pressure_series.empty:
                box_sources.append((
                    "Surcharge Pressure (kPa)",
                    pressure_series,
                    collect_labels(pressure_series.index),
                    get_rule("surcharge_pressure", "Surcharge_Pressure"),
                ))

        if box_sources:
            fig = make_subplots(rows=1, cols=len(box_sources), subplot_titles=[item[0] for item in box_sources])
            for idx, (title, series, labels, rule) in enumerate(box_sources, start=1):
                fig.add_trace(
                    go.Box(
                        y=series,
                        name=title,
                        boxpoints="outliers",
                        hovertext=labels,
                        hoverinfo="y+text",
                        marker=dict(color="#546E7A" if idx == 1 else "#00838F"),
                    ),
                    row=1,
                    col=idx,
                )
                fig.update_xaxes(title_text="", row=1, col=idx)
                fig.update_yaxes(title_text="", row=1, col=idx)
                if rule:
                    fig.add_hline(
                        y=rule.threshold,
                        line_color="#D32F2F",
                        line_dash="dot",
                        annotation_text=f"Threshold {rule.threshold:.2f}",
                        annotation_position="top right",
                        row=1,
                        col=idx,
                    )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=50, r=30, t=70, b=40),
                title_text="Rate & Pressure Distributions",
            )
            charts.append(
                ChartPayload(title="Rate & Pressure Distributions", figure_json=cast(str, pio.to_json(fig, validate=False)))
            )

        if doc_col in frame.columns:
            doc_series = pd.to_numeric(frame[doc_col], errors="coerce").dropna()
            if not doc_series.empty:
                labels = collect_labels(doc_series.index)
                doc_df = pd.DataFrame({
                    "Plate": labels,
                    "Asaoka_DOC": doc_series.values,
                }).sort_values("Asaoka_DOC", ascending=False)
                bar_fig = px.bar(doc_df, x="Plate", y="Asaoka_DOC", text="Asaoka_DOC")
                doc_min = float(doc_df["Asaoka_DOC"].min())
                doc_max = float(doc_df["Asaoka_DOC"].max())
                lower_bound = 50.0 if doc_min >= 50.0 else max(0.0, doc_min - 5.0)
                upper_bound = max(100.0, doc_max + 2.0)
                bar_fig.update_traces(marker_color="#1565C0", texttemplate="%{text:.1f}", textposition="outside")
                bar_fig.update_layout(
                    margin=dict(l=60, r=40, t=80, b=80),
                    title="Asaoka DOC by Plate",
                    yaxis=dict(range=[lower_bound, upper_bound]),
                )
                doc_rule = get_rule("Asaoka_DOC", "asaoka_doc", "degreeofconsolidation")
                if doc_rule:
                    bar_fig.add_hline(
                        y=doc_rule.threshold,
                        line_dash="dot",
                        line_color="#FF7043",
                        annotation_text=f"DOC threshold {doc_rule.threshold:.1f}",
                        annotation_position="top right",
                    )
                charts.append(
                    ChartPayload(title="Asaoka DOC by Plate", figure_json=cast(str, pio.to_json(bar_fig, validate=False)))
                )

        return charts

    def _derive_focus_from_history(self) -> tuple[str, str]:
        history = self.conversation.history()
        for turn in history:
            topic = self._sanitize_focus_topic(turn.user)
            if topic:
                sentence = self._build_focus_sentence(topic, prefix="We were last focusing on")
                return sentence, topic
        return "", ""

    def _conversation_hint(self, ctx: TemplateContext) -> str:
        if ctx.dialog_state == "smalltalk":
            hint = "Just keeping the monitoring dashboards warm—say the word when you want to dive back into the project."
            return self._ensure_sentence(hint)
        if ctx.focus_topic and not ctx.focus_repeat:
            base = f'We were last focusing on "{ctx.focus_topic}".'
        elif ctx.focus_topic:
            base = "I'm keeping that focus thread in view."  # reassure without repeating wording
        else:
            base = "We haven’t anchored on a specific plate cluster yet."
        state = ctx.dialog_state
        if state in {"clarify", "investigate"}:
            follow = "Could you spell out the plates, time window, or KPIs you want me to pull?"
        elif state == "goal_setting":
            follow = "Point me to the next milestone or compliance gate and I'll prep the insights."
        elif state == "rapport":
            follow = "Appreciate the check-in—ready when you want to move forward."
        elif state == "smalltalk":
            follow = "Whenever you're ready, just drop the plate IDs or focus area."
        else:
            follow = "Let me know if we should stay on that track or pivot to another angle."
        hint = f"{base} {follow}".strip()
        return self._ensure_sentence(hint)

    def _conversation_next_step(self, ctx: TemplateContext) -> str:
        if ctx.dialog_state == "smalltalk":
            return "Whenever you're ready, mention a plate, date window, or KPI and we'll switch straight back to the instrumentation details"
        if ctx.dialog_state == "goal_setting":
            return "Outline the next compliance checkpoint or deliverable and I'll line up the supporting metrics"
        stage = ctx.conversation_stage
        if stage == "early":
            return "Share the plate IDs, survey dates, or KPIs you want in scope and I'll gather the latest readings"
        if stage == "late":
            return "If there are lingering compliance checks or reports, flag them and I'll close the loop"
        return "Tell me the next plate cluster or time block to inspect and I'll surface the key metrics"

    def _smalltalk_reply(self, user_input: str) -> str:
        lowered = user_input.lower()
        if re.search(r"\b(thank|thanks|thank you|appreciate|amazing|awesome|great job)\b", lowered):
            return "You're welcome—always happy to keep the settlement story tidy."
        if re.search(r"how (are|ya|you)", lowered):
            return "Doing well—just keeping the Tuas dashboards humming in the background."
        if re.search(r"what('?s| is) up|sup", lowered):
            return "All quiet on my side, just watching the settlement feeds in case anything drifts."
        if re.search(r"purpose|do you do", lowered):
            return "I'm here to translate your questions into settlement insights and highlight compliance signals."
        cleaned = lowered.strip(" ?!.")
        if cleaned:
            return "Appreciate the check-in—ready whenever you want to jump back into the data."
        return "Good to hear from you—I'm standing by."

    def _smalltalk_pivot(self, ctx: TemplateContext) -> str:
        focus = ctx.domain_focus or "the project"
        return f"When you're ready, just drop a plate ID or time window and we'll look at how {focus} is tracking."

    def _sanitize_focus_topic(self, text: Optional[str]) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"\s+", " ", text.strip())
        cleaned = cleaned.strip("\"'")

        smalltalk = {"hi", "hey", "hello", "thanks", "thank you", "ok", "okay", "yo", "hey there"}
        lowered = cleaned.lower()
        if not cleaned or lowered in smalltalk or len(cleaned) < 4:
            return ""
        if len(cleaned.split()) == 1 and len(cleaned) < 10:
            return ""
        if len(cleaned) > 160:
            cleaned = cleaned[:157].rstrip() + "..."
        return cleaned

    def _build_focus_sentence(self, topic: str, prefix: str) -> str:
        if not topic:
            return ""
        prefix_clean = prefix.rstrip(". ")
        return f'{prefix_clean} "{topic}".'

    def _ensure_sentence(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""
        if stripped[-1] not in ".!?":
            stripped += "."
        return stripped

    def _push_knowledge(self, lines: List[str]) -> None:
        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue
            if self._knowledge and trimmed == self._knowledge[-1]:
                continue
            self._knowledge.append(trimmed)

    def _knowledge_prompt(self) -> str:
        if not self._knowledge:
            return ""
        recent = list(self._knowledge)[-2:]
        if len(recent) == 1:
            sentence = f"Recent fact: {recent[0]}"
        else:
            sentence = "Recent focus points: " + "; ".join(recent)
        return self._ensure_sentence(sentence) + " "

    def _load_function_config(self) -> Dict[str, Any]:
        path = self._default_function_config_path()
        if not path or yaml is None:
            return {"functions": {}, "metadata": {}}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except Exception:
            return {"functions": {}, "metadata": {}}
        if not isinstance(data, dict):
            return {"functions": {}, "metadata": {}}
        return data

    def _extract_function_intents(self, config: Dict[str, Any]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        functions = config.get("functions", {})
        if not isinstance(functions, dict):
            return mapping
        for name, payload in functions.items():
            if isinstance(payload, dict) and payload.get("intent"):
                mapping[name] = str(payload["intent"])
        return mapping

    def _build_keyword_index(self, config: Dict[str, Any]) -> List[Tuple[str, str]]:
        index: List[Tuple[str, str]] = []
        functions = config.get("functions", {})
        if not isinstance(functions, dict):
            return index
        for payload in functions.values():
            if not isinstance(payload, dict):
                continue
            intent = str(payload.get("intent", "")).strip()
            keywords = payload.get("keywords", [])
            if not intent or not isinstance(keywords, list):
                continue
            for keyword in keywords:
                if isinstance(keyword, str) and keyword.strip():
                    index.append((keyword.strip().lower(), intent))
        index.sort(key=lambda item: len(item[0]), reverse=True)
        return index

    def _match_keyword_intent(self, text: str) -> Optional[str]:
        lowered = text.lower()
        for phrase, intent in self._intent_keywords:
            if phrase in lowered:
                return intent
        return None

    def _resolve_intent(
        self,
        func_name: Optional[str],
        grammar_intent: str,
        keyword_intent: Optional[str],
        glossary_score: float,
    ) -> str:
        if func_name and func_name in self.function_intents:
            return self.function_intents[func_name]
        if glossary_score >= self._definition_threshold:
            return "definition"
        if keyword_intent and grammar_intent in {"conversation", "fallback"}:
            return keyword_intent
        if grammar_intent == "fallback" and keyword_intent:
            return keyword_intent
        if grammar_intent == "fallback":
            return "conversation"
        return grammar_intent

    def _map_state_to_dialog_act(self, state: str) -> str:
        if state in {"clarify", "investigate"}:
            return "question"
        if state == "goal_setting":
            return "directive"
        return "statement"

    def _default_function_config_path(self) -> Optional[Path]:
        base_dir = Path(__file__).resolve().parent.parent
        path = base_dir / "config" / "industry" / "function_intents.yaml"
        return path if path.exists() else None

    def _default_rules_path(self) -> Optional[str]:
        base_dir = Path(__file__).resolve().parent.parent
        path = base_dir / "config" / "industry" / "business_rules.yaml"
        return str(path) if path.exists() else None


__all__ = [
    "ResponseGenerator",
    "ResponsePayload",
    "TablePayload",
    "ChartPayload",
]
