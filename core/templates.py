"""Dynamic template system for conversational and insight responses.

This module keeps things lightweight by using JSON template banks when present,
but also ships with reasonable fallbacks so the assistant can operate without
external assets. Templates are chosen with small variability tweaks to avoid
robotic phrasing while still staying fully deterministic for the given inputs.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Small pool of phrasings used to inject variation into otherwise static
# template strings. This keeps responses feeling natural without relying on
# heavy-weight generation models.
_INTENSIFIERS = [
    "notably",
    "worth flagging",
    "keep in mind",
    "it's good to note",
    "just so it's crystal clear",
    "for quick visibility",
    "flagging this for awareness",
    "worth bookmarking",
    "front of mind",
    "parking this on the watchlist",
    "good to have on the radar",
    "for easy recall",
    "just to anchor the context",
    "calling this out explicitly",
    "keeping this in the short-term memory",
]
_GREETINGS = [
    "Hi",
    "Hello",
    "Hey there",
    "Good to see you",
    "Welcome back",
    "Great to see you again",
    "Hi again",
    "Hello there",
    "Hey, ready to dive in?",
    "Good to have you back",
    "Morning",
    "Good afternoon",
    "Good evening",
    "Hey team",
    "Hiya",
]
_CLOSINGS = [
    "Let me know if you want to dig deeper.",
    "Happy to expand on any of this.",
    "Feel free to ask for more detail if needed.",
    "We can dig into plate-by-plate behaviour whenever you're ready.",
    "Ping me with the next angle you want to explore and I'll line it up.",
    "Shout if you want a breakout on any KPI.",
    "If you need more numbers, just say so.",
    "Tap me when you want the next slice.",
    "I can pull comparisons whenever you're set.",
    "Ready to unpack any of this further on request.",
    "Wave me over if you want supporting charts.",
    "Happy to line up the next drilldown when you are.",
    "Give me a nod if you want to pivot topics.",
    "I can run a deeper compliance check any time.",
    "Let's keep iterating whenever you need.",
]
_PROJECT_FOCUS = [
    "I'm keeping the {domain_focus} consolidation checkpoints in view so compliance stays on schedule.",
    "I’m lining everything up against the {domain_focus} timeline to keep the surcharge story tidy.",
    "Every turn stays anchored to the {domain_focus} readiness milestones we’re driving toward.",
    "All signals stay referenced against the {domain_focus} delivery gates.",
    "I'm mapping pivots back to the {domain_focus} punch list.",
    "Our guardrails remain the {domain_focus} compliance markers.",
    "I'm syncing the insight stream with the {domain_focus} milestones.",
    "Each recommendation tracks the {domain_focus} performance indicators.",
    "We are pacing everything to the {domain_focus} schedule board.",
    "Context is locked to the {domain_focus} control envelope.",
    "I am threading in {domain_focus} KPIs so nothing drops.",
    "Every flag ties back to the {domain_focus} readiness scoreboard.",
]
_SUPPORT_LINES = [
    "If you drop plate IDs or a date window, I’ll surface the freshest behaviour and flag anything off-spec.",
    "Say the word and I’ll tee up the relevant charts, PDFs, or compliance calls before anything slips.",
    "I'm ready to thread in the instrumentation highlights so nothing falls out of the programme cadence.",
    "Need benchmarks? I can line up historical baselines on request.",
    "I can ping the settlement summaries the moment you need them.",
    "Want the raw feed? I'll stream the latest readings right away.",
    "If you need filters applied, I can reshape the dataset on the fly.",
    "Ask and I'll bundle the references into a shareable packet.",
    "I can prep a quick side-by-side comparison any time.",
    "Need escalation ready? I'll craft the bullet points.",
    "Drop a threshold and I'll watch for breaches in real time.",
    "Say when and I'll spin up the PDF artifacts for distribution.",
]
_SMALLTALK_OPENINGS = [
    "All good on my end—just keeping the dashboards steady.",
    "Doing fine here, still watching the Tuas data feeds.",
    "Everything's smooth over here—systems are ticking along.",
    "Still on standby with the latest logs ready.",
    "Nothing wild my side, instruments are calm.",
    "Keeping an eye on the charts while you settle in.",
    "Holding the fort with the compliance board humming.",
    "Cash flow of data still healthy over here.",
    "No alarms tripped; just cruising through routine checks.",
    "Enjoying the quiet hum of the telemetry room.",
    "Ready whenever you want to nudge the next data point.",
]
_SMALLTALK_FOLLOWUPS = [
    "When you feel like diving back into the details, just point me at a plate or KPI.",
    "Happy to chat—drop a plate ID or question whenever you want to switch gears.",
    "Ready when you are; say the word and we can pivot straight to the settlement story.",
    "If curiosity strikes, toss me the angle and I'll pick it up.",
    "Whenever you want numbers, I've got the spreadsheets primed.",
    "When the next checkpoint calls, I'll line up the supporting stats.",
    "If you need a Statuto refresher, just flag it.",
    "We can swap back to instrumentation mode whenever you like.",
    "If the agenda shifts, I can pivot the context on the fly.",
    "Happy to keep bantering till you summon the analytics.",
    "Give me a cue and I'll swing us back to compliance mode.",
]
_SMALLTALK_CLOSINGS = [
    "Whenever you want to look at the data, I'm right here.",
    "Just let me know when it's time to get analytical again.",
    "Shout when you want to dig back into the instrumentation details.",
    "Glad it helped, just ping me when you want to dive back in.",
    "Happy to keep things steady; drop the next question whenever you're ready.",
    "I'll be here keeping the dashboards tidy in the meantime.",
    "Catch you when the next compliance check-in pops up.",
    "I'll stay on standby till the next agenda item pings.",
    "Enjoy the breather; I'll guard the data thread.",
    "I'll keep the telemetry warm for whenever you swing back.",
    "Take your time; I'll keep notes ready.",
    "I'll keep the map slices cached for the next round.",
    "Ping me when you want deeper slices teed up.",
    "I'll hold position until the next task surfaces.",
    "Keeping the cadence warm till you're back in data mode.",
]
_OPENINGS_EARLY = [
    "Great to connect on the Tuas datasets",
    "Looking forward to navigating this instrumentation journey with you",
    "Ready to map out the settlement story with you",
    "Let's get the foundations of the data story lined up",
    "Excited to set the tone for the assurance pathway",
    "Fresh session, clean slate—let's chart the signals",
    "Ready to sync on the early compliance outlook",
    "Let's anchor the opening moves on solid telemetry",
    "Pumped to kick off the data trench together",
    "Starting strong with the latest readings on tap",
    "Let's set the agenda for a crisp analytics run",
]
_OPENINGS_MID = [
    "Thanks for keeping the momentum going",
    "Appreciate the steady flow of context",
    "Continuing right where we left off",
    "Settling back into the groove of this data thread",
    "Momentum feels good—let's keep threading insights",
    "Keeping the cadence lively as we scan the next metrics",
    "Right back into the mix with the latest checkpoints",
    "Happy to stay on beat with your follow-up",
    "Building on the earlier diagnostics without missing a beat",
    "Keeping the analysis torch lit while you steer",
    "Ready to stitch the next chapter onto our running log",
]
_OPENINGS_LATE = [
    "Still right by your side as we round out the next checkpoints",
    "Keeping the cadence tight as we move through the remaining queries",
    "Locked in on closing any remaining loops",
    "Here to help tie a bow on the final threads",
    "Let's land the outstanding signals cleanly",
    "Ready to tuck in any loose compliance ends",
    "Sticking with you through the final settlement sweep",
    "Keeping the dashboards aligned till we sign off",
    "Focused on wrapping the agenda without drift",
    "Ready to archive the session once you're satisfied",
    "Standing by for any last clarifications before we close",
]

_TONE_GREETINGS: Dict[str, List[str]] = {
    "professional": [
        "Hello again",
        "Good afternoon",
        "Good evening",
        "Greetings",
        "Welcome, let's get to it",
    ],
    "empathetic": [
        "Hi there, hope you're doing well",
        "Hello, I'm right here with you",
        "Hey there, let's take this at your pace",
        "Hi, ready when you are",
        "Hello, let's keep things steady",
    ],
    "casual": [
        "Hey hey",
        "Yo",
        "Hey team",
        "Hiya, ready to roll",
        "Hey, let's dive in",
    ],
}

_TONE_CLOSINGS: Dict[str, List[str]] = {
    "professional": [
        "Let me know when you want the next checkpoint lined up.",
        "Reach out if you'd like a compliance summary drafted.",
        "Happy to prep the next briefing whenever required.",
    ],
    "empathetic": [
        "I'm here if you'd like to unpack anything further.",
        "Just say the word if you want to pause or go deeper.",
        "Reach out if you need a calmer walkthrough of the numbers.",
    ],
    "casual": [
        "Ping me when you want the next slice.",
        "Give me a shout if you want more stats.",
        "Catch me when you want to dive back in.",
    ],
}

_TONE_SUPPORT_LINES: Dict[str, List[str]] = {
    "professional": [
        "I can brief the leadership deck once you sign off.",
        "Say the word and I'll prep the compliance memo.",
    ],
    "empathetic": [
        "If you'd prefer a slower walk-through, I'm ready.",
        "Let me know if you want me to keep an extra gentle eye on the alerts.",
    ],
    "casual": [
        "Want me to keep the charts warm while you check other things?",
        "Need a rapid-fire recap later? I can sling one over.",
    ],
}

_TONE_SMALLTALK_CLOSINGS: Dict[str, List[str]] = {
    "professional": [
        "I'll keep the records tidy until you're back.",
        "I'll standby with the dashboards refreshed.",
    ],
    "empathetic": [
        "Take your time; I'll keep the pace calm.",
        "I'll watch the dials and let you know if anything needs care.",
    ],
    "casual": [
        "I'll hang out here till you're ready to jam on the data again.",
        "I'll keep things chill over here until you tap back in.",
    ],
}


def _merge_with_base(primary: Iterable[str], fallback: Iterable[str]) -> List[str]:
    merged: List[str] = []
    for source in (primary, fallback):
        for item in source:
            if item not in merged:
                merged.append(item)
    return merged


@dataclass
class TemplateContext:
    """Lightweight context container used by the template engine."""

    data_pattern: Optional[str] = None
    sentiment: str = "neutral"
    persona: str = "default"
    tone: str = "professional"
    conversation_stage: str = "mid"
    conversation_summary: str = ""
    engagement: str = "normal"
    domain_focus: str = "Tuas Terminal Phase 2"
    focus_statement: str = ""
    focus_topic: str = ""
    knowledge_line: str = ""
    dialog_state: str = "clarify"
    focus_repeat: bool = False


class DynamicTemplateManager:
    """Template manager with basic variability injection.

    Templates are discovered from `template_dir` if available. Each JSON file
    should contain a list of strings. When the files are not found the manager
    falls back to packaged defaults defined in `_DEFAULT_TEMPLATES` below.
    """

    _DEFAULT_TEMPLATES: Dict[str, List[str]] = {
        "greeting": [
            "{greeting}! {opening}",
            "{greeting}! {opening}. Let's make this quick and clear.",
            "{greeting}! {opening}. Ready when you are.",
            "{greeting}! {opening}. I already have the dashboards open.",
            "{greeting}! {opening}—shall we dive straight in?",
            "{greeting}! {opening}. I'm primed to navigate the next data slice.",
            "{greeting}! {opening}; just point me at the next objective.",
            "{greeting}! {opening}. I'm keeping the instrumentation feed warm.",
            "{greeting}! {opening}. Let's keep the cadence tight.",
            "{greeting}! {opening}. I've queued the latest settlement notes.",
            "{greeting}! {opening}. Happy to keep momentum rolling."
        ],
        "clarification": [
            "{greeting}! {opening}. {context_line} {persona_clause}{knowledge_line}Before I lock anything in, {hint_line} {support_line_sentence} {next_step_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {focus_line} {persona_clause}{knowledge_line}Before we commit, {hint_line} {support_line_sentence} {next_step_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {focus_line} {persona_clause}{knowledge_line}Before I proceed, {hint_line} {support_line_sentence} {next_step_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {persona_clause}{knowledge_line}Before I finalise anything, {hint_line} {support_line_sentence} {next_step_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {focus_line} {persona_clause}{knowledge_line}Before we advance, {hint_line} {support_line_sentence} {next_step_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {focus_line} {persona_clause}{knowledge_line}Before signing off, {hint_line} {support_line_sentence} {next_step_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {persona_clause}{knowledge_line}Before I run the function, {hint_line} {support_line_sentence} {next_step_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {focus_line} {persona_clause}{knowledge_line}Before I commit parameters, {hint_line} {support_line_sentence} {next_step_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {persona_clause}{knowledge_line}Before triggering anything, {hint_line} {support_line_sentence} {next_step_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {focus_line} {persona_clause}{knowledge_line}Before we move on, {hint_line} {support_line_sentence} {next_step_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {focus_line} {persona_clause}{knowledge_line}Before I press ahead, {hint_line} {support_line_sentence} {next_step_sentence} {closing}"
        ],
        "data_insight": [
            "Headline insight: {insight_sentence} {closing}",
            "Key compliance signals: {insight_sentence} {closing}",
            "Operational snapshot → {insight_sentence} {closing}",
            "Impact assessment: {insight_sentence} {closing}",
            "{lead}: {insight_sentence} {closing}",
            "Focus pulse → {insight_sentence} {closing}",
            "Momentum check: {insight_sentence} {closing}",
            "Here's the quick sweep: {insight_sentence} {closing}",
            "Programme pulse: {insight_sentence} {closing}",
            "Risk radar update → {insight_sentence} {closing}",
            "Compliance radar: {insight_sentence} {closing}",
            "Signal brief: {insight_sentence} {closing}",
            "Instrumentation highlight: {insight_sentence} {closing}",
            "Quick scan → {insight_sentence} {closing}",
            "Trendline headline: {insight_sentence} {closing}",
            "Settling snapshot: {insight_sentence} {closing}",
            "KPI flash report: {insight_sentence} {closing}",
            "Tactical overview: {insight_sentence} {closing}",
            "Immediate readout: {insight_sentence} {closing}",
            "Momentum pulse: {insight_sentence} {closing}"
        ],
        "fallback": [
            "Sorry, I don't understand that yet. Could you rephrase or try another request?",
            "I'm not sure I follow. Please try again with a different phrasing.",
            "That phrasing missed my parser—mind restating it differently?",
            "I could not map that to a known task. Another angle might help.",
            "No clear match found. Could you give me a bit more detail?",
            "I'm drawing a blank on that one. Try rewording or adding context.",
            "That went past my current rule set. Let's try a clearer prompt.",
            "I need a little more structure in the request. Mind rephrasing?",
            "I couldn't detect the intent there. Can you spell it out another way?",
            "No dispatcher route matched. Could you specify what you need?",
            "That request is outside my playbook. A fresh phrasing might do it."
        ],
        "definition": [
            "{term} refers to {definition}. {closing}",
            "In this project, {term} means {definition}. {closing}",
            "{term}: {definition} {closing}",
            "We treat {term} as {definition}. {closing}",
            "Operationally, {term} covers {definition}. {closing}",
            "Around here, {term} maps to {definition}. {closing}",
            "Practically speaking, {term} is {definition}. {closing}",
            "For this engagement, {term} translates to {definition}. {closing}",
            "Whenever I mention {term}, assume {definition}. {closing}",
            "Documentation shorthand: {term} equals {definition}. {closing}",
            "Quick refresher—{term}: {definition}. {closing}"
        ],
        "conversation": [
            "{greeting}! {opening}. {context_line} {focus_line} {persona_clause}{knowledge_line}{hint_line} {support_line_sentence} {next_step_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {persona_clause}{knowledge_line}{hint_line} {support_line_sentence} {next_step_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {focus_line} {knowledge_line}{support_line_sentence} {next_step_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {focus_line} {persona_clause}{knowledge_line}{hint_line} {support_line_sentence} {closing} {next_step_sentence}",
            "{greeting}! {opening}. {context_line} {persona_clause}{knowledge_line}{hint_line} {support_line_sentence} {closing} {next_step_sentence}",
            "{greeting}! {opening}. {context_line} {focus_line} {persona_clause}{knowledge_line}{support_line_sentence} {next_step_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {persona_clause}{knowledge_line}{support_line_sentence} {next_step_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {focus_line} {persona_clause}{knowledge_line}{hint_line} {next_step_sentence} {support_line_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {persona_clause}{knowledge_line}{hint_line} {next_step_sentence} {support_line_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {focus_line} {knowledge_line}{hint_line} {support_line_sentence} {next_step_sentence} {closing}",
            "{greeting}! {opening}. {context_line} {persona_clause}{knowledge_line}{hint_line} {support_line_sentence} {closing} {next_step_sentence}",
            "{greeting}! {opening}. {context_line} {focus_line} {persona_clause}{knowledge_line}{hint_line} {closing} {support_line_sentence} {next_step_sentence}"
        ],
        "smalltalk": [
            "{greeting}! {smalltalk_opening} {smalltalk_followup} {closing}",
            "{greeting}! {smalltalk_opening} {closing}",
            "{greeting}! {smalltalk_opening} {smalltalk_followup}",
            "{greeting}! {smalltalk_opening} Got anything else on your mind?",
            "{greeting}! {smalltalk_opening} {smalltalk_followup} I'll be ready when you pivot back.",
            "{greeting}! {smalltalk_opening} Whenever you're ready, we can dive into data.",
            "{greeting}! {smalltalk_opening} {smalltalk_followup} Want to check the dashboards next?",
            "{greeting}! {smalltalk_opening} I can line up the latest readings any time.",
            "{greeting}! {smalltalk_opening} {smalltalk_followup} Just say when to switch gears.",
            "{greeting}! {smalltalk_opening} I'm keeping everything warm for the next task.",
            "{greeting}! {smalltalk_opening} {smalltalk_followup} Ready to toggle back to insights whenever you are."
        ],
    }

    def __init__(self, template_dir: Optional[str] = None, seed: int = 1337) -> None:
        self._base_path = Path(template_dir).expanduser() if template_dir else None
        self._rng = random.Random(seed)
        self.template_banks: Dict[str, List[str]] = {}
        self._last_choices: Dict[str, str] = {}
        self._load_templates()

    # Public API -----------------------------------------------------------------
    def get_template(
        self,
        category: str,
        context: Optional[TemplateContext] = None,
    ) -> str:
        """Return a template string for the given category.

        The selection is influenced by a few contextual signals but remains
        deterministic for the seeded RNG.
        """

        context = context or TemplateContext()
        bank = self.template_banks.get(category)
        if not bank:
            bank = self._DEFAULT_TEMPLATES.get(category, self._DEFAULT_TEMPLATES["fallback"])

        filtered = self._apply_context_filters(bank, context)
        chosen = self._rng.choice(filtered)
        return self._inject_variability(chosen, context)

    def render(self, template: str, **kwargs: Any) -> str:
        """Render a template using keyword arguments.

        Missing keys result in graceful fallbacks rather than hard errors so the
        assistant can still reply even when certain data points are unavailable.
        """

        safe_kwargs = {key: ("" if value is None else value) for key, value in kwargs.items()}
        try:
            return template.format(**safe_kwargs)
        except KeyError:
            return template

    # Internals ------------------------------------------------------------------
    def _load_templates(self) -> None:
        if not self._base_path or not self._base_path.exists():
            self.template_banks = dict(self._DEFAULT_TEMPLATES)
            return

        for name, default in self._DEFAULT_TEMPLATES.items():
            path = self._base_path / f"{name}_templates.json"
            if path.exists():
                try:
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list) and all(isinstance(item, str) for item in data):
                            self.template_banks[name] = data
                            continue
                except Exception:
                    pass
            self.template_banks[name] = list(default)

    def _apply_context_filters(self, templates: Iterable[str], context: TemplateContext) -> List[str]:
        """Apply light filtering based on sentiment/persona."""

        filtered = list(templates)
        if context.sentiment == "negative":
            filtered = [tpl for tpl in filtered if "reassure" not in tpl.lower()] or filtered
        requires_persona = context.persona not in {"default", "casual"}
        if requires_persona and getattr(context, "dialog_state", "") != "smalltalk":
            filtered = [tpl for tpl in filtered if "{persona_clause}" in tpl] or filtered
        return filtered

    def _inject_variability(self, template: str, context: TemplateContext) -> str:
        """Inject small lexical variations into the template."""

        class _SafeDict(dict):
            def __missing__(self, key: str) -> str:  # pragma: no cover - trivial
                return "{" + key + "}"

        tone = (context.tone or "professional").lower()
        greeting_choices = _merge_with_base(_TONE_GREETINGS.get(tone, []), _GREETINGS)
        if context.sentiment == "negative":
            greeting_choices = [item for item in greeting_choices if item.lower().startswith(("hi", "hello"))] or ["Hello", "Hi"]
        greeting = self._unique_choice(greeting_choices, "greeting")

        smalltalk_mode = getattr(context, "dialog_state", "") == "smalltalk"
        if smalltalk_mode:
            opening = self._unique_choice(_SMALLTALK_OPENINGS, "smalltalk_opening")
            context_line = ""
            support_line_sentence = self._unique_choice(_SMALLTALK_FOLLOWUPS, "smalltalk_followup")
            if support_line_sentence and support_line_sentence[-1] not in ".!?":
                support_line_sentence += "."
            smalltalk_closing_choices = _merge_with_base(
                _TONE_SMALLTALK_CLOSINGS.get(tone, []),
                _SMALLTALK_CLOSINGS,
            )
            closing = self._unique_choice(smalltalk_closing_choices, "smalltalk_closing")
            persona_clause = ""
            knowledge_line = ""
        else:
            if context.conversation_stage == "early":
                opening_pool = _OPENINGS_EARLY
            elif context.conversation_stage == "late":
                opening_pool = _OPENINGS_LATE
            else:
                opening_pool = _OPENINGS_MID
            opening = self._unique_choice(opening_pool, "opening")

            context_line = self._unique_choice(_PROJECT_FOCUS, "project_focus").format(
                domain_focus=context.domain_focus or "programme"
            )
            if context_line and context_line[-1] not in ".!?":
                context_line += "."

            support_choices = _merge_with_base(_TONE_SUPPORT_LINES.get(tone, []), _SUPPORT_LINES)
            support_line_sentence = self._unique_choice(support_choices, "support_line")
            if support_line_sentence and support_line_sentence[-1] not in ".!?":
                support_line_sentence += "."

            closing_choices = _merge_with_base(_TONE_CLOSINGS.get(tone, []), _CLOSINGS)
            closing = self._unique_choice(closing_choices, "closing")

            persona_clause = {
                "default": "I'll keep the conversation grounded in actionable, engineering-friendly detail.",
                "analytical": "I'll keep the lens analytical and tie our chat back to measurable compliance signals.",
                "empathetic": "I'll make sure we acknowledge any concerns before we dive into the metrics.",
                "casual": "Keeping it light—just say when you want to swing back to the instrumentation checks.",
            }.get(context.persona, "")
            if persona_clause and persona_clause[-1] not in ".!?":
                persona_clause += "."
            if persona_clause:
                persona_clause += " "
            knowledge_line = context.knowledge_line or ""

        substitutions = _SafeDict(
            greeting=greeting,
            closing=closing,
            opening=opening,
            context_line=context_line,
            support_line_sentence=support_line_sentence,
            persona_clause=persona_clause,
            knowledge_line=knowledge_line if not smalltalk_mode else "",
        )

        return template.format_map(substitutions)

    def _unique_choice(self, pool: Iterable[str], key: str) -> str:
        items = list(pool)
        if not items:
            return ""
        last = self._last_choices.get(key)
        candidates = [item for item in items if item != last] or items
        choice = self._rng.choice(candidates)
        self._last_choices[key] = choice
        return choice


__all__ = ["DynamicTemplateManager", "TemplateContext"]
