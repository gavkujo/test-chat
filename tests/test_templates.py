import string

import pytest

from core.templates import DynamicTemplateManager, TemplateContext


_COMMON_FIELDS = {
    "greeting",
    "opening",
    "closing",
    "context_line",
    "support_line_sentence",
    "persona_clause",
    "knowledge_line",
    "hint_line",
    "focus_line",
    "next_step_sentence",
    "lead",
    "insight_sentence",
    "term",
    "definition",
    "smalltalk_opening",
    "smalltalk_followup",
    "smalltalk_closing",
}

_CATEGORY_FIELDS = {
    "data_insight": {"lead", "insight_sentence"},
    "definition": {"term", "definition"},
    "conversation": {"focus_line", "hint_line", "next_step_sentence"},
    "clarification": {"focus_line", "hint_line", "next_step_sentence"},
    "smalltalk": {"smalltalk_opening", "smalltalk_followup", "smalltalk_closing"},
}


@pytest.mark.parametrize("category", sorted(DynamicTemplateManager._DEFAULT_TEMPLATES.keys()))
def test_default_template_placeholders(category: str) -> None:
    formatter = string.Formatter()
    base_allowed = set(_COMMON_FIELDS)
    base_allowed.update(_CATEGORY_FIELDS.get(category, set()))
    for template in DynamicTemplateManager._DEFAULT_TEMPLATES[category]:
        fields = {field for _, field, _, _ in formatter.parse(template) if field}
        unknown = fields - base_allowed
        assert not unknown, f"Template for {category} uses unsupported placeholders: {unknown}"


@pytest.mark.parametrize(
    "category, ctx, kwargs",
    [
        ("data_insight", TemplateContext(), {"lead": "Headline", "insight_sentence": "Metrics look steady."}),
        ("definition", TemplateContext(), {"term": "DOC", "definition": "degree of consolidation."}),
        (
            "conversation",
            TemplateContext(),
            {
                "focus_line": "Let's concentrate on F3 plates.",
                "hint_line": "Share the latest readings if you can.",
                "next_step_sentence": "Shall we surface the map next?",
            },
        ),
        (
            "clarification",
            TemplateContext(),
            {
                "focus_line": "Let's lock in the target plates.",
                "hint_line": "Drop the IDs or timeframe you want me to use.",
                "next_step_sentence": "Once we confirm, I'll run the function.",
            },
        ),
        (
            "smalltalk",
            TemplateContext(dialog_state="smalltalk"),
            {
                "smalltalk_opening": "All calm on my side.",
                "smalltalk_followup": "Wave when you want the dashboards back.",
            },
        ),
        ("greeting", TemplateContext(), {}),
        ("fallback", TemplateContext(), {}),
    ],
)
def test_templates_render_without_key_errors(category: str, ctx: TemplateContext, kwargs: dict) -> None:
    manager = DynamicTemplateManager()
    template = manager.get_template(category, ctx)
    manager.render(template, **kwargs)
