# Boskalis GeoChat Assistant

A deterministic, on-premise chatbot that surfaces settlement-plate intelligence without calling external LLM APIs. The assistant combines rule-aware intent modelling, deterministic parsing, and a template-driven response layer to deliver consistent, auditable guidance for Boskalis projects.

---

## System Architecture

```
┌──────────┐   user utterance   ┌────────────────┐   resolved intent   ┌────────────────────┐
│ Streamlit│ ─────────────────▶ │ Intent Model    │ ───────────────────▶│ Dispatcher         │
│ UI (app) │   dialog context   │ (MiniTransformer│   required slots    │ + Slot Extractors  │
└──────────┘                    │  + Grammar Rules)│                     └────────┬──────────┘
      ▲                         └────────────────┘                                │
      │ chat replies                                                         function call
      │                                                                        │
      │                          templated response + artefacts               ▼
┌───────────────┐ ◀──────────────────────────────────────────────────────── ┌──────────────────┐
│ Response Layer│                                                        │ Domain Functions │
│ (core/*)      │ ◀── insights, charts, recommendations ─────────────────── │ (functions.py,  │
└──────┬────────┘                                                          │ helpers/*)      │
       │                                                                    └────────────────┘
       │ business rules, templates, tone                                      ▲
       ▼                                                                      │ data connectors
┌────────────────────┐                                                        │
│ Config & Knowledge │◀─ YAML grammars, tone pools, overrides ────────────────┘
└────────────────────┘
```

### Core Components

- **Presentation Layer (`app.py`)** handles Streamlit session state, chat rendering, and download affordances for PDFs and charts.
- **Intent Layer (`core/intent_model.py`, `models/transformer.py`)** blends a MiniTransformer classifier with YAML-defined grammar rules and optional industry overrides to map utterances to intents, dialog states, and transition hints.
- **Dispatcher (`dispatcher.py`, `helpers/datasources.py`)** orchestrates slot extraction, parameter validation, and routing to the correct domain function. Missing slots raise typed exceptions that drive follow-up questions in the UI.
- **Domain Functions (`functions.py`, `helpers/*`)** encapsulate deterministic tasks such as Asaoka consolidation retrieval, PDF generation, settlement plotting, and reporting exports. Plotly and ReportLab produce artefacts saved under `static/`.
- **Response Layer (`core/conversation.py`, `core/response_generator.py`, `core/templates.py`)** assembles the final assistant reply. It merges intent outcomes, business rule highlights, tone-aware phrasing, and chart metadata into consistent chat turns.
- **Knowledge Configuration (`config/*`)** defines intent patterns, tone-aware template pools, glossary rules, and industry-specific business-rule thresholds used throughout the stack.

### Conversation Pipeline

1. **User Input** arrives via the Streamlit UI and is normalised into the `Conversation` history in `core/conversation.py`.
2. **Intent Resolution** leverages the MiniTransformer logits plus grammar priors from `config/universal/intent_patterns.yaml`, merged with any industry overrides.
3. **Slot Collection** uses deterministic parsers (`helpers/data_structures.py`, `data/parser_test.py`) to extract IDs, date ranges, or KPI names. When parsing fails, `dispatcher.py` requests clarifying input.
4. **Task Execution** invokes the mapped function in `functions.py`. Supporting modules query databases, assemble PDFs, or build Plotly figures.
5. **Business Context** from `core/business_rules.py` evaluates compliance thresholds and produces structured insights.
6. **Response Composition** in `core/response_generator.py` blends templates, tone, persona, and artefacts into a multi-part chat message ready for Streamlit.

---

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <link>
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the Streamlit chat**
   ```bash
   streamlit run app.py
   ```

Generated PDFs, charts, and CSV exports will appear inside `static/` once functions execute.

---

## Key Modules at a Glance

- `core/templates.py` – Tone-aware template manager with persona/tone pools and placeholder validation tests in `tests/test_templates.py`.
- `core/response_generator.py` – Builds intros, insights, charts, and follow-ups using deterministic randomisation for variability.
- `core/business_rules.py` – Evaluates domain-specific thresholds and attaches recommendations to insight payloads.
- `helpers/` – Data adapters (`datasources.py`), PDF/report builders (`reporter.py`), and settlement analysis tooling (`asaoka.py`, `settlement_data.py`).
- `models/transformer.py` & `tokenizer/` – MiniTransformer classifier and SentencePiece assets.
- `config/` – Universal and industry-scoped YAML definitions for intents, business rules, and glossary concepts.

---

## Directory Overview

- `app.py` – Streamlit entry point, session state, and chat I/O wiring.
- `dispatcher.py` – Bridges parsed intents to domain functions and surfaces missing-slot prompts.
- `functions.py` – Deterministic settlement workflows (Asaoka retrieval, PDFs, plots, overviews).
- `infer.py` – Lightweight harness for probing the saved classifier checkpoint.
- `llm_main.py` / `main.py` – Orchestrate classifier inference, rule heuristics, and router glue.
- `core/` – Conversation engine, templates, business rules, statistics, and retrieval helpers underpinning responses.
- `helpers/` – Data connectors, PDF/report tooling, and settlement analytics utilities (`asaoka.py`, `reporter.py`, etc.).
- `data/` – Parser fixtures and slot-extraction utilities used by the dispatcher.
- `models/` – MiniTransformer implementation and model utilities.
- `config/` – YAML intent grammars, template pools, and industry overrides.
- `static/` – Generated artefacts (PDFs, charts, CSV snapshots) exposed via Streamlit downloads.
- `saved/` – Runtime checkpoints such as `best_model.pt` for the intent model.
- `tests/` – Regression checks with emphasis on template placeholder validation.
- `tokenizer/` – SentencePiece vocabulary model paired with the classifier.

---

## Development Tips

- The production classifier checkpoint lives in `saved/best_model.pt`; load it via `infer.py` for dry runs.
- Run `pytest tests/test_templates.py` to safeguard template placeholder integrity after editing phrasing pools.
- Use `python3 -m compileall core/` to catch syntax regressions quickly during refactors.

---

## Contact

For questions, bug reports, or collaboration, reach out to `garv.sachdev@boskalis.com`.
