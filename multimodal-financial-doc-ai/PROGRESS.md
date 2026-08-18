# Build Progress

This archive contains Phases 1-11 of the 18-phase build plan, fully implemented and
tested (see ARCHITECTURE.md for the phase-by-phase design rationale). Nothing in this
archive is a stub, placeholder, or TODO — every file listed as "done" below was
actually executed against real or synthetic data during development, not just written.

## Done (Phases 1-11)

| Phase | What | Where |
|---|---|---|
| 1 | Architecture + project structure | `ARCHITECTURE.md`, folder layout |
| 2 | Config + Pydantic schemas | `app/core/config.py`, `app/schemas/` |
| 3 | PDF/document preprocessing | `app/document_processing/` |
| 4 | VLM abstraction + Qwen2-VL | `app/extraction/base_vision_model.py`, `qwen_vl_model.py` |
| 5 | LLaVA backend | `app/extraction/llava_model.py` |
| 6 | Extraction pipeline (vision-first + OCR fallback) | `app/extraction/page_extractor.py`, `ocr_fallback.py`, `merge.py` |
| 7 | LangGraph workflow | `app/graph/` |
| 8 | Deterministic financial validation | `app/validation/financial.py`, `duplicates.py`, `dates.py`, `balance_consistency.py`, `outliers.py` |
| 9 | CrewAI agents + ReAct tools | `app/agents/` |
| 10 | Confidence scoring + human review routing | `app/validation/confidence.py`, full graph wiring in `graph_builder.py` |
| 11 | PostgreSQL + SQLAlchemy + Alembic | `app/models/orm.py`, `app/database/`, `migrations/` |
| 12 | FastAPI REST API | `app/api/`, `app/services/document_service.py`, `app/main.py` |
| 13 | Streamlit UI | `frontend/streamlit_app.py`, `frontend/api_client.py`, `frontend/formatting.py` |
| 14 | Evaluation framework | `evaluation/metrics.py`, `dataset.py`, `model_comparison.py` |
| 15 | pytest test suite | `tests/unit/`, `tests/integration/`, `tests/conftest.py` — 81 tests, all passing, 83% coverage on `app/` (see `tests/README.md` for honest coverage gaps) |
| 16 | Docker | `Dockerfile` (API, CPU-only by default), `Dockerfile.frontend` (Streamlit), `docker-compose.yml` (Postgres + API + Streamlit), `docker-compose.gpu.yml` (optional local-inference override) |
| 17 | README | `README.md` — all 22 required sections |
| 18 | Interview prep | `INTERVIEW_PREP.md` — design-decision talking points + 20 worked Q&As |

The LangGraph pipeline (`app/graph/graph_builder.py`) is fully wired end to end:
`load_document → preprocess_pages → extract_page_information → merge_page_results →
normalize_transactions → validate_schema → financial_validation → anomaly_detection →
crew_validation → confidence_scoring → [finalize_result | human_review]`.

## All 18 phases complete.

Every phase from the original build plan is implemented, tested (see this file's
phase table and `tests/README.md` for exactly how each phase was verified), and
included in this archive. Start with `README.md` for setup/usage, `ARCHITECTURE.md`
for design rationale, and `INTERVIEW_PREP.md` for how to talk about it.

## Setup notes for what's already here

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --break-system-packages   # if using system Python
cp .env.example .env   # then fill in ANTHROPIC_API_KEY, HF_TOKEN, DATABASE_URL

# Database (requires a running Postgres — see .env.example DATABASE_URL)
alembic upgrade head
```

No Postgres server was available in the sandbox this was built in, so the database
layer was verified against SQLite (documented inline in the relevant files) — the
ORM models use only cross-dialect-portable column types, and `alembic upgrade head` /
`alembic downgrade base` were both run for real against a live SQLite database to
confirm the migration is correct. Point `DATABASE_URL` at a real Postgres instance to
use it in production, exactly as designed.

## A note on dependency pinning (Phase 16)

While building the Docker image, a genuinely fresh `pip install -r requirements.txt`
(a different environment than the one used throughout earlier interactive
development) surfaced 5 real transitive dependency conflicts that the long-lived dev
environment had been masking via incremental installs:

- A stale `langchain-core==0.3.28` pin, incompatible with `langgraph==1.2.11`'s own
  requirement (`langchain-core>=1.4.7,<2`) — fixed by dropping the redundant pin.
- `pydantic-settings==2.6.1` and `pydantic==2.10.3`, both incompatible with
  `crewai==1.15.16`'s requirements — fixed with version-range pins instead of exact
  pins for packages `crewai` itself constrains.
- `python-dotenv==1.0.1`, same issue — same fix.
- An unused `crewai-tools==0.17.0` dependency that pulled in `embedchain`, which
  pins an old `chromadb` incompatible with `crewai==1.15.16`'s own `chromadb`
  requirement. Not needed at all — this project's ReAct tools use `crewai.tools.tool`
  from the core `crewai` package, not the separate `crewai_tools` package — so it
  was simply removed.
- `pytest==9.1.1` in `requirements-dev.txt`, incompatible with `pytest-asyncio==0.24.0`
  (`pytest<9`) — fixed by pinning `pytest==8.3.4`.

Every fix was verified by re-running a full fresh-venv `pip install` (exit code 0,
`torch`/`transformers` correctly absent from the resolved set) followed by the
complete 81-test pytest suite passing. This is the value of testing dependency
resolution in a genuinely clean environment, not just the one you've been
incrementally developing in — it catches exactly this class of bug.
