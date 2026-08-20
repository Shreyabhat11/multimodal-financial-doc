# Multimodal Financial Document Understanding System

A production-oriented, end-to-end pipeline that reads financial documents (bank
statements, credit card statements, invoices, loan statements) using a
vision-language model, validates the extraction with deterministic arithmetic *and*
a multi-agent CrewAI review, and serves the result through a REST API and a
Streamlit dashboard.

Built as a portfolio project to demonstrate: **multimodal LLMs → LangGraph
orchestration → deterministic validation → agentic review → structured APIs →
persistence → evaluation → deployment.**

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design rationale and
[`PROGRESS.md`](PROGRESS.md) for a phase-by-phase build log with honesty notes about
what was and wasn't executed in this development environment. See
[`INTERVIEW_PREP.md`](INTERVIEW_PREP.md) for how to talk about this project and 20
worked interview Q&As. See [`SETUP_AND_DEPLOYMENT.md`](SETUP_AND_DEPLOYMENT.md) for
a practical run/deploy checklist and an explicit list of what must be changed
(secrets, keys, config) before this is safe to run for real.

---

## 1. Project Overview

Upload a multi-page PDF statement. The system:

1. Renders each page and reads it **visually** with a vision-language model (no
   OCR-first — VLMs understand tables, layout, and structure the way a human
   analyst does).
2. Merges the per-page extractions into one document.
3. Runs **deterministic Python arithmetic** — never an LLM — to check whether the
   opening balance, transactions, and closing balance actually reconcile.
4. Runs a **four-agent CrewAI crew** using ReAct-style tool calling to interpret and
   cross-check those deterministic results.
5. Computes a **document-level confidence score** combining model self-reports with
   the validation outcomes above.
6. Routes to `completed` or `needs_human_review` — a route that **cannot be
   overridden by an agent's opinion** if the deterministic checks failed. (This is
   proven, not just claimed — see `tests/integration/test_langgraph_pipeline.py::
   TestFinancialValidationCannotBeOverriddenByCrew`.)

## 2. Architecture

```
PDF upload
   │
   ▼
Document Preprocessor (PyMuPDF: PDF → page images, auto rotation correction, DPI/size normalization)
   │
   ▼
┌─────────────────────────── LangGraph StateGraph ───────────────────────────┐
│ load_document → preprocess_pages → extract_page_information               │
│   (per page: vision-first extraction → confidence check → OCR fallback)   │
│     → merge_page_results → normalize_transactions → validate_schema       │
│       → financial_validation (deterministic) → anomaly_detection          │
│         → crew_validation (CrewAI, 4 agents, ReAct tools)                 │
│           → confidence_scoring → [route] → finalize_result | human_review │
└──────────────────────────────────────────────────────────────────────────┘
   │
   ▼
PostgreSQL  ◄──►  FastAPI  ◄──►  Streamlit
```

Full rationale for every design decision — why LangGraph wraps CrewAI rather than the
reverse, why balance math is never delegated to an LLM, why the VLM abstraction
exists — is in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 3. Features

- Multimodal extraction via **Qwen2-VL** or **LLaVA-NeXT**, swappable through one
  config value (`MODEL_PROVIDER`), each with local-GPU and Hugging-Face-Inference-API
  backends
- Vision-first extraction with **automatic OCR fallback** for low-confidence pages
- **Deterministic financial validation**: balance reconciliation, duplicate
  detection, date validation, running-balance consistency, outlier detection
- **CrewAI four-agent validation crew** (Extraction Validator, Financial Validator,
  Anomaly Analyst, Final Reviewer) using **ReAct tool-calling**, never exposing raw
  chain-of-thought to the application
- **Document-level confidence scoring** combining model self-report with
  deterministic outcomes
- Human-review routing that deterministic checks can force **regardless of agent
  opinion**
- REST API (FastAPI) with async background processing, structured error responses
- Streamlit dashboard: upload, live status polling, transactions table, validation
  results, confidence visualization, JSON download
- PostgreSQL persistence via SQLAlchemy 2.0 + Alembic, with **account numbers
  encrypted at rest** (Fernet) and masked in every API/UI response
- Evaluation framework: field-level P/R/F1, tolerance-based numeric accuracy,
  transaction-level matching, schema validity, Qwen-VL vs. LLaVA vs. OCR-baseline
  comparison
- 81 passing pytest tests, 83% coverage on `app/`
- Docker + Docker Compose, GPU-optional

## 4. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Vision-Language Models | Qwen2-VL-7B-Instruct, LLaVA-v1.6-Mistral-7B (via `transformers` or HF Inference API) |
| OCR fallback | Tesseract (`pytesseract`) |
| Orchestration | LangGraph 1.2.x |
| Multi-agent validation | CrewAI 1.15.x |
| API | FastAPI 0.141.x |
| Database | PostgreSQL, SQLAlchemy 2.0, Alembic |
| Frontend | Streamlit |
| Document processing | PyMuPDF, Pillow |
| Testing | pytest |
| Containerization | Docker, Docker Compose |

## 5. Installation

```bash
git clone <this-repo>
cd multimodal-financial-doc-ai
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Only if you plan to run Qwen-VL/LLaVA locally on a GPU:
pip install -r requirements-local-inference.txt
# For running tests:
pip install -r requirements-dev.txt
```

System dependency: **Tesseract OCR** must be installed for the OCR-fallback path.

```bash
# Debian/Ubuntu
sudo apt-get install tesseract-ocr
# macOS
brew install tesseract
```

## 6. Environment Variables

Copy `.env.example` to `.env` and fill in real values:

```bash
cp .env.example .env
```

Key variables (full list, with defaults and explanations, is in `.env.example`):

| Variable | Purpose |
|---|---|
| `MODEL_PROVIDER` | `qwen-vl` \| `llava` \| `hf-inference` (default — no local GPU needed) |
| `MODEL_NAME` | HF model id, e.g. `Qwen/Qwen2-VL-7B-Instruct` |
| `HF_TOKEN` | Required for `hf-inference` and gated model downloads |
| `MODEL_LOAD_IN_4BIT` | 4-bit quantization for local GPU inference |
| `ANTHROPIC_API_KEY` | Powers the CrewAI validation crew's reasoning LLM |
| `DATABASE_URL` | PostgreSQL connection string |
| `SECRET_KEY` | Derives the account-number encryption key — treat as a real secret |
| `CONFIDENCE_THRESHOLD` | Document-level confidence below which human review is triggered |
| `MAX_FILE_SIZE_MB`, `MAX_PAGES` | Upload limits |
| `CREWAI_DISABLE_TELEMETRY` | Set `true` for offline/CI/sandboxed environments |

## 7. Model Setup

**Default (no GPU required):** `MODEL_PROVIDER=hf-inference` with a valid `HF_TOKEN`
calls Qwen2-VL over the Hugging Face Inference API.

**Local GPU inference:**

| Model | bf16 VRAM | 4-bit VRAM |
|---|---|---|
| Qwen2-VL-7B-Instruct | ~16GB | ~6-7GB |
| LLaVA-v1.6-Mistral-7B | ~15-16GB | ~7-8GB |

Set `MODEL_PROVIDER=qwen-vl` (or `llava`), `MODEL_DEVICE=cuda`, install
`requirements-local-inference.txt`. CPU-only local inference works but is not
practical beyond a single-page smoke test — use `hf-inference` instead.

## 8. Running Locally

```bash
# 1. Start PostgreSQL (or point DATABASE_URL at an existing instance)
docker run -d -p 5432:5432 -e POSTGRES_USER=findoc_user -e POSTGRES_PASSWORD=findoc_pass \
  -e POSTGRES_DB=findoc_db postgres:16-alpine

# 2. Run migrations
alembic upgrade head

# 3. Start the API
uvicorn app.main:app --reload --port 8000

# 4. In a second terminal, start the UI
streamlit run frontend/streamlit_app.py
```

API docs: `http://localhost:8000/docs` · UI: `http://localhost:8501`

## 9. Running with Docker

```bash
cp .env.example .env   # fill in ANTHROPIC_API_KEY at minimum
docker compose up --build
```

This starts PostgreSQL, the API (`:8000`), and Streamlit (`:8501`), runs migrations
automatically on API container start, and persists uploaded files + Postgres data in
named volumes.

For local GPU inference:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build \
  --build-arg INSTALL_LOCAL_INFERENCE=true api
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

## 10. API Usage

```bash
# Upload
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@data/samples/clean_statement.pdf"
# -> {"document_id": "...", "status": "uploaded", "message": "..."}

# Poll status
curl http://localhost:8000/documents/{document_id}/status

# Get full result once status is "completed" or "needs_human_review"
curl http://localhost:8000/documents/{document_id}/result

# Reprocess (re-runs the pipeline against the originally uploaded file)
curl -X POST http://localhost:8000/documents/{document_id}/reprocess

# Health check
curl http://localhost:8000/health
```

Full endpoint list and request/response schemas: `http://localhost:8000/docs`
(FastAPI's auto-generated Swagger UI).

## 11. Streamlit Usage

1. Open `http://localhost:8501`.
2. Upload a PDF — a local page-preview thumbnail renders immediately (rendered
   client-side, before upload, reusing `app.document_processing.pdf_loader`).
3. Click **Start Processing** — the app polls status until a terminal state.
4. Browse results across tabs: Account & Totals, Transactions, Validation,
   Confidence, Raw JSON (with a download button).
5. Use **Reprocess** to re-run the pipeline against the same uploaded file (e.g.
   after changing `CONFIDENCE_THRESHOLD`), or enter an existing Document ID in the
   sidebar to reload prior results.

## 12. Example Input / Output

Real generated sample documents live in `data/samples/` (see
`evaluation/dataset.py` — these are the same PDFs the evaluation harness uses).

`data/samples/clean_statement.ground_truth.json`:

```json
{
  "account_holder": "Jane Doe",
  "account_number": "9988776655",
  "bank_name": "First National",
  "opening_balance": 1000.0,
  "closing_balance": 1300.0,
  "currency": "USD",
  "statement_period": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
  "transactions": [
    {"date": "2024-01-05", "description": "Salary", "debit": 0, "credit": 500.0},
    {"date": "2024-01-10", "description": "Groceries", "debit": 200.0, "credit": 0}
  ]
}
```

The API's `GET /documents/{id}/result` response wraps this in the full
`FinalExtractionResult` shape, with the account number masked
(`"account_number": "******6655"`) and `validation_results`/`overall_confidence`
attached — see `app/schemas/extraction_result.py` for the exact schema.

## 13. LangGraph Workflow

```
load_document → preprocess_pages → extract_page_information
  → merge_page_results → normalize_transactions → validate_schema
    → financial_validation → anomaly_detection → crew_validation
      → confidence_scoring → [finalize_result | human_review]
```

Conditional edges: preprocessing/extraction failures short-circuit to `FAILED`
without ever calling the VLM further; a page-extraction failure rate over 50%
triggers exactly one retry of the whole extraction pass; schema-invalid documents
route straight to `human_review`; the final routing decision after
`confidence_scoring` triggers `human_review` if **either** any validator reported
`FAILED` **or** confidence is below threshold — proven in
`tests/integration/test_langgraph_pipeline.py`.

## 14. CrewAI Architecture

Four agents, `Process.sequential`, each scoped to only the tools it needs:

- **Extraction Validator** — checks for missing/malformed fields, no tools (reasons
  from the document summary alone)
- **Financial Validator** — `calculate_total`, `calculate_balance`,
  `check_reported_totals`
- **Anomaly Analyst** — `check_duplicate_transactions`,
  `check_transaction_consistency`, `validate_dates`, `detect_anomalies`
- **Final Reviewer** — synthesizes all prior findings into one
  `output_pydantic`-enforced structured verdict

Every tool is a thin wrapper over `app/validation/*` (Phase 8's deterministic
functions) — agents interpret arithmetic results, they never perform arithmetic
themselves.

## 15. ReAct Implementation

`app/agents/tools.py`'s 7 tools are built via a per-document closure factory
(`build_validation_tools`), so agent tool calls are effectively argument-free — the
document context is already bound, keeping every tool call's payload small
regardless of transaction count. Tools return compact JSON, never free-text
reasoning; the Final Reviewer's task uses `output_pydantic=CrewValidationOutput` so
CrewAI itself enforces the structured output contract, and `crew_verbose=False` in
production ensures no intermediate reasoning trace ever reaches application logs or
API responses.

## 16. Evaluation Methodology

```bash
python -c "
from evaluation.dataset import build_synthetic_dataset
from evaluation.model_comparison import run_comparison
from app.extraction.qwen_vl_model import QwenVLModel
from app.extraction.llava_model import LLaVAModel

samples = build_synthetic_dataset()
models = {
    'qwen-vl': QwenVLModel(backend='hf-inference'),
    'llava': LLaVAModel(backend='hf-inference'),
}
report = run_comparison(samples, models)
print(report.to_markdown())
"
```

Metrics: field-level precision/recall/F1/exact-match, tolerance-based numeric
accuracy, transaction-level P/R/F1 (with same-day/same-amount disambiguation via
description similarity), schema validity rate. See `evaluation/metrics.py` and
`tests/unit` for the exact semantics, verified against hand-constructed cases
including deliberately corrupted extractions to confirm the metrics actually
distinguish model quality (not just report plausible-looking numbers).

## 17. Testing

```bash
pytest                                       # all 81 tests
pytest --cov=app --cov-report=term-missing   # with coverage (83% on app/)
pytest tests/unit                            # fast, no I/O
pytest tests/integration                     # real SQLite, real HTTP, real LangGraph execution
```

See [`tests/README.md`](tests/README.md) for the full brief-requirement-to-test
mapping and an honest breakdown of coverage gaps (real network/GPU call bodies are
deliberately never exercised by this network-free suite).

## 18. Limitations

- VLM extraction quality on real (non-synthetic) scanned documents has not been
  benchmarked in this environment — no GPU/API access was available during
  development. The evaluation framework is real and correct (verified with fake
  backends of deliberately varying quality); running it against real models and
  real statements is the natural next step.
- OCR fallback structuring depends on a text-completion LLM call, adding latency;
  for consistently low-quality scans, a dedicated OCR-tuned prompt or model might
  outperform the current generic structuring prompt.
- Local-file storage (`app/services/file_storage.py`) is single-instance —
  multi-replica API deployments need S3/GCS/Azure Blob instead (the module's narrow
  interface makes this a contained change).
- No authentication/authorization layer — every endpoint is open. A real deployment
  needs API keys or OAuth2 in front of this.
- Confidence-weight defaults (`configs/config.yaml`) are reasonable priors, not
  empirically tuned against a real labeled dataset.

## 19. Security Considerations

- Account numbers are **encrypted at rest** (Fernet, key derived from `SECRET_KEY`)
  and **masked in every API response and UI view** — verified in
  `tests/integration/test_database.py` that the stored ciphertext contains no trace
  of the plaintext.
- `SECRET_KEY` must be a real secret in production — the `.env.example` default is
  explicitly insecure and flagged as such.
- No sensitive data is ever logged (masking utilities in `app/schemas/parsing.py`;
  CrewAI's raw reasoning traces are never returned by the application, only the
  final structured verdict).
- CORS origins are explicitly configured (`API_CORS_ORIGINS`), not wildcarded.
- Docker images run as a non-root user.
- `CorruptedDocumentError`/`FileTooLargeError`/etc. are typed exceptions with
  bounded, generic messages to API clients — internal exception detail never leaks
  over HTTP (see `app/api/exception_handlers.py`'s catch-all handler).

## 20. Future Improvements

- Real-model evaluation against a larger, ideally partially-real (de-identified)
  document set
- Authentication/authorization
- Object storage (S3/GCS) for uploaded files instead of local disk
- A dedicated `/health/deep` endpoint checking DB/VLM connectivity
- Batched multi-page VLM calls where the backend supports it, to reduce per-document
  latency
- Human-review UI (currently the Streamlit app displays `needs_human_review`
  documents but has no dedicated correction/approval workflow)
- CI pipeline running the pytest suite + a `docker build` check on every PR

---

For how to talk about this system's design decisions in an interview setting, plus
20 worked Q&As, see [`INTERVIEW_PREP.md`](INTERVIEW_PREP.md).
