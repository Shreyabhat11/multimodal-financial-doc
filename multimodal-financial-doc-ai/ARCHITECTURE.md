# Architecture — Multimodal Financial Document Understanding System

## 1. Revised Pipeline (and why it differs slightly from the brief)

Your proposed flow was:

```
PDF → VLM → LangGraph → CrewAI validation → JSON
```

I'm keeping that shape but making two structural changes, both deliberate:

**(a) LangGraph wraps everything, including the CrewAI call — not the other way around.**
CrewAI's "Crew" abstraction is good at *role-based collaboration* (a small team of
agents jointly working a task) but it is not a durable, resumable state machine. LangGraph
*is*. So LangGraph owns the state (`DocumentState`), the retries, the conditional routing
(low confidence → human review, validation failure → re-extraction), and checkpointing.
CrewAI is invoked as a single node (`crew_validation`) inside that graph — it's a
specialist tool the graph calls, not the top-level orchestrator. This mirrors how these
two frameworks are actually used together in production: LangGraph for control flow,
CrewAI for a bounded multi-agent reasoning task.

**(b) Deterministic financial math is a separate module the graph calls directly, not
something delegated to an LLM or agent.** Balance reconciliation, duplicate detection, and
totals are pure arithmetic on structured data. Agents *interpret* the results of that
arithmetic (e.g., "is a $0.02 rounding gap acceptable?") — they don't perform it. This is
the single most important design decision in the project and the one most worth
explaining in an interview: LLMs are unreliable at arithmetic and you should never let one
be the source of truth for a number you can compute in Python.

Final pipeline:

```
PDF upload
   │
   ▼
Document Preprocessor  (PyMuPDF: PDF → page images, rotation fix, DPI normalization)
   │
   ▼
┌─────────────────────────── LangGraph StateGraph ───────────────────────────┐
│ load_document                                                              │
│   → preprocess_pages                                                      │
│     → extract_page_information   (VLM call per page, BaseVisionModel)     │
│       → merge_page_results        (stitch multi-page transactions)        │
│         → normalize_transactions  (dates, currency, amount parsing)       │
│           → validate_schema       (Pydantic — structural correctness)     │
│             → financial_validation (deterministic Python: balances/totals)│
│               → anomaly_detection  (deterministic: duplicates, outliers)  │
│                 → crew_validation  (CrewAI: interpret + cross-check)      │
│                   → confidence_scoring                                    │
│                     → [conditional edge] ─┬─> human_review (low conf.)    │
│                                            └─> finalize_result             │
└──────────────────────────────────────────────────────────────────────────┘
   │
   ▼
Structured JSON  ──────────────┬───────────────────────────┐
                                ▼                           ▼
                          PostgreSQL                 Streamlit UI
                     (documents/transactions/    (upload, review, download)
                      validation_results/...)
                                ▲
                                │
                          FastAPI (REST layer, background jobs, status polling)
```

## 2. VLM abstraction

```
BaseVisionModel (ABC)
   ├── extract(image: PIL.Image, prompt: str) -> RawVLMResponse
   ├── supports_batch: bool
   └── name: str
        │
        ├── QwenVLModel      (Qwen2-VL via transformers OR Hugging Face Inference API)
        ├── LLaVAModel       (llava-hf checkpoints via transformers)
        └── OCRFallbackModel (pytesseract — same interface, used when vision confidence is low)
```

All three implement the same interface so the LangGraph node that calls the model
(`extract_page_information`) never branches on model type — it just calls
`self.vision_model.extract(...)`. Model selection is a config value
(`MODEL_PROVIDER=qwen-vl|llava|hf-inference`), not a code change. This is the standard
Strategy pattern and it's worth naming explicitly in an interview — it's *why* swapping
models doesn't touch the pipeline.

**Hardware reality check (stated up front, not glossed over):**
- Qwen2-VL-7B in bf16 needs ~16GB VRAM to run comfortably; 4-bit (bitsandbytes) brings
  it to ~6-7GB.
- LLaVA-1.6-7B/13B has similar requirements; 13B in 4-bit is ~9-10GB.
- On a laptop GPU (8GB or less) or CPU-only, local inference will be slow-to-impractical
  for anything beyond a handful of pages. The practical path for most people building
  this as a portfolio project is: implement the local `transformers` backend for
  correctness/architecture demonstration, but default the running config to the
  Hugging Face Inference API (or a quantized 4-bit local model if you have a 12GB+ GPU)
  for actually processing documents. Both paths are implemented behind the same
  interface — you choose via `.env`.

## 3. Why these framework versions (verified today, Aug 2026)

| Library | Version pinned | Notes |
|---|---|---|
| Python | 3.11 | per your spec |
| langgraph | ^1.2 | current stable line; uses `StateGraph` + `add_conditional_edges`, checkpointer API stable since 1.x |
| crewai | ^1.14 | current stable; native LiteLLM multi-provider support, `Agent`/`Task`/`Crew`/`Process` API |
| langchain-core | pinned via langgraph's own dependency | avoid pinning a conflicting separate version |
| fastapi | ^0.115 | current, Pydantic v2 native |
| pydantic | ^2.9 | v2 throughout — `model_validate`, `field_validator`, not v1 `@validator` |
| sqlalchemy | ^2.0 | 2.0-style `Mapped[...]` declarative models, not legacy `Column` style |
| alembic | ^1.13 | migrations |
| streamlit | ^1.38 | dashboard |
| transformers | ^4.46 | Qwen2-VL support requires ≥4.45 |
| pytest | ^8.3 | testing |

I will pin exact versions in `requirements.txt` in Phase 2 and re-verify anything
version-sensitive (particularly `transformers`/Qwen2-VL compatibility) at that point.

## 4. Project structure (created on disk)

```
multimodal-financial-doc-ai/
├── app/
│   ├── api/                 # FastAPI routers, request/response models, exception handlers
│   ├── agents/               # CrewAI agent + task definitions, ReAct tools
│   ├── core/                 # config loading, logging setup, custom exceptions, security/masking
│   ├── database/             # SQLAlchemy engine/session, repositories
│   ├── document_processing/  # PDF -> images, preprocessing, limits/validation
│   ├── extraction/            # BaseVisionModel + QwenVL/LLaVA/OCR implementations, prompt templates
│   ├── graph/                 # LangGraph StateGraph, node functions, DocumentState
│   ├── models/                # SQLAlchemy ORM models
│   ├── schemas/               # Pydantic schemas (Document, Transaction, ValidationResult, ...)
│   ├── services/              # orchestration glue between API, graph, and DB (use-case layer)
│   ├── validation/            # deterministic financial validation + anomaly detection functions
│   └── main.py                 # FastAPI app entrypoint
├── frontend/streamlit_app.py
├── configs/config.yaml
├── data/{raw,processed,samples}
├── evaluation/                 # metrics + eval harness + sample dataset
├── tests/{unit,integration,fixtures}
├── scripts/                    # setup/seed/run helper scripts
├── migrations/                 # Alembic
├── Dockerfile, docker-compose.yml
├── requirements.txt, .env.example, .gitignore, Makefile, README.md
```

### Why `services/` exists as its own layer
The API layer (FastAPI routes) should not call the LangGraph graph or the database
directly — that couples HTTP concerns to orchestration logic and makes both harder to
test. `services/document_service.py` (built in Phase 13) is the use-case layer: it takes
a validated request, runs the graph, persists results, and returns a plain Python
object. Routes stay thin; the graph and DB stay swappable/testable in isolation. This is
the one piece of "clean architecture" ceremony in the project that earns its keep — I'm
intentionally *not* adding a full repository-per-aggregate/CQRS layer on top, since for a
project this size that would be over-engineering, not architecture.

## 5. Common production problems this design anticipates
- **VLM hallucinated numbers** → deterministic validation catches balance mismatches
  regardless of what the model "confidently" reported.
- **Partial failures on multi-page documents** → LangGraph node-level retries + per-page
  error capture so one bad page doesn't kill a 40-page statement.
- **Silent low-confidence extractions shipping to users** → confidence threshold routes to
  a `human_review` terminal state instead of `finalize_result`.
- **Sensitive data in logs** → masking utility applied at the logging boundary, not
  scattered through business logic (Phase 20).

---

Next: **Phase 2 — Configuration (`configs/config.yaml`, `.env.example`, `app/core/config.py`)
and all Pydantic schemas (`app/schemas/`)**, with exact dependency versions in
`requirements.txt`.
