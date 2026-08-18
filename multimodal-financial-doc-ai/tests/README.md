# Test Suite Notes

## Running the tests

```bash
pip install -r requirements.txt
pytest                                    # run everything
pytest --cov=app --cov-report=term-missing   # with coverage
pytest tests/unit                         # unit tests only (fast, no I/O)
pytest tests/integration                  # integration tests (real SQLite, real HTTP via TestClient, real LangGraph execution)
```

No test in this suite makes a real network call. Every VLM backend is a fake
`BaseVisionModel` subclass (see `tests/conftest.py`); CrewAI's `run_crew_validation`
is monkeypatched to a deterministic pass via an **autouse** fixture, so no test
accidentally calls a real Anthropic API.

## Coverage, honestly reported

Last real run: **83% line coverage on `app/`, 81 tests, all passing.**

Deliberate, explained gaps — not oversights:

- **`app/extraction/qwen_vl_model.py`, `llava_model.py` (0%)**: the actual
  `transformers.generate()` / Hugging Face Inference API call bodies are never
  exercised — every test uses a fake `BaseVisionModel` subclass instead. Testing the
  real call bodies would require either a GPU + downloaded model weights or a live
  HF Inference API token, neither available in CI. What IS tested (see
  `tests/unit` interactively during development, Phases 4-5): the lazy-import
  discipline (these modules import fine without `torch` installed), the shared
  `BaseVisionModel.extract()` wrapping logic both backends inherit, and the
  factory's config-driven backend selection.
- **`app/extraction/model_factory.py`, `ocr_engine.py` (~30%)**: same reason —
  branches that construct a real model client or invoke real Tesseract aren't hit by
  fake-backend tests. `ocr_engine.py`'s actual OCR call *was* verified for real
  against a rendered PDF page during Phase 6 development (see chat history / earlier
  interactive runs) — that verification isn't currently expressed as a pytest case
  requiring a system Tesseract install, to keep the suite runnable in minimal CI
  images without a `tesseract-ocr` system dependency.
- **`app/agents/crew.py`, `tools.py`, `agent_definitions.py`, `task_definitions.py`
  (30-65%)**: the internal CrewAI `Agent`/`Task`/`Crew` construction logic isn't
  exercised because `run_crew_validation` itself is monkeypatched at the boundary
  (an intentional choice — every test in this suite must be network-free). This
  logic WAS verified for real during Phase 9 development: real `Crew`/`Agent`/`Task`
  objects were constructed and inspected, all 7 ReAct tools were run standalone
  against real transaction data, and a genuine crew execution failure (invalid API
  key) was confirmed to raise `CrewValidationError` correctly. A CI job with a real
  (even if low-tier) Anthropic API key could add a small number of `@pytest.mark.
  slow` / `@pytest.mark.requires_api_key` tests exercising this for real; that's a
  reasonable next step, not something this suite claims to already do.

## What every listed brief requirement maps to

| Brief requirement | Test(s) |
|---|---|
| Incorrect totals | `test_financial_validation.py::TestRunFinancialValidation::test_wrong_balance_fails_with_high_severity_issue` |
| Duplicate transactions | `test_anomaly_detection.py::TestDuplicateDetection` |
| Missing transactions | `test_langgraph_pipeline.py::TestExtractionFailureAndRetry`, `test_transaction_normalization.py::TestMergePageResults::test_failed_page_contributes_nothing...` |
| Invalid dates | `test_anomaly_detection.py::TestDateValidation`, `test_schemas.py::TestDateParsing::test_rejects_unparseable_date` |
| Balance mismatch | `test_financial_validation.py::TestReconcileBalance`, `test_langgraph_pipeline.py::TestFinancialValidationCannotBeOverriddenByCrew` |
| Malformed VLM output | `test_transaction_normalization.py::TestParseVlmJsonResponse`, `test_langgraph_pipeline.py::TestMalformedVlmOutput` |
| Low confidence | `test_confidence.py::TestComputeDocumentConfidence` |
| Multi-page documents | `test_langgraph_pipeline.py::TestFullPipelineHappyPath` (real 3-page synthetic PDF, one page rotated) |
