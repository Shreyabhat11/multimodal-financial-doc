"""
Graph assembly - now complete end to end.

    load_document -> preprocess_pages -> extract_page_information -> merge_page_results
    -> normalize_transactions -> validate_schema -> financial_validation
    -> anomaly_detection -> crew_validation -> confidence_scoring
    -> [route] -> finalize_result (COMPLETED) | human_review (NEEDS_HUMAN_REVIEW)

This is the same module built incrementally across Phases 7-10, exactly as promised
in the Phase 7 docstring: VALIDATING was always a placeholder terminal status, and the
comment marking exactly where financial_validation would attach is now realized below.

Conditional edges, complete list:
  1. load_document -> [FAILED: END] | [ok: preprocess_pages]
  2. preprocess_pages -> [FAILED: END] | [ok: extract_page_information]
  3. extract_page_information -> [too many page failures AND retries remain: loop
     back via increment_extraction_retry] | [ok: merge_page_results]
  4. validate_schema -> [structurally invalid: human_review] | [ok: financial_validation]
  5. financial_validation -> anomaly_detection (unconditional -- always run both
     deterministic validators regardless of whether the first one failed, since a
     human reviewing a balance mismatch also wants to know about any anomalies found)
  6. anomaly_detection -> crew_validation (unconditional, same reasoning)
  7. crew_validation -> confidence_scoring (unconditional)
  8. confidence_scoring -> [any FAILED validation_result, OR confidence below
     threshold: human_review] | [ok: finalize_result]
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from app.core.config import Settings, get_settings
from app.core.exceptions import VisionModelError
from app.document_processing.pipeline import DocumentPreprocessor
from app.extraction.base_vision_model import BaseVisionModel
from app.extraction.page_extractor import PageExtractor
from app.graph.nodes import (
    anomaly_detection_node,
    financial_validation_node,
    finalize_result_node,
    human_review_node,
    increment_extraction_retry_node,
    load_document_node,
    make_confidence_scoring_node,
    make_crew_validation_node,
    make_extract_page_information_node,
    make_preprocess_pages_node,
    merge_page_results_node,
    normalize_transactions_node,
    validate_schema_node,
)
from app.graph.state import DocumentState
from app.schemas.enums import ProcessingStatus, ValidationStatus

MAX_EXTRACTION_RETRIES = 1
PAGE_FAILURE_RATE_RETRY_THRESHOLD = 0.5


def _route_after_load(state: DocumentState) -> str:
    return END if state.get("status") == ProcessingStatus.FAILED else "preprocess_pages"


def _route_after_preprocess(state: DocumentState) -> str:
    if state.get("status") == ProcessingStatus.FAILED:
        return END
    return "extract_page_information"


def _route_after_extraction(state: DocumentState) -> str:
    outcomes = state.get("page_outcomes", [])
    if not outcomes:
        return "merge_page_results"

    failure_rate = sum(1 for o in outcomes if o.source == "failed") / len(outcomes)
    retry_count = state.get("page_extraction_retry_count", 0)

    if failure_rate > PAGE_FAILURE_RATE_RETRY_THRESHOLD and retry_count < MAX_EXTRACTION_RETRIES:
        return "increment_extraction_retry"
    return "merge_page_results"


def _route_after_validate_schema(state: DocumentState) -> str:
    if state.get("status") == ProcessingStatus.NEEDS_HUMAN_REVIEW:
        return "human_review"
    return "financial_validation"


def _route_after_confidence_scoring(state: DocumentState) -> str:
    """The single most consequential routing decision in the whole pipeline: does
    this document ship as COMPLETED, or does it go to a human? Two independent
    triggers, either one is sufficient:
      (a) any validation stage (financial_validation / anomaly_detection /
          crew_validation) reported FAILED
      (b) the aggregated document confidence (this phase) is below
          settings.confidence_threshold
    Deliberately OR, not AND -- a document could have perfect confidence scores but
    still fail balance reconciliation (the VLM was "confident" about numbers that
    turned out not to add up), and that must route to human review regardless of
    confidence.
    """
    validation_results = state.get("validation_results", [])
    any_failed = any(r.status == ValidationStatus.FAILED for r in validation_results)

    final_document = state.get("final_document") or {}
    confidence_data = final_document.get("confidence") or {}
    below_threshold = confidence_data.get("below_threshold", True)

    if any_failed or below_threshold:
        return "human_review"
    return "finalize_result"


def build_document_graph(
    vision_model: BaseVisionModel,
    *,
    ocr_fallback_model: BaseVisionModel | None = None,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Construct and compile the full document-processing graph."""
    settings = settings or get_settings()

    preprocessor = DocumentPreprocessor(
        max_file_size_mb=settings.max_file_size_mb,
        max_pages=settings.max_pages,
        page_render_dpi=settings.page_render_dpi,
    )
    page_extractor = PageExtractor(
        vision_model,
        ocr_fallback_model=ocr_fallback_model,
        ocr_enabled=settings.ocr_enabled,
        ocr_trigger_confidence_below=settings.ocr_trigger_confidence_below,
    )

    graph = StateGraph(DocumentState)

    graph.add_node("load_document", load_document_node)
    graph.add_node("preprocess_pages", make_preprocess_pages_node(preprocessor))
    graph.add_node(
        "extract_page_information",
        make_extract_page_information_node(page_extractor),
        retry_policy=RetryPolicy(max_attempts=2, retry_on=(VisionModelError, ConnectionError, TimeoutError)),
    )
    graph.add_node("increment_extraction_retry", increment_extraction_retry_node)
    graph.add_node("merge_page_results", merge_page_results_node)
    graph.add_node("normalize_transactions", normalize_transactions_node)
    graph.add_node("validate_schema", validate_schema_node)
    graph.add_node("financial_validation", financial_validation_node)
    graph.add_node("anomaly_detection", anomaly_detection_node)
    graph.add_node("crew_validation", make_crew_validation_node(settings))
    graph.add_node("confidence_scoring", make_confidence_scoring_node(settings))
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize_result", finalize_result_node)

    graph.add_edge(START, "load_document")
    graph.add_conditional_edges(
        "load_document", _route_after_load, {"preprocess_pages": "preprocess_pages", END: END}
    )
    graph.add_conditional_edges(
        "preprocess_pages",
        _route_after_preprocess,
        {"extract_page_information": "extract_page_information", END: END},
    )
    graph.add_conditional_edges(
        "extract_page_information",
        _route_after_extraction,
        {"increment_extraction_retry": "increment_extraction_retry", "merge_page_results": "merge_page_results"},
    )
    graph.add_edge("increment_extraction_retry", "extract_page_information")
    graph.add_edge("merge_page_results", "normalize_transactions")
    graph.add_edge("normalize_transactions", "validate_schema")
    graph.add_conditional_edges(
        "validate_schema",
        _route_after_validate_schema,
        {"human_review": "human_review", "financial_validation": "financial_validation"},
    )
    graph.add_edge("financial_validation", "anomaly_detection")
    graph.add_edge("anomaly_detection", "crew_validation")
    graph.add_edge("crew_validation", "confidence_scoring")
    graph.add_conditional_edges(
        "confidence_scoring",
        _route_after_confidence_scoring,
        {"human_review": "human_review", "finalize_result": "finalize_result"},
    )
    graph.add_edge("human_review", END)
    graph.add_edge("finalize_result", END)

    return graph.compile()
