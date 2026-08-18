"""
DocumentState — the single typed state object threaded through the entire LangGraph
pipeline.

Design decision: fields are populated incrementally by each node (a node returns only
the keys it changed; LangGraph merges that into the running state), and nothing is
removed once set — later nodes can always see earlier nodes' output. This is what
makes the human-review UI (Streamlit, Phase 14) able to show "here's exactly what
happened at each stage" for a document stuck in NEEDS_HUMAN_REVIEW: the full history
survives in state, not just the final result.

Note on serializability: some fields (`preprocessed`, `page_outcomes`) hold live
objects (PIL images inside PageImage, RawVLMResponse dataclasses) that are NOT JSON
serializable. That's fine for in-memory graph execution (the default, and what this
project uses) but would need trimming/summarizing before being handed to a
LangGraph persistent checkpointer backed by a serializing store. We call this out
explicitly rather than silently limiting checkpointing support.
"""

from __future__ import annotations

from typing import TypedDict

from app.document_processing.models import PreprocessedDocument
from app.extraction.page_extractor import PageExtractionOutcome
from app.schemas.enums import DocumentType, ProcessingStatus
from app.schemas.validation import ValidationResult


class DocumentState(TypedDict, total=False):
    # --- Input ---
    document_id: str
    file_bytes: bytes
    original_filename: str
    expected_document_type: DocumentType | None

    # --- preprocess_pages ---
    preprocessed: PreprocessedDocument | None

    # --- extract_page_information ---
    page_outcomes: list[PageExtractionOutcome]
    page_extraction_retry_count: int

    # --- merge_page_results ---
    raw_merged: dict | None

    # --- normalize_transactions ---
    normalized_transaction_dicts: list[dict]
    normalization_errors: list[str]

    # --- validate_schema ---
    final_document: dict | None  # FinalExtractionResult.model_dump(), once schema-valid
    schema_validation_errors: list[str]

    # --- validation stages (financial / anomaly / crew — Phases 8-9) ---
    validation_results: list[ValidationResult]

    # --- terminal bookkeeping ---
    status: ProcessingStatus
    error_message: str | None
