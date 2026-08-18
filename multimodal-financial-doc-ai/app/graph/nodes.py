"""
Node functions for the document-processing graph.

Each node is a plain function `(state: DocumentState) -> dict`, returning only the
keys it changed — this is the standard LangGraph node contract, and it's what lets
LangGraph merge partial updates rather than requiring every node to reconstruct the
entire state.

Nodes are built via factory functions (`make_preprocess_pages_node`, etc.) that close
over injected dependencies (a `DocumentPreprocessor`, a `PageExtractor`). This is
dependency injection applied to graph nodes: the graph module never hardcodes "use
QwenVLModel" — `graph_builder.py` (below) is the one place that wires concrete
dependencies in, mirroring how `model_factory.py` is the one place that picks a
concrete VLM backend. Nodes themselves are fully unit-testable by constructing them
with fake dependencies, independent of LangGraph's execution machinery.

Error handling philosophy: EXPECTED failures (corrupted PDF, VLM call failure) are
caught inside nodes and turned into state updates (`status=FAILED`,
`error_message=...`) — they do not raise. This is deliberate: raising would abort
graph execution entirely and lose all state gathered so far, whereas a document that
fails validation is a normal, expected outcome we want to persist and show to a human
reviewer, not a crash. Only genuinely unexpected exceptions (a real bug) are allowed
to propagate, where LangGraph's node-level RetryPolicy (see graph_builder.py) gives
them one automatic retry before surfacing.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import ValidationError

from app.document_processing.pipeline import DocumentPreprocessor
from app.extraction.merge import merge_page_results
from app.extraction.page_extractor import PageExtractor
from app.core.exceptions import CrewValidationError, DocumentProcessingError
from app.graph.state import DocumentState
from app.schemas import Account, DocumentMetadata, FinancialTotals, StatementPeriod, Transaction
from app.schemas.enums import ProcessingStatus, RecommendedAction, ValidationStatus
from app.validation.anomaly_detection import run_anomaly_detection
from app.validation.confidence import compute_document_confidence
from app.validation.financial_validator import run_financial_validation

# Required for a document to be considered schema-valid enough to proceed to
# financial validation (Phase 8). Anything short of this routes to NEEDS_HUMAN_REVIEW
# rather than failing outright — a human can often fill in a missing account name
# faster than the whole document can be reprocessed.
_REQUIRED_FOR_SCHEMA_VALIDITY = (
    "account_number",
    "opening_balance",
    "closing_balance",
    "statement_period",
)


def load_document_node(state: DocumentState) -> dict:
    """Entry node: minimal sanity checks on the input before any real work starts."""
    if not state.get("file_bytes"):
        return {
            "status": ProcessingStatus.FAILED,
            "error_message": "No file bytes provided to the pipeline.",
        }
    if not state.get("document_id"):
        return {
            "status": ProcessingStatus.FAILED,
            "error_message": "No document_id provided to the pipeline.",
        }
    return {"status": ProcessingStatus.PROCESSING}


def make_preprocess_pages_node(preprocessor: DocumentPreprocessor):
    def preprocess_pages_node(state: DocumentState) -> dict:
        try:
            preprocessed = preprocessor.process(
                state["file_bytes"],
                original_filename=state["original_filename"],
                document_id=state["document_id"],
            )
        except DocumentProcessingError as exc:
            return {
                "status": ProcessingStatus.FAILED,
                "error_message": exc.message,
            }
        return {"preprocessed": preprocessed, "status": ProcessingStatus.PROCESSING}

    return preprocess_pages_node


def make_extract_page_information_node(page_extractor: PageExtractor):
    def extract_page_information_node(state: DocumentState) -> dict:
        preprocessed = state["preprocessed"]
        expected_type = state.get("expected_document_type")
        total_pages = preprocessed.page_count

        outcomes = [
            page_extractor.extract_page(
                page_image,
                total_pages=total_pages,
                expected_document_type=expected_type,
            )
            for page_image in preprocessed.pages
        ]

        retry_count = state.get("page_extraction_retry_count", 0)
        return {"page_outcomes": outcomes, "page_extraction_retry_count": retry_count}

    return extract_page_information_node


def increment_extraction_retry_node(state: DocumentState) -> dict:
    """Small dedicated node for the retry loop-back edge, so the retry counter
    increment is visible as its own graph step rather than hidden as a side effect of
    re-running extraction — makes the retry behavior legible in a graph visualization/
    trace, which matters a lot when debugging why a document took 3x longer than
    expected."""
    return {"page_extraction_retry_count": state.get("page_extraction_retry_count", 0) + 1}


def merge_page_results_node(state: DocumentState) -> dict:
    merged = merge_page_results(state["page_outcomes"])
    return {"raw_merged": merged}


def normalize_transactions_node(state: DocumentState) -> dict:
    """Parse each raw transaction dict through the Transaction schema (Phase 2),
    which is where amount/date parsing, sign normalization, and structural
    validation actually happen (see app.schemas.transaction). A single malformed
    transaction row does not abort the document — it's recorded in
    `normalization_errors` (visible to the human-review UI) and excluded from the
    normalized list, since one bad row souring an otherwise-good 200-row statement
    would be a worse outcome than flagging just that row.
    """
    raw_transactions = state["raw_merged"].get("transactions", [])
    normalized: list[dict] = []
    errors: list[str] = []

    for i, raw_txn in enumerate(raw_transactions):
        try:
            txn = Transaction(**raw_txn)
            normalized.append(txn.model_dump(mode="json"))
        except ValidationError as exc:
            errors.append(f"Transaction at index {i} (source_page={raw_txn.get('source_page')}): {exc}")

    return {"normalized_transaction_dicts": normalized, "normalization_errors": errors}


def validate_schema_node(state: DocumentState) -> dict:
    """Attempt to assemble the document-level schema objects (Account,
    StatementPeriod, FinancialTotals) from the merged raw data. This is purely
    STRUCTURAL validation — "can we even build a well-typed object out of what we
    extracted" — as distinct from financial_validation (Phase 8), which checks
    whether the numbers are internally consistent. A document can pass this node and
    still fail financial_validation (wrong closing balance), and it can fail this
    node without ever reaching financial_validation (no account number found at all).
    """
    raw = state["raw_merged"]
    errors: list[str] = []

    for field_name in _REQUIRED_FOR_SCHEMA_VALIDITY:
        if not raw.get(field_name):
            errors.append(f"Missing required field: '{field_name}'")

    if errors:
        return {
            "schema_validation_errors": errors,
            "status": ProcessingStatus.NEEDS_HUMAN_REVIEW,
        }

    try:
        account = Account(
            account_holder=raw.get("account_holder") or "UNKNOWN",
            account_number=raw["account_number"],
            bank_name=raw.get("bank_name") or "",
        )
        statement_period = StatementPeriod(**raw["statement_period"])
        totals_raw = raw.get("totals") or {"total_debits": 0, "total_credits": 0}
        totals = FinancialTotals(**totals_raw)
    except ValidationError as exc:
        return {
            "schema_validation_errors": [str(exc)],
            "status": ProcessingStatus.NEEDS_HUMAN_REVIEW,
        }

    preprocessed = state["preprocessed"]
    metadata = DocumentMetadata(
        document_id=state["document_id"],
        original_filename=state["original_filename"],
        document_type=raw.get("document_type") or "unknown",
        page_count=preprocessed.page_count,
        file_size_bytes=preprocessed.file_size_bytes,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
    )

    final_document = {
        "document_id": state["document_id"],
        "document_type": metadata.document_type,
        "account": account.model_dump(mode="json"),
        "statement_period": statement_period.model_dump(mode="json"),
        "opening_balance": str(raw["opening_balance"]),
        "closing_balance": str(raw["closing_balance"]),
        "currency": raw.get("currency") or "USD",
        "transactions": state.get("normalized_transaction_dicts", []),
        "totals": totals.model_dump(mode="json"),
        "metadata": metadata.model_dump(mode="json"),
    }

    return {
        "final_document": final_document,
        "schema_validation_errors": [],
        "status": ProcessingStatus.VALIDATING,
    }


def financial_validation_node(state: DocumentState) -> dict:
    """Deterministic balance/totals reconciliation (Phase 8), wrapped as a graph
    node. Reconstructs typed Transaction/FinancialTotals objects from the
    final_document dict assembled by validate_schema — this is the one place those
    dicts get turned back into real Decimal-backed objects for the validation math to
    run on."""
    from decimal import Decimal

    final_document = state["final_document"]
    transactions = [Transaction(**t) for t in final_document["transactions"]]
    reported_totals = FinancialTotals(**final_document["totals"]) if final_document.get("totals") else None

    result = run_financial_validation(
        opening_balance=Decimal(final_document["opening_balance"]),
        closing_balance=Decimal(final_document["closing_balance"]),
        transactions=transactions,
        reported_totals=reported_totals,
    )
    existing = state.get("validation_results", [])
    return {"validation_results": existing + [result]}


def anomaly_detection_node(state: DocumentState) -> dict:
    """Deterministic pattern-based anomaly detection (Phase 8), wrapped as a graph node."""
    final_document = state["final_document"]
    transactions = [Transaction(**t) for t in final_document["transactions"]]
    statement_period = StatementPeriod(**final_document["statement_period"])

    result = run_anomaly_detection(transactions=transactions, statement_period=statement_period)
    existing = state.get("validation_results", [])
    return {"validation_results": existing + [result]}


def make_crew_validation_node(settings):
    """CrewAI four-agent validation crew (Phase 9), wrapped as a graph node.

    A crew execution failure (LLM provider unreachable, malformed structured output
    that survives CrewAI's own retry) is caught here and turned into a synthetic
    FAILED ValidationResult recommending human review — consistent with every other
    node's philosophy of "expected failures become state, not exceptions" (see the
    module docstring). The crew is a validation OPINION on top of deterministic
    checks that already ran; losing the crew's input for one document should degrade
    to human review, not crash the pipeline.
    """

    def crew_validation_node(state: DocumentState) -> dict:
        from decimal import Decimal

        from app.agents.crew import run_crew_validation

        final_document = state["final_document"]
        transactions = [Transaction(**t) for t in final_document["transactions"]]
        account = Account(**final_document["account"])
        statement_period = StatementPeriod(**final_document["statement_period"])
        reported_totals = FinancialTotals(**final_document["totals"]) if final_document.get("totals") else None

        try:
            result = run_crew_validation(
                account=account,
                statement_period=statement_period,
                opening_balance=Decimal(final_document["opening_balance"]),
                closing_balance=Decimal(final_document["closing_balance"]),
                currency=final_document["currency"],
                transactions=transactions,
                reported_totals=reported_totals,
                settings=settings,
            )
        except CrewValidationError as exc:
            from app.schemas.validation import ValidationResult

            result = ValidationResult(
                validator_name="crew_validation",
                status=ValidationStatus.FAILED,
                checks_performed=[],
                issues=[],
                recommendation=RecommendedAction.HUMAN_REVIEW,
            )
            existing = state.get("validation_results", [])
            return {
                "validation_results": existing + [result],
                "error_message": f"Crew validation unavailable: {exc.message}",
            }

        existing = state.get("validation_results", [])
        return {"validation_results": existing + [result]}

    return crew_validation_node


def make_confidence_scoring_node(settings):
    """Document-level confidence aggregation (this phase), wrapped as a graph node."""

    def confidence_scoring_node(state: DocumentState) -> dict:
        confidence = compute_document_confidence(
            raw_merged=state["raw_merged"],
            page_outcomes=state["page_outcomes"],
            validation_results=state.get("validation_results", []),
            weights=settings.confidence_weights,
            threshold=settings.confidence_threshold,
        )
        final_document = dict(state["final_document"])
        final_document["confidence"] = confidence.model_dump(mode="json")
        return {"final_document": final_document}

    return confidence_scoring_node


def human_review_node(state: DocumentState) -> dict:
    """Terminal node for documents routed to human review. Deliberately does NOT
    discard final_document if one was assembled — a document can reach here with a
    perfectly well-formed final_document that just failed a confidence/validation
    threshold, and a human reviewer benefits from seeing the full extraction next to
    whatever flagged it, not just an error message."""
    return {"status": ProcessingStatus.NEEDS_HUMAN_REVIEW}


def finalize_result_node(state: DocumentState) -> dict:
    """Terminal node for documents that pass every gate. Marks the document COMPLETED
    — this is the only node allowed to set that status, so "COMPLETED" in storage
    always means the full validation chain ran and passed, never a shortcut."""
    return {"status": ProcessingStatus.COMPLETED}
