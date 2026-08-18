"""
compute_document_confidence - the aggregation step referenced throughout earlier
phases as "Phase 11" (now folded into this phase alongside human-review routing,
since the two are directly coupled: this function's output is exactly what the
routing decision in graph_builder.py branches on).

Two input signals are combined, deliberately weighted differently:

1. Per-field VLM/OCR self-reported confidence (from RawVLMResponse.self_reported_
   confidence, threaded through PageExtractionOutcome) — the model's own read on how
   clearly it could see each field. Weighted per-field via settings.confidence_weights
   (transactions weighted highest at 0.35, since a wrong transaction list is the most
   consequential extraction error for this system's purpose).

2. Deterministic validation outcomes (ValidationResult.status from financial_
   validation, anomaly_detection, crew_validation) — applied as a document-level
   MULTIPLIER on top of the field-confidence average, not blended field-by-field.
   Rationale: a validation FAILURE (e.g. balance doesn't reconcile) casts doubt on the
   whole document's numeric extraction, not just one field — even if the VLM was
   "confident" about opening_balance and closing_balance individually, the fact that
   they don't reconcile with the transaction list means at least one of them is wrong,
   which self-reported per-field confidence alone cannot detect (it's a cross-field
   consistency signal, not a per-field one). This is precisely why the confidence
   score should NOT be computed purely from step 1 — it would systematically miss the
   exact failure mode deterministic validation exists to catch.
"""

from __future__ import annotations

from app.extraction.page_extractor import PageExtractionOutcome
from app.schemas.confidence import DocumentConfidence, FieldConfidence
from app.schemas.enums import ValidationStatus
from app.schemas.validation import ValidationResult

# Multiplier applied to the field-confidence average based on the worst validation
# status seen across financial_validation / anomaly_detection / crew_validation.
# FAILED docs are penalized heavily (but not zeroed — a single reconciliation miss
# doesn't mean the extraction was worthless, just that it needs a human look);
# PASSED_WITH_WARNINGS gets a light penalty; PASSED gets none.
_VALIDATION_STATUS_MULTIPLIER = {
    ValidationStatus.PASSED: 1.0,
    ValidationStatus.PASSED_WITH_WARNINGS: 0.9,
    ValidationStatus.FAILED: 0.6,
}


def _resolve_field_confidence(
    field_name: str,
    *,
    raw_merged: dict,
    page_outcomes: list[PageExtractionOutcome],
) -> FieldConfidence:
    """Find which page contributed this document-level field's value, and use that
    page's self-reported confidence for the field if available, falling back to the
    page's overall confidence score otherwise."""
    value = raw_merged.get(field_name)

    for outcome in sorted(page_outcomes, key=lambda o: o.page_number):
        if not outcome.succeeded:
            continue
        page_data = outcome.parsed_json
        if page_data.get(field_name):
            source = outcome.vlm_response.self_reported_confidence or {}
            if outcome.ocr_response is not None and outcome.source == "ocr_fallback":
                source = outcome.ocr_response.self_reported_confidence or source
            field_conf = source.get(field_name)
            confidence = float(field_conf) if isinstance(field_conf, (int, float)) else outcome.confidence
            return FieldConfidence(
                field_name=field_name,
                value=value,
                confidence=max(0.0, min(1.0, confidence)),
                source=outcome.source,
            )

    # Field never found on any successful page — zero confidence, not "unknown."
    return FieldConfidence(field_name=field_name, value=value, confidence=0.0, source="missing")


def _resolve_transactions_confidence(
    *,
    page_outcomes: list[PageExtractionOutcome],
) -> FieldConfidence:
    """Transactions confidence is the mean of each contributing page's
    self-reported 'transactions' confidence (or overall page confidence as
    fallback), weighted equally per page — a page that contributed zero
    transactions (e.g. a cover page) doesn't affect this score."""
    scores: list[float] = []
    for outcome in page_outcomes:
        if not outcome.succeeded:
            continue
        page_data = outcome.parsed_json
        if not page_data.get("transactions"):
            continue
        source = outcome.vlm_response.self_reported_confidence or {}
        field_conf = source.get("transactions")
        score = float(field_conf) if isinstance(field_conf, (int, float)) else outcome.confidence
        scores.append(max(0.0, min(1.0, score)))

    confidence = sum(scores) / len(scores) if scores else 0.0
    return FieldConfidence(field_name="transactions", value=None, confidence=confidence, source="aggregated")


def compute_document_confidence(
    *,
    raw_merged: dict,
    page_outcomes: list[PageExtractionOutcome],
    validation_results: list[ValidationResult],
    weights: dict[str, float],
    threshold: float,
) -> DocumentConfidence:
    field_scores: dict[str, FieldConfidence] = {}

    for field_name in weights:
        if field_name == "transactions":
            field_scores[field_name] = _resolve_transactions_confidence(page_outcomes=page_outcomes)
        else:
            field_scores[field_name] = _resolve_field_confidence(
                field_name, raw_merged=raw_merged, page_outcomes=page_outcomes
            )

    weighted_sum = sum(field_scores[name].confidence * weight for name, weight in weights.items())

    worst_status = ValidationStatus.PASSED
    for result in validation_results:
        if result.status == ValidationStatus.FAILED:
            worst_status = ValidationStatus.FAILED
            break
        if result.status == ValidationStatus.PASSED_WITH_WARNINGS and worst_status == ValidationStatus.PASSED:
            worst_status = ValidationStatus.PASSED_WITH_WARNINGS

    multiplier = _VALIDATION_STATUS_MULTIPLIER[worst_status]
    overall = round(weighted_sum * multiplier, 4)
    overall = max(0.0, min(1.0, overall))

    return DocumentConfidence(
        field_scores=field_scores,
        overall_confidence=overall,
        below_threshold=overall < threshold,
    )
