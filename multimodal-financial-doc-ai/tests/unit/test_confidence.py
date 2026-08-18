"""Unit tests for app/validation/confidence.py."""

from __future__ import annotations

from app.extraction.base_vision_model import RawVLMResponse
from app.extraction.page_extractor import PageExtractionOutcome
from app.schemas.enums import ValidationStatus
from app.schemas.validation import ValidationResult
from app.validation.confidence import compute_document_confidence

WEIGHTS = {
    "account_number": 0.20,
    "statement_period": 0.15,
    "opening_balance": 0.15,
    "closing_balance": 0.15,
    "transactions": 0.35,
}


def _make_outcome(page_number, parsed_json, self_reported):
    response = RawVLMResponse(
        page_number=page_number,
        model_name="fake",
        raw_text="",
        parsed_json=parsed_json,
        latency_seconds=0.01,
        succeeded=True,
        self_reported_confidence=self_reported,
    )
    return PageExtractionOutcome(
        page_number=page_number,
        parsed_json=parsed_json,
        confidence=sum(self_reported.values()) / len(self_reported) if self_reported else 0.7,
        source="vlm",
        used_fallback=False,
        vlm_response=response,
    )


def _base_page_outcomes():
    return [
        _make_outcome(
            1,
            {"account_number": "9988776655", "opening_balance": 1000.0, "transactions": [{"date": "2024-01-05"}]},
            {"account_number": 0.95, "opening_balance": 0.9, "transactions": 0.85},
        ),
        _make_outcome(
            2,
            {
                "closing_balance": 1300.0,
                "statement_period": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
                "transactions": [{"date": "2024-01-10"}],
            },
            {"closing_balance": 0.88, "statement_period": 0.92, "transactions": 0.7},
        ),
    ]


class TestComputeDocumentConfidence:
    def test_high_confidence_when_all_validations_pass(self):
        conf = compute_document_confidence(
            raw_merged={
                "account_number": "9988776655",
                "opening_balance": 1000.0,
                "closing_balance": 1300.0,
                "statement_period": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
            },
            page_outcomes=_base_page_outcomes(),
            validation_results=[ValidationResult.passed("financial_validation", ["balance_reconciliation"])],
            weights=WEIGHTS,
            threshold=0.75,
        )
        assert conf.overall_confidence > 0.75
        assert conf.below_threshold is False

    def test_failed_validation_drags_confidence_below_threshold(self):
        raw_merged = {
            "account_number": "9988776655",
            "opening_balance": 1000.0,
            "closing_balance": 1300.0,
            "statement_period": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
        }
        page_outcomes = _base_page_outcomes()

        passing = compute_document_confidence(
            raw_merged=raw_merged,
            page_outcomes=page_outcomes,
            validation_results=[ValidationResult.passed("financial_validation", [])],
            weights=WEIGHTS,
            threshold=0.75,
        )
        failing = compute_document_confidence(
            raw_merged=raw_merged,
            page_outcomes=page_outcomes,
            validation_results=[
                ValidationResult(
                    validator_name="financial_validation",
                    status=ValidationStatus.FAILED,
                    checks_performed=["balance_reconciliation"],
                )
            ],
            weights=WEIGHTS,
            threshold=0.75,
        )
        assert failing.overall_confidence < passing.overall_confidence
        assert failing.below_threshold is True

    def test_missing_field_scores_zero_confidence(self):
        conf = compute_document_confidence(
            raw_merged={"account_number": None},  # never found on any page
            page_outcomes=[],
            validation_results=[],
            weights=WEIGHTS,
            threshold=0.75,
        )
        assert conf.field_scores["account_number"].confidence == 0.0
        assert conf.field_scores["account_number"].source == "missing"
