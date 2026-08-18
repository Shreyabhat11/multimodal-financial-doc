"""Unit tests for app/validation/financial.py and financial_validator.py."""

from __future__ import annotations

from decimal import Decimal

from app.schemas.document import FinancialTotals
from app.schemas.transaction import Transaction
from app.validation.financial import calculate_transaction_totals, reconcile_balance, reconcile_reported_totals
from app.validation.financial_validator import run_financial_validation


def _txns():
    return [
        Transaction(date="2024-01-05", description="Salary", credit="500.00", debit=0),
        Transaction(date="2024-01-10", description="Groceries", debit="200.00", credit=0),
        Transaction(date="2024-01-15", description="Rent", debit="300.00", credit=0),
    ]


class TestCalculateTotals:
    def test_sums_debits_and_credits_correctly(self):
        totals = calculate_transaction_totals(_txns())
        assert totals.total_debits == Decimal("500.00")
        assert totals.total_credits == Decimal("500.00")

    def test_empty_transaction_list(self):
        totals = calculate_transaction_totals([])
        assert totals.total_debits == Decimal("0")
        assert totals.total_credits == Decimal("0")


class TestReconcileBalance:
    def test_correct_reconciliation(self):
        totals = calculate_transaction_totals(_txns())
        result = reconcile_balance(
            opening_balance=Decimal("1000.00"), reported_closing_balance=Decimal("1000.00"), totals=totals
        )
        assert result.is_reconciled is True
        assert result.expected_closing_balance == Decimal("1000.00")

    def test_wrong_closing_balance_detected(self):
        totals = calculate_transaction_totals(_txns())
        result = reconcile_balance(
            opening_balance=Decimal("1000.00"), reported_closing_balance=Decimal("1500.00"), totals=totals
        )
        assert result.is_reconciled is False
        assert result.difference == Decimal("500.00")

    def test_tolerance_is_respected(self):
        totals = calculate_transaction_totals(_txns())
        # off by exactly the tolerance -> still reconciled
        result = reconcile_balance(
            opening_balance=Decimal("1000.00"),
            reported_closing_balance=Decimal("1000.01"),
            totals=totals,
            tolerance=Decimal("0.01"),
        )
        assert result.is_reconciled is True

    def test_just_outside_tolerance_fails(self):
        totals = calculate_transaction_totals(_txns())
        result = reconcile_balance(
            opening_balance=Decimal("1000.00"),
            reported_closing_balance=Decimal("1000.02"),
            totals=totals,
            tolerance=Decimal("0.01"),
        )
        assert result.is_reconciled is False


class TestReconcileReportedTotals:
    def test_no_reported_totals_is_not_a_failure(self):
        result = reconcile_reported_totals(transactions=_txns(), reported_totals=None)
        assert result.is_reconciled is True

    def test_matching_reported_totals(self):
        result = reconcile_reported_totals(
            transactions=_txns(), reported_totals=FinancialTotals(total_debits="500.00", total_credits="500.00")
        )
        assert result.is_reconciled is True

    def test_mismatched_reported_totals(self):
        result = reconcile_reported_totals(
            transactions=_txns(), reported_totals=FinancialTotals(total_debits="999.00", total_credits="500.00")
        )
        assert result.is_reconciled is False


class TestRunFinancialValidation:
    def test_clean_document_passes(self):
        result = run_financial_validation(
            opening_balance=Decimal("1000.00"),
            closing_balance=Decimal("1000.00"),
            transactions=_txns(),
            reported_totals=FinancialTotals(total_debits="500.00", total_credits="500.00"),
        )
        assert result.status.value == "passed"
        assert result.recommendation.value == "auto_approve"
        assert len(result.issues) == 0

    def test_wrong_balance_fails_with_high_severity_issue(self):
        result = run_financial_validation(
            opening_balance=Decimal("1000.00"),
            closing_balance=Decimal("1500.00"),
            transactions=_txns(),
            reported_totals=None,
        )
        assert result.status.value == "failed"
        assert result.recommendation.value == "human_review"
        assert any(issue.severity == "high" for issue in result.issues)

    def test_totals_mismatch_alone_is_only_a_warning(self):
        result = run_financial_validation(
            opening_balance=Decimal("1000.00"),
            closing_balance=Decimal("1000.00"),  # balance itself is correct
            transactions=_txns(),
            reported_totals=FinancialTotals(total_debits="999.00", total_credits="500.00"),  # but totals line is wrong
        )
        assert result.status.value == "passed_with_warnings"
        assert result.recommendation.value == "auto_approve"
