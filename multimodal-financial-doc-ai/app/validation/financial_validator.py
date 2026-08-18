"""
run_financial_validation - the deterministic-arithmetic validator.

This is one of two ValidationResult producers wired into the LangGraph
`financial_validation` node (graph_builder.py, extended below). The other,
`run_anomaly_detection` (anomaly_detection.py), handles pattern-based checks
(duplicates, outliers, date issues). Splitting them into two ValidationResults
(rather than one giant validator) keeps each one's `checks_performed` list
meaningful and lets the human-review UI show "financial reconciliation: FAILED,
anomaly detection: PASSED_WITH_WARNINGS" as two distinct signals rather than one
conflated status.
"""

from __future__ import annotations

from decimal import Decimal

from app.schemas.document import FinancialTotals
from app.schemas.enums import RecommendedAction, ValidationStatus
from app.schemas.transaction import Transaction
from app.schemas.validation import ValidationIssue, ValidationResult
from app.validation.financial import calculate_transaction_totals, reconcile_balance, reconcile_reported_totals


def run_financial_validation(
    *,
    opening_balance: Decimal,
    closing_balance: Decimal,
    transactions: list[Transaction],
    reported_totals: FinancialTotals | None,
    balance_tolerance: Decimal = Decimal("0.01"),
) -> ValidationResult:
    """Run balance reconciliation and reported-totals reconciliation, and combine the
    results into one ValidationResult.

    Status logic:
    - Both checks pass -> PASSED
    - Totals mismatch alone -> PASSED_WITH_WARNINGS (the transaction list is usable,
      but the document's own totals line disagrees with it - worth a note, not
      necessarily worth blocking auto-approval on its own)
    - Balance reconciliation fails -> FAILED (this is the check that most directly
      indicates something is wrong with the extracted numbers themselves, so it's
      treated as the harder failure)
    """
    checks_performed = ["balance_reconciliation", "reported_totals_reconciliation"]
    issues: list[ValidationIssue] = []

    # Balance reconciliation always uses totals COMPUTED from the transaction list,
    # never the document-reported totals - even if the document's totals line is
    # present, using it here would let a self-consistent-but-wrong totals line mask a
    # real transaction-list error. Reported totals are only used in the separate
    # reconcile_reported_totals cross-check below.
    computed_totals_for_reconciliation = calculate_transaction_totals(transactions)
    balance_result = reconcile_balance(
        opening_balance=opening_balance,
        reported_closing_balance=closing_balance,
        totals=computed_totals_for_reconciliation,
        tolerance=balance_tolerance,
    )

    totals_result = reconcile_reported_totals(
        transactions=transactions, reported_totals=reported_totals, tolerance=balance_tolerance
    )

    if not totals_result.is_reconciled:
        issues.append(
            ValidationIssue(
                field="totals",
                severity="medium",
                message=(
                    f"Document-reported totals (debits={totals_result.reported_totals.total_debits}, "
                    f"credits={totals_result.reported_totals.total_credits}) do not match totals computed "
                    f"from the transaction list (debits={totals_result.computed_totals.total_debits}, "
                    f"credits={totals_result.computed_totals.total_credits})."
                ),
            )
        )

    if not balance_result.is_reconciled:
        issues.append(
            ValidationIssue(
                field="closing_balance",
                severity="high",
                message=(
                    f"Expected closing balance {balance_result.expected_closing_balance} "
                    f"(opening {balance_result.opening_balance} + credits - debits) does not match "
                    f"reported closing balance {balance_result.reported_closing_balance} "
                    f"(difference: {balance_result.difference}, tolerance: {balance_result.tolerance})."
                ),
            )
        )

    if balance_result.is_reconciled and totals_result.is_reconciled:
        return ValidationResult(
            validator_name="financial_validation",
            status=ValidationStatus.PASSED,
            checks_performed=checks_performed,
            issues=issues,
            recommendation=RecommendedAction.AUTO_APPROVE,
        )

    if not balance_result.is_reconciled:
        return ValidationResult(
            validator_name="financial_validation",
            status=ValidationStatus.FAILED,
            checks_performed=checks_performed,
            issues=issues,
            recommendation=RecommendedAction.HUMAN_REVIEW,
        )

    return ValidationResult(
        validator_name="financial_validation",
        status=ValidationStatus.PASSED_WITH_WARNINGS,
        checks_performed=checks_performed,
        issues=issues,
        recommendation=RecommendedAction.AUTO_APPROVE,
    )
