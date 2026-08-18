"""
run_anomaly_detection - combines all pattern-based deterministic checks (duplicates,
date issues, running-balance consistency, large-transaction outliers, dual-sided
rows) into one ValidationResult.

Severity-to-status mapping is deliberately conservative: only CRITICAL/HIGH severity
anomalies fail the validation outright. MEDIUM/LOW severity anomalies are surfaced
(PASSED_WITH_WARNINGS) but don't block auto-approval on their own — e.g. a single
unusually large transaction is common and often legitimate (a rent payment, a payroll
deposit); it deserves visibility, not automatic rejection. This threshold is a
config-driven policy choice, not a hardcoded opinion — see the severity check below,
which reads directly off each Anomaly's `.severity` field so the policy stays in one
place (Anomaly construction sites in duplicates.py/dates.py/outliers.py/
balance_consistency.py) rather than being re-derived here.
"""

from __future__ import annotations

from decimal import Decimal

from app.schemas.document import StatementPeriod
from app.schemas.enums import RecommendedAction, Severity, ValidationStatus
from app.schemas.transaction import Transaction
from app.schemas.validation import ValidationResult
from app.validation.balance_consistency import validate_running_balance_consistency
from app.validation.dates import validate_transaction_dates
from app.validation.duplicates import find_duplicate_transactions
from app.validation.outliers import find_dual_sided_transactions, find_unusually_large_transactions

_BLOCKING_SEVERITIES = (Severity.HIGH, Severity.CRITICAL)


def run_anomaly_detection(
    *,
    transactions: list[Transaction],
    statement_period: StatementPeriod,
    large_transaction_multiplier: float = 5.0,
    balance_tolerance: Decimal = Decimal("0.01"),
) -> ValidationResult:
    checks_performed = [
        "duplicate_transaction_detection",
        "date_validation",
        "running_balance_consistency",
        "large_transaction_outlier_detection",
        "debit_credit_consistency",
    ]

    anomalies = (
        find_duplicate_transactions(transactions)
        + validate_transaction_dates(transactions, statement_period)
        + validate_running_balance_consistency(transactions, tolerance=balance_tolerance)
        + find_unusually_large_transactions(transactions, multiplier=large_transaction_multiplier)
        + find_dual_sided_transactions(transactions)
    )

    has_blocking_anomaly = any(a.severity in _BLOCKING_SEVERITIES for a in anomalies)

    if not anomalies:
        status = ValidationStatus.PASSED
        recommendation = RecommendedAction.AUTO_APPROVE
    elif has_blocking_anomaly:
        status = ValidationStatus.FAILED
        recommendation = RecommendedAction.HUMAN_REVIEW
    else:
        status = ValidationStatus.PASSED_WITH_WARNINGS
        recommendation = RecommendedAction.AUTO_APPROVE

    return ValidationResult(
        validator_name="anomaly_detection",
        status=status,
        checks_performed=checks_performed,
        anomalies=anomalies,
        recommendation=recommendation,
    )
