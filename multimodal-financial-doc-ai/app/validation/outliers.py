"""Large-transaction outlier detection and debit/credit consistency.

`find_unusually_large_transactions` uses the MEDIAN transaction amount as the
baseline, not the mean — the mean is easily skewed by a single large legitimate
transaction (e.g. a mortgage payment on an otherwise low-value checking account),
which would raise the threshold and make the check less sensitive right when a
genuinely large outlier appears. Median is robust to exactly that kind of skew.
"""

from __future__ import annotations

import statistics
from decimal import Decimal

from app.schemas.anomaly import Anomaly
from app.schemas.enums import AnomalyType, Severity
from app.schemas.transaction import Transaction

_MIN_TRANSACTIONS_FOR_OUTLIER_CHECK = 5  # below this, "median" is too noisy to be meaningful


def find_unusually_large_transactions(
    transactions: list[Transaction],
    *,
    multiplier: float = 5.0,
) -> list[Anomaly]:
    """Flag transactions whose absolute amount exceeds `multiplier` times the median
    absolute transaction amount in the document."""
    amounts = [abs(t.net_amount) for t in transactions if t.net_amount != 0]
    if len(amounts) < _MIN_TRANSACTIONS_FOR_OUTLIER_CHECK:
        return []

    median_amount = statistics.median(amounts)
    if median_amount == 0:
        return []

    threshold = median_amount * Decimal(str(multiplier))
    flagged_indices = [i for i, t in enumerate(transactions) if abs(t.net_amount) > threshold]

    if not flagged_indices:
        return []

    return [
        Anomaly(
            anomaly_type=AnomalyType.UNUSUALLY_LARGE_TRANSACTION,
            severity=Severity.MEDIUM,
            message=(
                f"{len(flagged_indices)} transaction(s) exceed {multiplier}x the document's "
                f"median transaction amount ({median_amount})."
            ),
            affected_transaction_indices=flagged_indices,
        )
    ]


def find_dual_sided_transactions(transactions: list[Transaction]) -> list[Anomaly]:
    """Flag transactions that report a nonzero value in BOTH debit and credit — an
    uncommon but not always erroneous pattern (see Transaction.is_dual_sided,
    Phase 2), surfaced here as its own low-severity anomaly so it's visible to
    review rather than silently passing through."""
    flagged_indices = [i for i, t in enumerate(transactions) if t.is_dual_sided]
    if not flagged_indices:
        return []
    return [
        Anomaly(
            anomaly_type=AnomalyType.DEBIT_CREDIT_INCONSISTENCY,
            severity=Severity.LOW,
            message=(
                f"{len(flagged_indices)} transaction(s) report a nonzero value in both debit "
                "and credit fields, which is unusual."
            ),
            affected_transaction_indices=flagged_indices,
        )
    ]
