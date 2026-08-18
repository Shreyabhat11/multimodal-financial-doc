"""Running balance consistency.

Many bank statement formats print a running balance alongside every transaction row
(Transaction.balance, Phase 2). When present, this is a second, INDEPENDENT source of
truth beyond the document's opening/closing balance figures: balance[i] should equal
balance[i-1] + net_amount(transaction[i]) for every consecutive pair where both rows
report a balance. Checking this catches errors that pure opening/closing reconciliation
can miss entirely — e.g. two transactions with swapped debit/credit amounts that
happen to still sum to the correct total_debits/total_credits (and would therefore
still reconcile at the document level) will nonetheless break the running balance at
the specific row where the swap occurred, pinpointing exactly where the error is.
"""

from __future__ import annotations

from decimal import Decimal

from app.schemas.anomaly import Anomaly
from app.schemas.enums import AnomalyType, Severity
from app.schemas.transaction import Transaction


def validate_running_balance_consistency(
    transactions: list[Transaction],
    *,
    tolerance: Decimal = Decimal("0.01"),
) -> list[Anomaly]:
    """Check balance[i] == balance[i-1] + net_amount(transaction[i]) for every
    consecutive pair of transactions that both report a running balance.

    Transactions missing a `balance` value are skipped as anchors — we only compare
    against the most recent transaction that DID report a balance, so sparse balance
    reporting (some statement formats only print it every few rows) doesn't produce
    false positives.
    """
    anomalies: list[Anomaly] = []
    flagged_indices: list[int] = []

    last_known_balance: Decimal | None = None
    last_known_index: int | None = None

    for i, txn in enumerate(transactions):
        if txn.balance is None:
            continue

        if last_known_balance is not None:
            # Sum net amounts of every transaction between the last anchor and this one
            # (inclusive of this one), in case some intermediate rows lacked a balance.
            net_since_anchor = sum(
                (t.net_amount for t in transactions[last_known_index + 1 : i + 1]), Decimal("0")
            )
            expected_balance = last_known_balance + net_since_anchor
            if abs(txn.balance - expected_balance) > tolerance:
                flagged_indices.append(i)

        last_known_balance = txn.balance
        last_known_index = i

    if flagged_indices:
        anomalies.append(
            Anomaly(
                anomaly_type=AnomalyType.RUNNING_BALANCE_INCONSISTENCY,
                severity=Severity.HIGH,
                message=(
                    f"{len(flagged_indices)} transaction(s) have a running balance that doesn't "
                    "match opening balance + net amount of preceding transactions."
                ),
                affected_transaction_indices=flagged_indices,
            )
        )
    return anomalies
