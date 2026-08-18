"""Date validation.

Checks required by the brief: invalid dates (handled upstream - Transaction's
`_coerce_date` validator, Phase 2, already rejects genuinely unparseable dates at
schema-construction time, so a Transaction object reaching this module always has a
syntactically valid date), transactions outside the statement period, and impossible
date sequences.
"""

from __future__ import annotations

from datetime import timedelta

from app.schemas.anomaly import Anomaly
from app.schemas.document import StatementPeriod
from app.schemas.enums import AnomalyType, Severity
from app.schemas.transaction import Transaction


def find_transactions_outside_period(
    transactions: list[Transaction],
    statement_period: StatementPeriod,
) -> list[Anomaly]:
    """Flag any transaction dated before the statement's start_date or after its
    end_date. A statement listing a transaction outside its own declared period
    usually indicates either a misread date (VLM transposed digits) or a misread
    statement_period itself - either way, worth a human's attention rather than
    silently accepting it."""
    anomalies: list[Anomaly] = []
    out_of_range_indices = [
        i
        for i, txn in enumerate(transactions)
        if txn.date < statement_period.start_date or txn.date > statement_period.end_date
    ]
    if out_of_range_indices:
        anomalies.append(
            Anomaly(
                anomaly_type=AnomalyType.DATE_OUT_OF_STATEMENT_PERIOD,
                severity=Severity.MEDIUM,
                message=(
                    f"{len(out_of_range_indices)} transaction(s) fall outside the statement "
                    f"period {statement_period.start_date} to {statement_period.end_date}."
                ),
                affected_transaction_indices=out_of_range_indices,
            )
        )
    return anomalies


def find_date_sequence_anomalies(transactions: list[Transaction]) -> list[Anomaly]:
    """Flag large backward jumps in transaction date ordering.

    Statement transactions are almost always printed in chronological (or reverse
    chronological) order. We don't hard-require strict ordering - some statement
    formats interleave same-day transactions in a non-obvious sub-order, and that's
    normal - but a large date regression (e.g. the 15th appearing between two
    transactions dated the 3rd) is a strong signal that either a row was misread or
    rows from a different page got merged out of order. We flag jumps backward of
    more than a few days as MEDIUM severity, letting the anomaly analyst (Phase 9)
    weigh it against everything else rather than treating it as a hard failure -
    the false-positive rate for "any" backward movement would be too high given how
    often same-day transactions are printed out of strict order.
    """
    anomalies: list[Anomaly] = []
    backward_jump_threshold = timedelta(days=3)

    flagged_indices: list[int] = []
    for i in range(1, len(transactions)):
        delta = transactions[i - 1].date - transactions[i].date
        if delta > backward_jump_threshold:
            flagged_indices.append(i)

    if flagged_indices:
        anomalies.append(
            Anomaly(
                anomaly_type=AnomalyType.INVALID_DATE,
                severity=Severity.LOW,
                message=(
                    f"{len(flagged_indices)} transaction(s) show a date more than "
                    f"{backward_jump_threshold.days} days earlier than the immediately preceding "
                    "transaction - possible out-of-order or misread rows."
                ),
                affected_transaction_indices=flagged_indices,
            )
        )
    return anomalies


def validate_transaction_dates(
    transactions: list[Transaction],
    statement_period: StatementPeriod,
) -> list[Anomaly]:
    """Run all date checks and return the combined anomaly list."""
    return find_transactions_outside_period(transactions, statement_period) + find_date_sequence_anomalies(
        transactions
    )
