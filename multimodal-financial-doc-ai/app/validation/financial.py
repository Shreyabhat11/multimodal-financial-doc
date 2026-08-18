"""
Deterministic financial validation: balance reconciliation and transaction totals.

This is the module the whole reliability story of the system rests on (see
ARCHITECTURE.md, Section 1b). Every function here is pure arithmetic on Decimal
values — no LLM call anywhere in this file. Agents (Phase 9) call these functions and
interpret their output; they do not replicate this math themselves.

Why Decimal, not float, one more time (it matters enough to repeat): float arithmetic
introduces binary rounding error (0.1 + 0.2 == 0.30000000000000004 in float, exactly
0.3 in Decimal). A balance reconciliation check built on floats would either need a
suspiciously large tolerance to absorb its own rounding noise, or would produce false
positives on perfectly correct statements. Decimal makes the tolerance parameter mean
what it says: genuine rounding/display differences in the source document, not
arithmetic error introduced by our own validation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.schemas.document import FinancialTotals
from app.schemas.transaction import Transaction


def calculate_transaction_totals(transactions: list[Transaction]) -> FinancialTotals:
    """Sum debits and credits across all transactions. This is the COMPUTED totals —
    kept distinct from whatever totals the document itself printed (which the VLM may
    have extracted separately); compare the two in `reconcile_reported_totals` below.
    """
    total_debits = sum((t.debit for t in transactions), Decimal("0"))
    total_credits = sum((t.credit for t in transactions), Decimal("0"))
    return FinancialTotals(total_debits=total_debits, total_credits=total_credits)


@dataclass
class BalanceReconciliationResult:
    opening_balance: Decimal
    reported_closing_balance: Decimal
    expected_closing_balance: Decimal
    difference: Decimal
    tolerance: Decimal
    is_reconciled: bool

    @property
    def abs_difference(self) -> Decimal:
        return abs(self.difference)


def reconcile_balance(
    *,
    opening_balance: Decimal,
    reported_closing_balance: Decimal,
    totals: FinancialTotals,
    tolerance: Decimal = Decimal("0.01"),
) -> BalanceReconciliationResult:
    """Check: opening_balance + total_credits - total_debits ≈ reported_closing_balance.

    `tolerance` absorbs genuine rounding differences some statement formats exhibit
    (e.g. a document that displays rounded-to-cent figures for numbers that were
    computed with more precision internally) — NOT arithmetic error, which Decimal
    already eliminates. A tolerance of $0.01 (the default) means "the numbers agree
    to the cent"; anything wider should be a deliberate, documented policy decision,
    not a default assumption.
    """
    expected_closing = opening_balance + totals.total_credits - totals.total_debits
    difference = reported_closing_balance - expected_closing
    return BalanceReconciliationResult(
        opening_balance=opening_balance,
        reported_closing_balance=reported_closing_balance,
        expected_closing_balance=expected_closing,
        difference=difference,
        tolerance=tolerance,
        is_reconciled=abs(difference) <= tolerance,
    )


@dataclass
class TotalsReconciliationResult:
    computed_totals: FinancialTotals
    reported_totals: FinancialTotals | None
    debits_difference: Decimal | None
    credits_difference: Decimal | None
    is_reconciled: bool  # True if no reported totals to compare against, or they match


def reconcile_reported_totals(
    *,
    transactions: list[Transaction],
    reported_totals: FinancialTotals | None,
    tolerance: Decimal = Decimal("0.01"),
) -> TotalsReconciliationResult:
    """Compare the totals computed from the transaction list against whatever totals
    line the document itself printed (if the VLM found and extracted one). A mismatch
    here is a different, and arguably more direct, signal than balance reconciliation
    failing — it means the transaction list itself doesn't sum to what the document
    claims, independent of any question about the opening/closing balance figures.
    """
    computed = calculate_transaction_totals(transactions)

    if reported_totals is None:
        return TotalsReconciliationResult(
            computed_totals=computed,
            reported_totals=None,
            debits_difference=None,
            credits_difference=None,
            is_reconciled=True,  # nothing to compare against — not a failure, just not checkable
        )

    debits_diff = reported_totals.total_debits - computed.total_debits
    credits_diff = reported_totals.total_credits - computed.total_credits
    is_reconciled = abs(debits_diff) <= tolerance and abs(credits_diff) <= tolerance

    return TotalsReconciliationResult(
        computed_totals=computed,
        reported_totals=reported_totals,
        debits_difference=debits_diff,
        credits_difference=credits_diff,
        is_reconciled=is_reconciled,
    )
