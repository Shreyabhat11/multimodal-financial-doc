"""
ReAct tools for the validation crew.

This module implements Section 10 of the brief (ReAct tool-use) and is deliberately
built alongside the agents (Section 9) rather than as a separate later phase, because
the two are inseparable in practice: a CrewAI agent's "ReAct reasoning" IS its
tool-calling loop — there's no meaningful way to build/test one without the other.

CRITICAL DESIGN POINT: every tool here is a thin wrapper that calls into
app.validation.* (Phase 8) — the deterministic, pure-Python arithmetic layer. No tool
here does its own math. The tool's job is only to (a) expose Phase 8's functions to
the agent's tool-calling loop and (b) format the result as a compact string the LLM
can reason over. This is what makes the earlier architectural claim
("agents interpret validation results, they don't compute them") literally true in
the code, not just in the docs — if you trace any of these tools, the arithmetic
bottoms out in app/validation/financial.py, never in an LLM call.

Factory pattern: `build_validation_tools(document)` returns a fresh list of tools
bound (via closure) to ONE document's data. This means agent tool calls can be
argument-free/near-argument-free ("call calculate_balance, no parameters needed") —
matching the brief's tool signatures exactly (`calculate_total()`,
`calculate_balance()`, no document payload in the call) — because the document
context is already captured in the closure rather than needing to be serialized into
every tool call the LLM makes. This also keeps LLM context usage down: the agent
never has to paste a 200-row transaction list into a tool-call argument.
"""

from __future__ import annotations

import json
from decimal import Decimal

from crewai.tools import tool

from app.schemas.document import FinancialTotals, StatementPeriod
from app.schemas.transaction import Transaction
from app.validation.anomaly_detection import run_anomaly_detection
from app.validation.balance_consistency import validate_running_balance_consistency
from app.validation.dates import validate_transaction_dates
from app.validation.duplicates import find_duplicate_transactions
from app.validation.financial import calculate_transaction_totals, reconcile_balance, reconcile_reported_totals
from app.validation.outliers import find_dual_sided_transactions, find_unusually_large_transactions


def _anomalies_to_json(anomalies) -> str:
    return json.dumps(
        [
            {
                "type": a.anomaly_type.value,
                "severity": a.severity.value,
                "message": a.message,
                "affected_transaction_indices": a.affected_transaction_indices,
            }
            for a in anomalies
        ]
    )


def build_validation_tools(
    *,
    opening_balance: Decimal,
    closing_balance: Decimal,
    reported_totals: FinancialTotals | None,
    transactions: list[Transaction],
    statement_period: StatementPeriod,
    balance_tolerance: Decimal = Decimal("0.01"),
    large_transaction_multiplier: float = 5.0,
):
    """Build the six ReAct tools for one document's validation crew run.

    Each tool returns a short, structured (JSON) string — never a natural-language
    explanation with embedded reasoning — because the tool's return value becomes
    part of the LLM's context for its NEXT reasoning step. Keeping it structured
    keeps the agent's downstream reasoning grounded in facts rather than in whatever
    narrative phrasing a previous free-text tool response happened to use.
    """

    @tool("calculate_total")
    def calculate_total() -> str:
        """Compute the total debits and total credits across all transactions in
        this document, calculated directly from the transaction list (not from any
        totals line the document itself printed)."""
        totals = calculate_transaction_totals(transactions)
        return json.dumps({"total_debits": str(totals.total_debits), "total_credits": str(totals.total_credits)})

    @tool("calculate_balance")
    def calculate_balance() -> str:
        """Check whether opening_balance + total_credits - total_debits reconciles
        with the document's reported closing_balance, within the configured
        tolerance. Returns the expected closing balance, the reported closing
        balance, the difference, and whether they reconcile."""
        totals = calculate_transaction_totals(transactions)
        result = reconcile_balance(
            opening_balance=opening_balance,
            reported_closing_balance=closing_balance,
            totals=totals,
            tolerance=balance_tolerance,
        )
        return json.dumps(
            {
                "opening_balance": str(result.opening_balance),
                "reported_closing_balance": str(result.reported_closing_balance),
                "expected_closing_balance": str(result.expected_closing_balance),
                "difference": str(result.difference),
                "tolerance": str(result.tolerance),
                "is_reconciled": result.is_reconciled,
            }
        )

    @tool("check_reported_totals")
    def check_reported_totals() -> str:
        """Compare the document's own printed totals line (if any) against totals
        computed from the transaction list. Returns whether they match."""
        result = reconcile_reported_totals(
            transactions=transactions, reported_totals=reported_totals, tolerance=balance_tolerance
        )
        return json.dumps(
            {
                "computed_total_debits": str(result.computed_totals.total_debits),
                "computed_total_credits": str(result.computed_totals.total_credits),
                "reported_total_debits": str(result.reported_totals.total_debits) if result.reported_totals else None,
                "reported_total_credits": str(result.reported_totals.total_credits)
                if result.reported_totals
                else None,
                "is_reconciled": result.is_reconciled,
            }
        )

    @tool("check_duplicate_transactions")
    def check_duplicate_transactions() -> str:
        """Detect transactions that are exact duplicates of each other (same date,
        description, amount, and reference). Returns a list of duplicate groups found."""
        anomalies = find_duplicate_transactions(transactions)
        return _anomalies_to_json(anomalies)

    @tool("check_transaction_consistency")
    def check_transaction_consistency() -> str:
        """Check whether each transaction's printed running balance (if present) is
        consistent with the opening balance plus the net effect of preceding
        transactions. Also flags transactions with a nonzero value in both debit and
        credit fields."""
        anomalies = validate_running_balance_consistency(
            transactions, tolerance=balance_tolerance
        ) + find_dual_sided_transactions(transactions)
        return _anomalies_to_json(anomalies)

    @tool("validate_dates")
    def validate_dates() -> str:
        """Check every transaction date against the statement period and check for
        implausible date sequencing (e.g. a large backward jump)."""
        anomalies = validate_transaction_dates(transactions, statement_period)
        return _anomalies_to_json(anomalies)

    @tool("detect_anomalies")
    def detect_anomalies() -> str:
        """Run the full deterministic anomaly-detection suite (duplicates, dates,
        running-balance consistency, unusually large transactions) in one call and
        return the combined list plus an overall status. Use this for a
        comprehensive pass; use the individual check_* / validate_* tools when you
        want to investigate one specific concern in more depth."""
        result = run_anomaly_detection(
            transactions=transactions,
            statement_period=statement_period,
            large_transaction_multiplier=large_transaction_multiplier,
            balance_tolerance=balance_tolerance,
        )
        return json.dumps(
            {
                "status": result.status.value,
                "anomalies": [
                    {
                        "type": a.anomaly_type.value,
                        "severity": a.severity.value,
                        "message": a.message,
                        "affected_transaction_indices": a.affected_transaction_indices,
                    }
                    for a in result.anomalies
                ],
            }
        )

    return [
        calculate_total,
        calculate_balance,
        check_reported_totals,
        check_duplicate_transactions,
        check_transaction_consistency,
        validate_dates,
        detect_anomalies,
    ]
