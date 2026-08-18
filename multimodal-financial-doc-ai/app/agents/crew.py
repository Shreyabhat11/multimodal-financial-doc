"""
run_crew_validation - the entry point the LangGraph `crew_validation` node (Phase 7
extension) calls.

This is the ONLY function outside this package that other code should import from
app.agents - everything else (tools.py, agent_definitions.py, task_definitions.py,
schemas.py) is an internal implementation detail of how the crew is built, mirroring
the same "one entry point, hide the internals" discipline as model_factory.py for the
VLM layer.

IMPORTANT (brief requirement): this function NEVER returns or logs the crew's raw
intermediate reasoning traces (CrewAI's verbose output / agent "Thought:" steps). It
returns only the final structured ValidationResult, built from the Final Reviewer's
`output_pydantic`-enforced CrewValidationOutput. If `settings.crew_verbose` is True,
CrewAI's internal reasoning IS printed to stdout during execution (useful for local
debugging) but that is a developer-facing console stream, never something this
function returns to application code - keep `crew_verbose=False` (the production
default) to ensure nothing reasoning-shaped reaches logs or API responses.
"""

from __future__ import annotations

import json
from decimal import Decimal

from crewai import Crew, Process

from app.agents.agent_definitions import (
    build_anomaly_analyst,
    build_extraction_validator,
    build_final_reviewer,
    build_financial_validator,
    build_llm,
)
from app.agents.schemas import CrewValidationOutput
from app.agents.task_definitions import (
    build_anomaly_analysis_task,
    build_extraction_validation_task,
    build_final_review_task,
    build_financial_validation_task,
)
from app.agents.tools import build_validation_tools
from app.core.config import Settings, get_settings
from app.core.exceptions import CrewValidationError
from app.schemas.document import Account, FinancialTotals, StatementPeriod
from app.schemas.enums import RecommendedAction, ValidationStatus
from app.schemas.transaction import Transaction
from app.schemas.validation import ValidationIssue, ValidationResult

_STATUS_MAP = {
    "passed": ValidationStatus.PASSED,
    "passed_with_warnings": ValidationStatus.PASSED_WITH_WARNINGS,
    "failed": ValidationStatus.FAILED,
}
_RECOMMENDATION_MAP = {
    "auto_approve": RecommendedAction.AUTO_APPROVE,
    "human_review": RecommendedAction.HUMAN_REVIEW,
    "reprocess": RecommendedAction.REPROCESS,
    "reject": RecommendedAction.REJECT,
}


def _build_document_summary(
    *,
    account: Account,
    statement_period: StatementPeriod,
    opening_balance: Decimal,
    closing_balance: Decimal,
    currency: str,
    transaction_count: int,
    reported_totals: FinancialTotals | None,
) -> str:
    """Build the compact document summary given to agents in their task descriptions.

    Deliberately does NOT include the full transaction list - agents that need
    transaction-level detail get it through their tools (which already have the full
    list bound via closure), not through the prompt. This keeps every task
    description's token cost roughly constant regardless of whether the document has
    5 or 500 transactions.
    """
    return json.dumps(
        {
            "account_holder": account.account_holder,
            "account_number_masked": account.masked_account_number,
            "bank_name": account.bank_name,
            "statement_period": {
                "start_date": str(statement_period.start_date),
                "end_date": str(statement_period.end_date),
            },
            "opening_balance": str(opening_balance),
            "closing_balance": str(closing_balance),
            "currency": currency,
            "transaction_count": transaction_count,
            "document_reported_totals": (
                {
                    "total_debits": str(reported_totals.total_debits),
                    "total_credits": str(reported_totals.total_credits),
                }
                if reported_totals
                else None
            ),
        },
        indent=2,
    )


def build_validation_crew(
    *,
    account: Account,
    statement_period: StatementPeriod,
    opening_balance: Decimal,
    closing_balance: Decimal,
    currency: str,
    transactions: list[Transaction],
    reported_totals: FinancialTotals | None,
    settings: Settings,
) -> Crew:
    llm = build_llm(settings)
    tools = build_validation_tools(
        opening_balance=opening_balance,
        closing_balance=closing_balance,
        reported_totals=reported_totals,
        transactions=transactions,
        statement_period=statement_period,
        balance_tolerance=Decimal(str(settings.balance_tolerance)),
        large_transaction_multiplier=settings.large_transaction_multiplier,
    )
    summary = _build_document_summary(
        account=account,
        statement_period=statement_period,
        opening_balance=opening_balance,
        closing_balance=closing_balance,
        currency=currency,
        transaction_count=len(transactions),
        reported_totals=reported_totals,
    )

    extraction_validator = build_extraction_validator(llm, verbose=settings.crew_verbose)
    financial_validator = build_financial_validator(llm, tools, verbose=settings.crew_verbose)
    anomaly_analyst = build_anomaly_analyst(llm, tools, verbose=settings.crew_verbose)
    final_reviewer = build_final_reviewer(llm, verbose=settings.crew_verbose)

    extraction_task = build_extraction_validation_task(extraction_validator, document_summary=summary)
    financial_task = build_financial_validation_task(
        financial_validator, document_summary=summary, extraction_task=extraction_task
    )
    anomaly_task = build_anomaly_analysis_task(
        anomaly_analyst, document_summary=summary, extraction_task=extraction_task
    )
    final_task = build_final_review_task(
        final_reviewer, financial_task=financial_task, anomaly_task=anomaly_task, extraction_task=extraction_task
    )

    return Crew(
        agents=[extraction_validator, financial_validator, anomaly_analyst, final_reviewer],
        tasks=[extraction_task, financial_task, anomaly_task, final_task],
        process=Process.sequential,
        verbose=settings.crew_verbose,
    )


def _crew_output_to_validation_result(output: CrewValidationOutput) -> ValidationResult:
    status = _STATUS_MAP.get(output.status.lower(), ValidationStatus.FAILED)
    recommendation = _RECOMMENDATION_MAP.get(output.recommendation.lower(), RecommendedAction.HUMAN_REVIEW)
    return ValidationResult(
        validator_name="crew_validation",
        status=status,
        checks_performed=output.checks_performed,
        issues=[ValidationIssue(field=i.field, severity=i.severity, message=i.message) for i in output.issues],
        recommendation=recommendation,
    )


def run_crew_validation(
    *,
    account: Account,
    statement_period: StatementPeriod,
    opening_balance: Decimal,
    closing_balance: Decimal,
    currency: str,
    transactions: list[Transaction],
    reported_totals: FinancialTotals | None,
    settings: Settings | None = None,
) -> ValidationResult:
    """Run the four-agent validation crew and return a ValidationResult.

    Raises CrewValidationError (not a bare exception) on failure - e.g. if the LLM
    provider is unreachable, or if the Final Reviewer's output can't be coerced into
    CrewValidationOutput even after CrewAI's built-in structured-output retry. The
    LangGraph `crew_validation` node (Phase 7 extension) catches this specifically and
    routes to human review rather than letting a crew infrastructure failure crash
    the whole pipeline - a crew that can't run is itself a signal for human review,
    not a document-processing failure.
    """
    settings = settings or get_settings()

    crew = build_validation_crew(
        account=account,
        statement_period=statement_period,
        opening_balance=opening_balance,
        closing_balance=closing_balance,
        currency=currency,
        transactions=transactions,
        reported_totals=reported_totals,
        settings=settings,
    )

    try:
        result = crew.kickoff()
    except Exception as exc:
        raise CrewValidationError(f"Validation crew execution failed: {exc}") from exc

    output = result.pydantic
    if not isinstance(output, CrewValidationOutput):
        raise CrewValidationError(
            "Validation crew did not produce a structured CrewValidationOutput.",
            details={"raw_output": str(result.raw)[:1000]},
        )

    return _crew_output_to_validation_result(output)
