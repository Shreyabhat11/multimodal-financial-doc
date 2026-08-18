"""
Task definitions for the validation crew.

Process.sequential (crew.py) means these tasks run in the order defined below, and
each task after the first receives the prior tasks' outputs via CrewAI's `context`
parameter - this is how findings flow from Extraction Validator -> Financial
Validator -> Anomaly Analyst -> Final Reviewer without needing a shared external
state store. It also means the Financial Validator and Anomaly Analyst tasks run
AFTER the Extraction Validator, so if extraction validation finds the document is
missing its account_number entirely, later agents still run (deliberately - a
missing account number doesn't mean the transaction list itself is unreliable) but
have that context available.

Every task's `expected_output` explicitly says "structured JSON, no explanation
outside the JSON" - reinforcing at the task level (not just the final agent's output
schema) that intermediate reasoning should stay internal to the agent's tool-calling
loop and not leak into what gets reported.
"""

from __future__ import annotations

from crewai import Agent, Task

from app.agents.schemas import CrewValidationOutput


def build_extraction_validation_task(agent: Agent, *, document_summary: str) -> Task:
    return Task(
        description=(
            "Review this extracted financial document summary for missing required fields, "
            "malformed values, or internal inconsistencies (e.g. a transaction count of zero "
            "on a multi-page statement, a currency code that isn't a valid 3-letter code).\n\n"
            f"Document summary:\n{document_summary}\n\n"
            "List every issue you find, or state clearly that you found none."
        ),
        expected_output=(
            "A structured list of issues (field, severity, message), or confirmation that no "
            "issues were found. No explanation outside of the listed issues."
        ),
        agent=agent,
    )


def build_financial_validation_task(agent: Agent, *, document_summary: str, extraction_task: Task) -> Task:
    return Task(
        description=(
            "Using your calculation tools (never mental arithmetic), determine whether this "
            "document's balances and totals reconcile. Call calculate_balance and "
            "check_reported_totals, and report exactly what they return.\n\n"
            f"Document summary:\n{document_summary}"
        ),
        expected_output=(
            "A structured report of what calculate_balance and check_reported_totals returned, "
            "and whether the document's financials reconcile. No explanation outside the report."
        ),
        agent=agent,
        context=[extraction_task],
    )


def build_anomaly_analysis_task(agent: Agent, *, document_summary: str, extraction_task: Task) -> Task:
    return Task(
        description=(
            "Using your detection tools, investigate this document for duplicate transactions, "
            "date issues, running-balance inconsistencies, and unusually large transactions. "
            "For each finding, briefly assess whether it looks like a genuine concern or a "
            "plausible benign pattern.\n\n"
            f"Document summary:\n{document_summary}"
        ),
        expected_output=(
            "A structured list of anomaly findings with your assessment of each, or confirmation "
            "that no anomalies were found. No explanation outside the findings."
        ),
        agent=agent,
        context=[extraction_task],
    )


def build_final_review_task(
    agent: Agent,
    *,
    financial_task: Task,
    anomaly_task: Task,
    extraction_task: Task,
) -> Task:
    return Task(
        description=(
            "Combine the extraction validation, financial validation, and anomaly analysis "
            "findings into ONE final verdict. Set status to 'failed' if financial validation "
            "did not reconcile or any high/critical severity anomaly was found; "
            "'passed_with_warnings' if there are only low/medium severity issues; 'passed' if "
            "there are no issues at all. Set recommendation to 'human_review' whenever status "
            "is 'failed', otherwise 'auto_approve'."
        ),
        expected_output=(
            "A single JSON object with fields: status, checks_performed (list of strings), "
            "issues (list of {field, severity, message}), recommendation. Output ONLY this "
            "JSON object, no other text."
        ),
        agent=agent,
        context=[extraction_task, financial_task, anomaly_task],
        output_pydantic=CrewValidationOutput,
    )
