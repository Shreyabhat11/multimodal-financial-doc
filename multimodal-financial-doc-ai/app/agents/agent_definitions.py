"""
Agent definitions: Extraction Validator, Financial Validator, Anomaly Analyst, Final
Reviewer (brief, Section 9).

Each agent's `backstory` is written to do real prompt-engineering work, not just
flavor text - it tells the agent what NOT to do (recompute arithmetic itself, invent
values) at least as much as what to do. This matters more for CrewAI agents than it
might for a single-shot prompt, because an under-constrained agent with tool access
and multiple reasoning steps has more opportunities to drift into "let me just
estimate this" than a single-turn extraction call does.

`allow_delegation=False` on every agent is a deliberate choice: CrewAI supports
agents delegating sub-tasks to each other, which is powerful but adds
nondeterminism and cost we don't want in a validation pipeline where we need
predictable, auditable behavior - the Process.sequential crew (crew.py's
`build_validation_crew`) already gives each agent the prior agents' output via task
context, so delegation isn't needed to share information.
"""

from __future__ import annotations

from crewai import LLM, Agent


def build_llm(settings) -> LLM:
    """Build the CrewAI LLM object from settings. CrewAI/LiteLLM model strings are
    provider-prefixed (e.g. "anthropic/claude-sonnet-4-6"), which is why we prepend
    the provider here rather than storing the prefixed string directly in config -
    keeps AGENT_LLM_MODEL in .env human-readable as just a model name.
    """
    provider_prefix = {"anthropic": "anthropic", "openai": "openai", "ollama": "ollama"}[settings.agent_llm_provider]
    return LLM(
        model=f"{provider_prefix}/{settings.agent_llm_model}",
        temperature=0.1,
    )


def build_extraction_validator(llm: LLM, *, verbose: bool) -> Agent:
    return Agent(
        role="Extraction Validator",
        goal=(
            "Identify missing required fields, malformed values, and internally inconsistent "
            "data in the extracted financial document, based ONLY on the document summary "
            "provided to you - do not guess at values that weren't given."
        ),
        backstory=(
            "You are a meticulous document QA specialist. You have reviewed thousands of "
            "extracted financial statements and have a sharp eye for the difference between "
            "'this field is genuinely missing' and 'this field exists but looks wrong.' You "
            "never fabricate a value to fill a gap - an empty field is reported as missing, "
            "not silently ignored or guessed at."
        ),
        llm=llm,
        tools=[],
        allow_delegation=False,
        verbose=verbose,
        max_iter=5,
    )


def build_financial_validator(llm: LLM, tools: list, *, verbose: bool) -> Agent:
    return Agent(
        role="Financial Validator",
        goal=(
            "Determine whether this document's balances and totals reconcile correctly, using "
            "the calculation tools available to you - never estimate or recompute arithmetic "
            "yourself, always call a tool for any numeric check."
        ),
        backstory=(
            "You are a financial auditor. You have learned, the hard way, that mental "
            "arithmetic on financial figures is unreliable even for a careful reviewer - so "
            "you ALWAYS use your calculation tools for every numeric check, and you report "
            "exactly what they return. You never say a balance 'looks about right' - either "
            "the tool confirms it reconciles within tolerance, or it doesn't."
        ),
        llm=llm,
        tools=[t for t in tools if t.name in ("calculate_total", "calculate_balance", "check_reported_totals")],
        allow_delegation=False,
        verbose=verbose,
        max_iter=6,
    )


def build_anomaly_analyst(llm: LLM, tools: list, *, verbose: bool) -> Agent:
    return Agent(
        role="Anomaly Analyst",
        goal=(
            "Investigate this document's transactions for duplicates, suspicious patterns, "
            "date issues, and inconsistencies, using your detection tools, and assess how "
            "concerning each finding actually is in context."
        ),
        backstory=(
            "You are a fraud-and-error analyst for financial documents. You know that "
            "automated anomaly detectors flag real issues but also flag benign patterns (a "
            "large legitimate rent payment, same-day transactions printed out of strict "
            "order) - your job is to run the detection tools, then use judgment about which "
            "findings genuinely warrant concern versus which are explainable, rather than "
            "escalating every tool finding uncritically."
        ),
        llm=llm,
        tools=[
            t
            for t in tools
            if t.name
            in (
                "check_duplicate_transactions",
                "check_transaction_consistency",
                "validate_dates",
                "detect_anomalies",
            )
        ],
        allow_delegation=False,
        verbose=verbose,
        max_iter=6,
    )


def build_final_reviewer(llm: LLM, *, verbose: bool) -> Agent:
    return Agent(
        role="Final Reviewer",
        goal=(
            "Combine the extraction validation, financial validation, and anomaly analysis "
            "findings into ONE final structured verdict for this document - a single status, "
            "a consolidated issue list, and one clear recommendation."
        ),
        backstory=(
            "You are the senior reviewer who signs off on automated financial document "
            "processing. You synthesize your team's findings into a decision a downstream "
            "system can act on directly. You are conservative: if ANY prior finding indicates "
            "a high-severity issue, your recommendation is human_review, not auto_approve - "
            "when in doubt, you route to a human rather than optimistically approving."
        ),
        llm=llm,
        tools=[],
        allow_delegation=False,
        verbose=verbose,
        max_iter=4,
    )
