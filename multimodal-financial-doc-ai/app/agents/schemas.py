"""
Structured output contract for the validation crew.

CRITICAL: this is a Pydantic model passed to CrewAI's `Task(output_pydantic=...)`.
CrewAI enforces the LLM's final response against this schema (retrying/repairing if
the model doesn't comply) — this is what guarantees the crew's output is the compact
structured JSON shape from brief Section 10, never raw chain-of-thought or free text,
regardless of how verbose the underlying agent reasoning was internally.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CrewIssue(BaseModel):
    field: str
    severity: str = Field(..., description="low | medium | high | critical")
    message: str


class CrewValidationOutput(BaseModel):
    status: str = Field(..., description="passed | passed_with_warnings | failed")
    checks_performed: list[str] = Field(default_factory=list)
    issues: list[CrewIssue] = Field(default_factory=list)
    recommendation: str = Field(..., description="auto_approve | human_review | reprocess | reject")
