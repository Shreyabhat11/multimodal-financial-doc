"""ValidationResult schema.

One ``ValidationResult`` is produced by each validation stage (financial_validation,
anomaly_detection, crew_validation — see the LangGraph nodes in Phase 7) and they are
collected into a list on ``DocumentState.validation_results``. ``finalize_result``
combines them into the single top-level status exposed on ``FinalExtractionResult``.
Having each stage emit the *same* schema (rather than stage-specific ad hoc dicts) is
what lets `finalize_result` merge them generically instead of needing special-case
code per validator.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.anomaly import Anomaly
from app.schemas.enums import RecommendedAction, ValidationStatus


class ValidationIssue(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    field: str
    severity: str = Field(..., description="low | medium | high | critical")
    message: str


class ValidationResult(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    validator_name: str = Field(
        ..., description="Which validator produced this, e.g. 'financial_validation', 'crew_validation'."
    )
    status: ValidationStatus
    checks_performed: list[str] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    anomalies: list[Anomaly] = Field(default_factory=list)
    recommendation: RecommendedAction = RecommendedAction.AUTO_APPROVE

    @property
    def has_high_severity_issue(self) -> bool:
        return any(issue.severity in ("high", "critical") for issue in self.issues)

    @classmethod
    def passed(cls, validator_name: str, checks_performed: list[str]) -> "ValidationResult":
        return cls(
            validator_name=validator_name,
            status=ValidationStatus.PASSED,
            checks_performed=checks_performed,
            recommendation=RecommendedAction.AUTO_APPROVE,
        )
