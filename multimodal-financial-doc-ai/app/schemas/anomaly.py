"""Anomaly schema — output of app.validation.anomaly_detection (Phase 11)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.enums import AnomalyType, Severity


class Anomaly(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    anomaly_type: AnomalyType
    severity: Severity
    message: str = Field(..., min_length=1, max_length=1024)
    affected_transaction_indices: list[int] = Field(
        default_factory=list,
        description="Indices into the document's transactions list that this anomaly concerns.",
    )
    field: str | None = Field(
        default=None, description="Document-level field this anomaly concerns, if not transaction-specific."
    )

    @classmethod
    def duplicate(cls, indices: list[int], description: str) -> "Anomaly":
        return cls(
            anomaly_type=AnomalyType.DUPLICATE_TRANSACTION,
            severity=Severity.MEDIUM,
            message=f"Possible duplicate transaction(s) detected: {description}",
            affected_transaction_indices=indices,
        )

    @classmethod
    def balance_mismatch(cls, expected, reported, tolerance) -> "Anomaly":
        return cls(
            anomaly_type=AnomalyType.BALANCE_RECONCILIATION_MISMATCH,
            severity=Severity.HIGH,
            message=(
                f"Expected closing balance {expected} (opening + credits - debits) does not "
                f"match reported closing balance {reported} within tolerance {tolerance}."
            ),
            field="closing_balance",
        )
