"""FinalExtractionResult — the schema returned by GET /documents/{id}/result and stored
as the canonical output of a processing run.

This is the schema referenced in the brief's example JSON (Section 6), extended with
the confidence and validation data that section 12/9 add on top. Everything else in
app/schemas/ exists to build this one object.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.confidence import DocumentConfidence
from app.schemas.document import Account, DocumentMetadata, FinancialTotals, StatementPeriod
from app.schemas.enums import DocumentType, ProcessingStatus
from app.schemas.transaction import Transaction
from app.schemas.validation import ValidationResult


class FinalExtractionResult(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    document_id: str
    document_type: DocumentType
    status: ProcessingStatus

    account: Account
    statement_period: StatementPeriod
    opening_balance: Decimal
    closing_balance: Decimal
    currency: str = Field(default="USD", min_length=3, max_length=3)

    transactions: list[Transaction] = Field(default_factory=list)
    totals: FinancialTotals

    validation_results: list[ValidationResult] = Field(default_factory=list)
    confidence: DocumentConfidence

    metadata: DocumentMetadata

    @model_validator(mode="after")
    def _currency_consistency(self) -> "FinalExtractionResult":
        # Not a hard failure - mixed-currency statements exist (e.g. multi-currency
        # credit cards) - but transactions reporting a currency different from the
        # document-level currency is exactly the kind of thing anomaly_detection
        # should be told about, so we don't silently normalize it away here.
        return self

    @property
    def requires_human_review(self) -> bool:
        return self.status == ProcessingStatus.NEEDS_HUMAN_REVIEW

    def to_public_json(self) -> dict:
        """Serialize with the account number masked - used for API responses and the
        Streamlit UI, as opposed to internal storage where the full number is kept
        (encrypted at rest, per README security section) for reconciliation purposes."""
        data = self.model_dump(mode="json")
        data["account"]["account_number"] = self.account.masked_account_number
        return data
