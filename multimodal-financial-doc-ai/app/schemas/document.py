"""Document and Account schemas.

These represent the *identity and metadata* of a financial document — as opposed to
its line items (Transaction, in transaction.py) or the outcome of checking it
(ValidationResult/Anomaly, in validation.py/anomaly.py). Splitting along this line
(what the document IS vs. what's IN it vs. what we THINK of it) is what the brief
asked for as "separate schemas for Document / Account / Transaction / ..." and it also
maps directly onto separate database tables in Phase 12.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.enums import DocumentType
from app.schemas.parsing import mask_account_number, parse_flexible_date


class StatementPeriod(BaseModel):
    """The date range a statement covers."""

    model_config = ConfigDict(str_strip_whitespace=True)

    start_date: date
    end_date: date

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _coerce_date(cls, v):
        return parse_flexible_date(v) if not isinstance(v, date) else v

    @model_validator(mode="after")
    def _end_after_start(self) -> "StatementPeriod":
        if self.end_date < self.start_date:
            raise ValueError(
                f"statement_period.end_date ({self.end_date}) is before "
                f"start_date ({self.start_date})"
            )
        return self

    @property
    def num_days(self) -> int:
        return (self.end_date - self.start_date).days


class Account(BaseModel):
    """Account/holder metadata extracted from the document header/footer."""

    model_config = ConfigDict(str_strip_whitespace=True)

    account_holder: str = Field(..., min_length=1, max_length=256)
    account_number: str = Field(..., min_length=1, max_length=64)
    bank_name: str = Field(default="", max_length=256)
    branch: str | None = Field(default=None, max_length=256)

    @property
    def masked_account_number(self) -> str:
        return mask_account_number(self.account_number)

    def __repr__(self) -> str:  # avoid accidentally leaking the raw number in logs/repr
        return (
            f"Account(account_holder={self.account_holder!r}, "
            f"account_number={self.masked_account_number!r}, bank_name={self.bank_name!r})"
        )


CurrencyCode = Annotated[str, Field(min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")]


class DocumentMetadata(BaseModel):
    """Structural metadata about the source file, independent of its financial content."""

    model_config = ConfigDict(str_strip_whitespace=True)

    document_id: str
    original_filename: str
    document_type: DocumentType = DocumentType.UNKNOWN
    page_count: int = Field(..., ge=1)
    file_size_bytes: int = Field(..., ge=1)
    uploaded_at: str  # ISO-8601 timestamp string; kept as str at the schema boundary,
    # parsed to datetime only where needed (DB models use a real DateTime column).

    @field_validator("document_type", mode="before")
    @classmethod
    def _coerce_document_type(cls, v):
        if isinstance(v, DocumentType):
            return v
        try:
            return DocumentType(str(v).lower())
        except ValueError:
            return DocumentType.UNKNOWN


class FinancialTotals(BaseModel):
    """Aggregate totals reported by (or computed from) the document.

    Kept separate from Account/Document because it's produced twice in the pipeline:
    once as *extracted* (what the VLM read off the document's own "totals" line, if
    present) and once as *computed* (what deterministic validation calculates from the
    transaction list). Both instances use this same schema — see
    app.validation.financial for the computed version.
    """

    total_debits: Decimal = Field(default=Decimal("0"))
    total_credits: Decimal = Field(default=Decimal("0"))

    @field_validator("total_debits", "total_credits", mode="before")
    @classmethod
    def _non_negative(cls, v):
        from app.schemas.parsing import parse_amount

        amount = parse_amount(v) if not isinstance(v, Decimal) else v
        if amount < 0:
            raise ValueError(f"Totals must be non-negative, got {amount}")
        return amount

    @property
    def net_change(self) -> Decimal:
        return self.total_credits - self.total_debits
