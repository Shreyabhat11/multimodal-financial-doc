"""Transaction schema.

This is the highest-volume, highest-stakes schema in the system — a 30-page bank
statement might carry 800+ of these, and every downstream check (reconciliation,
duplicate detection, anomaly detection) operates on this exact shape. Money fields are
``Decimal``, never ``float`` — floats introduce binary rounding error
(``0.1 + 0.2 != 0.3``) that is unacceptable when the whole point of the validation
layer is catching penny-level reconciliation mismatches.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.parsing import parse_amount, parse_flexible_date


class Transaction(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    date: date
    description: str = Field(..., min_length=1, max_length=512)
    reference: str | None = Field(default=None, max_length=128)
    debit: Decimal = Field(default=Decimal("0"))
    credit: Decimal = Field(default=Decimal("0"))
    balance: Decimal | None = Field(
        default=None, description="Running balance after this transaction, if reported."
    )
    currency: str = Field(default="USD", min_length=3, max_length=3)

    # Set by normalize_transactions (Phase 6/7) — which page/row this came from, used
    # for traceability and for the human-review UI to point a reviewer at the right page.
    source_page: int | None = Field(default=None, ge=1)

    @field_validator("date", mode="before")
    @classmethod
    def _coerce_date(cls, v):
        return parse_flexible_date(v) if not isinstance(v, date) else v

    @field_validator("debit", "credit", mode="before")
    @classmethod
    def _coerce_amount(cls, v):
        if v is None:
            return Decimal("0")
        amount = parse_amount(v) if not isinstance(v, Decimal) else v
        return abs(amount)  # sign is carried by which field (debit vs credit), not the value

    @field_validator("balance", mode="before")
    @classmethod
    def _coerce_balance(cls, v):
        if v is None:
            return None
        return parse_amount(v) if not isinstance(v, Decimal) else v

    @field_validator("currency", mode="before")
    @classmethod
    def _uppercase_currency(cls, v):
        return str(v).upper() if v else "USD"

    @model_validator(mode="after")
    def _not_both_debit_and_credit(self) -> "Transaction":
        # A single transaction row being simultaneously a nonzero debit AND a nonzero
        # credit is not structurally invalid (some statement formats genuinely do this
        # for fee reversals), but it's rare enough to be worth flagging rather than
        # silently accepting — anomaly_detection (Phase 11) checks for this pattern via
        # `is_dual_sided`, so we expose it as a property rather than rejecting here.
        return self

    @property
    def is_dual_sided(self) -> bool:
        return self.debit > 0 and self.credit > 0

    @property
    def net_amount(self) -> Decimal:
        """Positive for credits, negative for debits — convenient for summing a ledger."""
        return self.credit - self.debit

    @property
    def dedup_key(self) -> tuple:
        """Key used by duplicate-transaction detection (date + description + amount + reference)."""
        return (self.date, self.description.strip().lower(), self.debit, self.credit, self.reference)
