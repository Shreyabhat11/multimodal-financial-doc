"""
SQLAlchemy 2.0 ORM models.

Deliberately kept SEPARATE from app/schemas/ (the Pydantic models) rather than
merged into one set of classes. This is a real architectural choice, not
boilerplate duplication: Pydantic schemas describe the shape of data crossing a
boundary (API request/response, VLM extraction output); ORM models describe how
data is physically stored and related in Postgres (foreign keys, indexes, column
types). A field that's `Decimal` in both worlds is still two different concerns —
`Transaction.debit: Decimal` (Pydantic) validates and parses; `TransactionORM.debit:
Mapped[Decimal]` (here) maps to a `NUMERIC(18, 2)` column with specific precision.
Conflating them would mean either the API schema leaking database column
constraints, or the database schema being weakened to match whatever Pydantic finds
convenient — decoupling avoids both.

Table design mirrors the schema split from Phase 2 exactly:
documents / pages / accounts / transactions / validation_results / anomalies /
processing_runs — one ORM class per brief-specified table (Section 13).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid_str() -> str:
    return str(uuid.uuid4())


class DocumentORM(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    document_type: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="uploaded", index=True)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    opening_balance: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True)
    closing_balance: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True)
    statement_start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    statement_end_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    overall_confidence: Mapped[float | None] = mapped_column(nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    pages: Mapped[list["PageORM"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    account: Mapped["AccountORM | None"] = relationship(
        back_populates="document", cascade="all, delete-orphan", uselist=False
    )
    transactions: Mapped[list["TransactionORM"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="TransactionORM.transaction_date"
    )
    validation_results: Mapped[list["ValidationResultORM"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    anomalies: Mapped[list["AnomalyORM"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    processing_runs: Mapped[list["ProcessingRunORM"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="ProcessingRunORM.started_at"
    )


class PageORM(Base):
    __tablename__ = "pages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)

    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    source_dpi: Mapped[int] = mapped_column(Integer, nullable=False)
    rotation_applied_degrees: Mapped[int] = mapped_column(Integer, default=0)

    extraction_source: Mapped[str | None] = mapped_column(String(32), nullable=True)  # "vlm" | "ocr_fallback" | "failed"
    extraction_confidence: Mapped[float | None] = mapped_column(nullable=True)
    used_fallback: Mapped[bool] = mapped_column(Boolean, default=False)

    document: Mapped["DocumentORM"] = relationship(back_populates="pages")


class AccountORM(Base):
    """One-to-one with DocumentORM. `account_number_encrypted` stores the account
    number encrypted at the application layer (see app/database/encryption.py) —
    never plaintext in the database, per the brief's "do not store raw sensitive
    financial information unnecessarily" requirement (Section 13)."""

    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), unique=True, index=True
    )

    account_holder: Mapped[str] = mapped_column(String(256), nullable=False)
    account_number_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    account_number_last4: Mapped[str] = mapped_column(
        String(4), nullable=False, index=True
    )  # searchable without decrypting — "find the statement ending in 6655"
    bank_name: Mapped[str] = mapped_column(String(256), default="")
    branch: Mapped[str | None] = mapped_column(String(256), nullable=True)

    document: Mapped["DocumentORM"] = relationship(back_populates="account")


class TransactionORM(Base):
    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)

    transaction_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    description: Mapped[str] = mapped_column(String(512), nullable=False)
    reference: Mapped[str | None] = mapped_column(String(128), nullable=True)
    debit: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=Decimal("0"))
    credit: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=Decimal("0"))
    balance: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    source_page: Mapped[int | None] = mapped_column(Integer, nullable=True)

    document: Mapped["DocumentORM"] = relationship(back_populates="transactions")


class ValidationResultORM(Base):
    __tablename__ = "validation_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)

    validator_name: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    checks_performed: Mapped[list] = mapped_column(JSON, default=list)
    issues: Mapped[list] = mapped_column(JSON, default=list)  # [{field, severity, message}, ...]
    recommendation: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    document: Mapped["DocumentORM"] = relationship(back_populates="validation_results")


class AnomalyORM(Base):
    __tablename__ = "anomalies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)

    anomaly_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    affected_transaction_indices: Mapped[list] = mapped_column(JSON, default=list)
    field: Mapped[str | None] = mapped_column(String(64), nullable=True)

    document: Mapped["DocumentORM"] = relationship(back_populates="anomalies")


class ProcessingRunORM(Base):
    """One row per attempt at processing a document — a document that gets
    reprocessed (POST /documents/{id}/reprocess, Phase 12) accumulates multiple
    ProcessingRun rows, giving a full audit trail of every attempt rather than only
    the latest one. This is what "processing_runs" (brief, Section 13) is for."""

    __tablename__ = "processing_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)

    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    page_extraction_retry_count: Mapped[int] = mapped_column(Integer, default=0)

    document: Mapped["DocumentORM"] = relationship(back_populates="processing_runs")
