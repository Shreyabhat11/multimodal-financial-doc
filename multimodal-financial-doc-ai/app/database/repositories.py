"""
Repositories: the only place in the codebase that constructs SQLAlchemy queries.

This is the Repository pattern, applied so that (a) `app/services/` (Phase 13, the
use-case layer) never imports SQLAlchemy directly — it calls
`DocumentRepository.get_by_id(...)`, not `db.query(DocumentORM).filter(...)` — and
(b) if the storage backend ever changed, only this file would need to change, not
every caller. For a project this size, a full "repository per aggregate root with
abstract interfaces" pattern would be over-engineering (see ARCHITECTURE.md's note on
deliberately NOT doing full CQRS) — these are concrete, pragmatic repository classes,
not an abstraction layer over an abstraction layer.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import DocumentNotFoundError
from app.database.encryption import FieldEncryptor
from app.database.mappers import (
    account_to_orm,
    anomaly_to_orm,
    transaction_to_orm,
    validation_result_to_orm,
)
from app.models.orm import AccountORM, DocumentORM, PageORM, ProcessingRunORM
from app.schemas.document import Account, StatementPeriod
from app.schemas.transaction import Transaction
from app.schemas.validation import ValidationResult


class DocumentRepository:
    def __init__(self, db: Session, encryptor: FieldEncryptor) -> None:
        self.db = db
        self.encryptor = encryptor

    def create(
        self,
        *,
        document_id: str,
        original_filename: str,
        document_type: str,
        page_count: int,
        file_size_bytes: int,
    ) -> DocumentORM:
        doc = DocumentORM(
            id=document_id,
            original_filename=original_filename,
            document_type=document_type,
            status="uploaded",
            page_count=page_count,
            file_size_bytes=file_size_bytes,
        )
        self.db.add(doc)
        self.db.flush()
        return doc

    def get_by_id(self, document_id: str) -> DocumentORM:
        doc = self.db.get(DocumentORM, document_id)
        if doc is None:
            raise DocumentNotFoundError(f"No document found with id '{document_id}'.")
        return doc

    def try_get_by_id(self, document_id: str) -> DocumentORM | None:
        return self.db.get(DocumentORM, document_id)

    def list_recent(self, *, limit: int = 50, offset: int = 0) -> list[DocumentORM]:
        stmt = select(DocumentORM).order_by(DocumentORM.uploaded_at.desc()).limit(limit).offset(offset)
        return list(self.db.execute(stmt).scalars().all())

    def update_status(self, document_id: str, status: str, *, error_message: str | None = None) -> None:
        doc = self.get_by_id(document_id)
        doc.status = status
        if error_message is not None:
            doc.error_message = error_message
        self.db.flush()

    def save_pipeline_result(
        self,
        *,
        document_id: str,
        status: str,
        account: Account | None,
        statement_period: StatementPeriod | None,
        opening_balance,
        closing_balance,
        currency: str,
        transactions: list[Transaction],
        validation_results: list[ValidationResult],
        overall_confidence: float | None,
        error_message: str | None = None,
        page_count: int | None = None,
        document_type: str | None = None,
    ) -> DocumentORM:
        """Persist the full result of one pipeline run against an existing document
        row. Replaces (rather than appends to) transactions/validation_results on
        each save — a reprocessing run (Phase 12) supersedes the prior extraction
        entirely, it doesn't merge with it. ProcessingRun rows (separately) preserve
        the full history of every attempt even though this table reflects only the
        latest."""
        doc = self.get_by_id(document_id)
        doc.status = status
        doc.currency = currency
        doc.opening_balance = opening_balance
        doc.closing_balance = closing_balance
        doc.error_message = error_message
        doc.overall_confidence = overall_confidence
        if page_count is not None:
            doc.page_count = page_count
        if document_type is not None:
            doc.document_type = document_type
        if statement_period is not None:
            doc.statement_start_date = statement_period.start_date
            doc.statement_end_date = statement_period.end_date

        # Replace account (one-to-one)
        if account is not None:
            existing_account = self.db.execute(
                select(AccountORM).where(AccountORM.document_id == document_id)
            ).scalar_one_or_none()
            if existing_account is not None:
                self.db.delete(existing_account)
                self.db.flush()
            self.db.add(account_to_orm(account, document_id=document_id, encryptor=self.encryptor))

        # Replace transactions
        for existing_txn in list(doc.transactions):
            self.db.delete(existing_txn)
        for txn in transactions:
            self.db.add(transaction_to_orm(txn, document_id=document_id))

        # Replace validation results
        for existing_result in list(doc.validation_results):
            self.db.delete(existing_result)
        for result in validation_results:
            self.db.add(validation_result_to_orm(result, document_id=document_id))
            for anomaly in result.anomalies:
                self.db.add(anomaly_to_orm(anomaly, document_id=document_id))

        self.db.flush()
        self.db.refresh(doc)
        return doc

    def add_pages(self, document_id: str, page_metadata: list[dict]) -> None:
        for page in page_metadata:
            self.db.add(
                PageORM(
                    document_id=document_id,
                    page_number=page["page_number"],
                    width=page["width"],
                    height=page["height"],
                    source_dpi=page["source_dpi"],
                    rotation_applied_degrees=page.get("rotation_applied_degrees", 0),
                    extraction_source=page.get("source"),
                    extraction_confidence=page.get("confidence"),
                    used_fallback=page.get("used_fallback", False),
                )
            )
        self.db.flush()


class ProcessingRunRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def start_run(self, document_id: str, *, model_provider: str) -> ProcessingRunORM:
        run = ProcessingRunORM(document_id=document_id, status="processing", model_provider=model_provider)
        self.db.add(run)
        self.db.flush()
        return run

    def complete_run(
        self, run_id: str, *, status: str, error_message: str | None = None, retry_count: int = 0
    ) -> None:
        run = self.db.get(ProcessingRunORM, run_id)
        if run is None:
            return
        run.status = status
        run.completed_at = datetime.utcnow()
        run.error_message = error_message
        run.page_extraction_retry_count = retry_count
        self.db.flush()

    def list_for_document(self, document_id: str) -> list[ProcessingRunORM]:
        stmt = select(ProcessingRunORM).where(ProcessingRunORM.document_id == document_id).order_by(
            ProcessingRunORM.started_at
        )
        return list(self.db.execute(stmt).scalars().all())
