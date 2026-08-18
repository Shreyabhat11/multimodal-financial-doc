"""
DocumentService — the use-case layer referenced in ARCHITECTURE.md Phase 1 ("Why
services/ exists as its own layer"). FastAPI routes call this; this calls the graph,
the repositories, and file storage. Routes never touch LangGraph, SQLAlchemy, or the
VLM factory directly.

Dependency injection: vision_model/ocr_fallback_model are constructed from the
model_factory by default but can be injected — this is what makes the service
(and therefore the API routes built on top of it) testable with a fake VLM, exactly
like every other layer in this project.
"""

from __future__ import annotations

import uuid

from app.core.config import Settings, get_settings
from app.core.exceptions import DocumentProcessingError
from app.database.encryption import FieldEncryptor
from app.database.repositories import DocumentRepository, ProcessingRunRepository
from app.database.session import session_scope
from app.document_processing.pipeline import DocumentPreprocessor
from app.extraction.base_vision_model import BaseVisionModel
from app.extraction.model_factory import get_vision_model
from app.extraction.ocr_fallback import OCRFallbackModel
from app.graph.graph_builder import build_document_graph
from app.schemas.document import Account, FinancialTotals, StatementPeriod
from app.schemas.enums import ProcessingStatus
from app.schemas.transaction import Transaction
from app.services.file_storage import load_uploaded_file, save_uploaded_file

_STATUS_TO_ORM = {
    ProcessingStatus.UPLOADED: "uploaded",
    ProcessingStatus.PROCESSING: "processing",
    ProcessingStatus.VALIDATING: "validating",
    ProcessingStatus.NEEDS_HUMAN_REVIEW: "needs_human_review",
    ProcessingStatus.COMPLETED: "completed",
    ProcessingStatus.FAILED: "failed",
}


class DocumentService:
    def __init__(
        self,
        *,
        vision_model: BaseVisionModel | None = None,
        ocr_fallback_model: BaseVisionModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vision_model = vision_model or get_vision_model(self.settings)
        self.ocr_fallback_model = (
            ocr_fallback_model
            if ocr_fallback_model is not None
            else (OCRFallbackModel() if self.settings.ocr_enabled else None)
        )

    def submit_document(self, *, file_bytes: bytes, original_filename: str) -> str:
        """Validate + persist the upload and create the initial DB row. Returns the
        new document_id. Does NOT run the pipeline — that's `process_document`,
        called separately (typically as a FastAPI background task) so the upload
        request returns quickly with a processing ID rather than blocking on VLM
        calls (brief, Section 14: "Return a processing ID").
        """
        # Cheap, fast validation before we persist anything or touch the DB — the
        # full validation (page count, corruption) happens inside the pipeline
        # itself during preprocess_pages, but rejecting an obviously-wrong upload
        # (wrong extension, empty file) immediately gives the caller a faster,
        # clearer error than waiting for a background task to fail.
        if not original_filename.lower().endswith(".pdf"):
            raise DocumentProcessingError(
                f"Unsupported file type for '{original_filename}'. Only .pdf is supported."
            )
        if not file_bytes:
            raise DocumentProcessingError("Uploaded file is empty.")

        document_id = str(uuid.uuid4())
        save_uploaded_file(document_id, file_bytes)

        with session_scope() as db:
            repo = DocumentRepository(db, FieldEncryptor(self.settings.secret_key))
            repo.create(
                document_id=document_id,
                original_filename=original_filename,
                document_type="unknown",
                page_count=0,  # unknown until preprocess_pages runs
                file_size_bytes=len(file_bytes),
            )

        return document_id

    def process_document(self, document_id: str) -> None:
        """Run the full pipeline for a document that was already submitted (or is
        being reprocessed) and persist the result. This is the function FastAPI's
        BackgroundTasks calls after the upload response has already been sent, and
        the function the reprocess endpoint calls synchronously-in-background too —
        one code path for both, since "reprocess" is just "process_document again"
        with a fresh ProcessingRun row.
        """
        file_bytes = load_uploaded_file(document_id)

        with session_scope() as db:
            encryptor = FieldEncryptor(self.settings.secret_key)
            doc_repo = DocumentRepository(db, encryptor)
            run_repo = ProcessingRunRepository(db)

            doc_repo.update_status(document_id, "processing")
            run = run_repo.start_run(document_id, model_provider=self.settings.model_provider)
            run_id = run.id

        graph = build_document_graph(
            self.vision_model, ocr_fallback_model=self.ocr_fallback_model, settings=self.settings
        )

        final_state = graph.invoke(
            {
                "document_id": document_id,
                "file_bytes": file_bytes,
                # We validated the extension at submit_document() time and store the
                # file at data/raw/{document_id}.pdf regardless of the original
                # filename, so a synthetic .pdf-suffixed name here satisfies
                # DocumentPreprocessor's own extension check without needing to
                # thread the original filename through storage as well.
                "original_filename": f"{document_id}.pdf",
            }
        )

        status: ProcessingStatus = final_state.get("status", ProcessingStatus.FAILED)
        final_document = final_state.get("final_document")
        validation_results = final_state.get("validation_results", [])
        error_message = final_state.get("error_message")

        with session_scope() as db:
            encryptor = FieldEncryptor(self.settings.secret_key)
            doc_repo = DocumentRepository(db, encryptor)
            run_repo = ProcessingRunRepository(db)

            if final_document is not None:
                account = Account(**final_document["account"])
                statement_period = StatementPeriod(**final_document["statement_period"])
                transactions = [Transaction(**t) for t in final_document["transactions"]]
                confidence_data = final_document.get("confidence") or {}

                doc_repo.save_pipeline_result(
                    document_id=document_id,
                    status=_STATUS_TO_ORM[status],
                    account=account,
                    statement_period=statement_period,
                    opening_balance=final_document["opening_balance"],
                    closing_balance=final_document["closing_balance"],
                    currency=final_document["currency"],
                    transactions=transactions,
                    validation_results=validation_results,
                    overall_confidence=confidence_data.get("overall_confidence"),
                    error_message=error_message,
                    page_count=final_document.get("metadata", {}).get("page_count"),
                    document_type=final_document.get("document_type"),
                )
            else:
                doc_repo.update_status(document_id, _STATUS_TO_ORM[status], error_message=error_message)

            run_repo.complete_run(
                run_id,
                status=_STATUS_TO_ORM[status],
                error_message=error_message,
                retry_count=final_state.get("page_extraction_retry_count", 0),
            )
