"""
Document endpoints.

Every route here is thin: parse the request, call DocumentService or a repository,
shape the response. No pipeline logic, no SQLAlchemy queries, no VLM calls happen in
this file — that's the point of the services/repositories layers underneath it (see
ARCHITECTURE.md). This is what keeps routes easy to read top-to-bottom as "what does
this endpoint do" without wading through orchestration details.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.api.dependencies import get_document_service
from app.api.schemas import (
    DocumentResultResponse,
    DocumentStatusResponse,
    DocumentSummaryResponse,
    ReprocessResponse,
    UploadResponse,
)
from app.core.config import get_settings
from app.core.exceptions import DocumentNotFoundError
from app.database.encryption import FieldEncryptor
from app.database.mappers import document_to_public_dict
from app.database.repositories import DocumentRepository
from app.database.session import get_db
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
) -> UploadResponse:
    """Accept a PDF upload, persist it, and queue processing in the background.

    Returns 202 Accepted (not 200/201) — deliberately signaling "we've accepted this
    for asynchronous processing, it is not done yet," which is the accurate status
    for a request that returns before the pipeline has run. The caller polls
    GET /documents/{id}/status (or /result once status is a terminal state).
    """
    content = await file.read()
    document_id = service.submit_document(file_bytes=content, original_filename=file.filename or "upload.pdf")
    background_tasks.add_task(service.process_document, document_id)
    return UploadResponse(document_id=document_id, status="uploaded")


@router.get("/{document_id}", response_model=DocumentSummaryResponse)
async def get_document(document_id: str, db: Session = Depends(get_db)) -> DocumentSummaryResponse:
    settings = get_settings()
    repo = DocumentRepository(db, FieldEncryptor(settings.secret_key))
    doc = repo.get_by_id(document_id)  # raises DocumentNotFoundError -> 404, handled globally
    return DocumentSummaryResponse(
        document_id=doc.id,
        status=doc.status,
        document_type=doc.document_type,
        original_filename=doc.original_filename,
        uploaded_at=doc.uploaded_at.isoformat() if doc.uploaded_at else None,
    )


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(document_id: str, db: Session = Depends(get_db)) -> DocumentStatusResponse:
    settings = get_settings()
    repo = DocumentRepository(db, FieldEncryptor(settings.secret_key))
    doc = repo.get_by_id(document_id)
    return DocumentStatusResponse(
        document_id=doc.id,
        status=doc.status,
        original_filename=doc.original_filename,
        page_count=doc.page_count,
        error_message=doc.error_message,
        uploaded_at=doc.uploaded_at.isoformat() if doc.uploaded_at else None,
    )


@router.get("/{document_id}/result", response_model=DocumentResultResponse)
async def get_document_result(document_id: str, db: Session = Depends(get_db)) -> DocumentResultResponse:
    """Return the full structured extraction result. Available once status is
    'completed' or 'needs_human_review' — for any earlier status this still returns
    the document's current state (mostly empty) rather than a 404 or 425, since
    "processing, check back later" is more useful conveyed through the `status`
    field than through an HTTP status code the caller has to special-case.
    """
    settings = get_settings()
    repo = DocumentRepository(db, FieldEncryptor(settings.secret_key))
    doc = repo.get_by_id(document_id)
    public_dict = document_to_public_dict(doc, encryptor=FieldEncryptor(settings.secret_key))
    return DocumentResultResponse(**public_dict)


@router.post("/{document_id}/reprocess", response_model=ReprocessResponse, status_code=202)
async def reprocess_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    service: DocumentService = Depends(get_document_service),
) -> ReprocessResponse:
    """Re-run the full pipeline against the originally uploaded file (loaded from
    storage via the document_id, not re-uploaded). Useful after fixing a config issue
    (e.g. lowering confidence_threshold, or after a model upgrade) without asking the
    user to find and re-upload the original PDF.
    """
    settings = get_settings()
    repo = DocumentRepository(db, FieldEncryptor(settings.secret_key))
    repo.get_by_id(document_id)  # raises DocumentNotFoundError -> 404 if it doesn't exist
    repo.update_status(document_id, "uploaded")
    # Explicit commit here (rather than relying on get_db's post-yield commit) is
    # deliberate: BackgroundTasks can begin executing before a yield-dependency's
    # cleanup has run, and the background task opens its OWN database session
    # immediately. On SQLite specifically (single-writer, even in WAL mode) an
    # uncommitted write from this request can block the background task's write
    # long enough to exceed the busy_timeout. Committing explicitly before queuing
    # the background task guarantees this request's write is durable and its lock
    # released first. Postgres (the production target) allows genuine concurrent
    # writers via MVCC and would not exhibit this specific failure mode, but
    # committing before handing off to a background task is correct practice
    # regardless of backend.
    db.commit()

    background_tasks.add_task(service.process_document, document_id)
    return ReprocessResponse(document_id=document_id, status="uploaded")
