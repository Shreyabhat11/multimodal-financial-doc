"""
FastAPI dependencies.

`get_document_service` returns a process-wide singleton (constructed lazily, once) —
constructing a DocumentService means constructing a vision model, which for the
local-inference backends is expensive (loading GPU weights). Tests override this
dependency via `app.dependency_overrides[get_document_service]` to inject a
DocumentService built with a fake vision model, exactly the same override pattern
FastAPI recommends for any expensive/external dependency.
"""

from __future__ import annotations

from app.services.document_service import DocumentService

_document_service: DocumentService | None = None


def get_document_service() -> DocumentService:
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service
