"""
API-facing Pydantic models.

Deliberately thin — these describe HTTP request/response shapes, not domain data
(that's app/schemas/). `DocumentResultResponse` is intentionally NOT a strict
re-declaration of every FinalExtractionResult field; it wraps the dict produced by
`document_to_public_dict` (Phase 11) with `model_config = ConfigDict(extra="allow")`
rather than duplicating that whole nested shape a second time here — duplicating it
would create exactly the kind of two-schemas-that-must-be-kept-in-sync problem this
project's schema/ORM split was designed to avoid elsewhere.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UploadResponse(BaseModel):
    document_id: str
    status: str
    message: str = "Document received and queued for processing."


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    original_filename: str
    page_count: int
    error_message: str | None = None
    uploaded_at: str | None = None


class DocumentSummaryResponse(BaseModel):
    document_id: str
    status: str
    document_type: str
    original_filename: str
    uploaded_at: str | None = None


class DocumentResultResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_id: str
    status: str


class ReprocessResponse(BaseModel):
    document_id: str
    status: str
    message: str = "Document queued for reprocessing."


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    timestamp: datetime
