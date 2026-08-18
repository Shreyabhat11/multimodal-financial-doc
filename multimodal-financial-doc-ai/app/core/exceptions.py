"""
Application-wide exception hierarchy.

Design decision
----------------
Every exception carries a machine-readable ``error_code`` in addition to the human
message. FastAPI's exception handlers (Phase 13) map ``error_code`` to a stable HTTP
status + JSON body, so callers (Streamlit, external API consumers) can branch on
``error_code`` instead of parsing message strings. Keeping exceptions specific (rather
than raising bare ``ValueError``/``Exception`` everywhere) is also what makes the
LangGraph error-routing in Phase 7 possible — nodes catch specific exception types and
route to different edges (retry vs. fail vs. human review) based on which one fired.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all application-raised errors."""

    error_code: str = "app_error"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"error_code": self.error_code, "message": self.message, "details": self.details}


# ---------------------------------------------------------------------------
# Document ingestion / preprocessing errors
# ---------------------------------------------------------------------------


class DocumentProcessingError(AppError):
    error_code = "document_processing_error"
    http_status = 422


class UnsupportedFileTypeError(DocumentProcessingError):
    error_code = "unsupported_file_type"
    http_status = 415


class FileTooLargeError(DocumentProcessingError):
    error_code = "file_too_large"
    http_status = 413


class TooManyPagesError(DocumentProcessingError):
    error_code = "too_many_pages"
    http_status = 422


class CorruptedDocumentError(DocumentProcessingError):
    error_code = "corrupted_document"
    http_status = 422


# ---------------------------------------------------------------------------
# VLM / extraction errors
# ---------------------------------------------------------------------------


class VisionModelError(AppError):
    error_code = "vision_model_error"
    http_status = 502


class VisionModelTimeoutError(VisionModelError):
    error_code = "vision_model_timeout"
    http_status = 504


class VisionModelResponseParsingError(VisionModelError):
    """Raised when the VLM returns text that cannot be parsed into structured JSON."""

    error_code = "vision_model_response_parsing_error"
    http_status = 502


# ---------------------------------------------------------------------------
# Schema / validation errors
# ---------------------------------------------------------------------------


class SchemaValidationError(AppError):
    error_code = "schema_validation_error"
    http_status = 422


class FinancialValidationError(AppError):
    """Raised only for programming/data errors in the validation module itself —
    NOT raised for a document that simply fails reconciliation. A document failing
    reconciliation is an expected outcome represented in ValidationResult, not an
    exception."""

    error_code = "financial_validation_error"
    http_status = 500


# ---------------------------------------------------------------------------
# Graph / agent errors
# ---------------------------------------------------------------------------


class GraphExecutionError(AppError):
    error_code = "graph_execution_error"
    http_status = 500


class CrewValidationError(AppError):
    error_code = "crew_validation_error"
    http_status = 502


# ---------------------------------------------------------------------------
# Resource / persistence errors
# ---------------------------------------------------------------------------


class DocumentNotFoundError(AppError):
    error_code = "document_not_found"
    http_status = 404


class DatabaseOperationError(AppError):
    error_code = "database_operation_error"
    http_status = 500
