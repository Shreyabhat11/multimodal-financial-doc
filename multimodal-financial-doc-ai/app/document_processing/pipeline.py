"""
DocumentPreprocessor — the single entry point the rest of the app calls.

This is intentionally a thin orchestrator: it doesn't know how to render PDFs or how
to enhance images itself, it just calls limits/pdf_loader/preprocessor in the right
order and assembles the result. This is what gets invoked from the LangGraph
`load_document` + `preprocess_pages` nodes (Phase 7) — those nodes are themselves thin
wrappers around this class, since the graph layer's job is state/control-flow, not
PDF-rendering logic.
"""

from __future__ import annotations

import uuid

from app.core.config import get_settings
from app.core.exceptions import UnsupportedFileTypeError
from app.document_processing.limits import validate_file_size, validate_page_count
from app.document_processing.models import PageImage, PreprocessedDocument
from app.document_processing.pdf_loader import get_pdf_page_count, render_pdf_pages
from app.document_processing.preprocessor import preprocess_page_image

SUPPORTED_EXTENSIONS = (".pdf",)


class DocumentPreprocessor:
    """Stateless service object. One instance can safely process many documents;
    all per-document data lives in the returned PreprocessedDocument, not on self."""

    def __init__(
        self,
        *,
        max_file_size_mb: int | None = None,
        max_pages: int | None = None,
        page_render_dpi: int | None = None,
    ) -> None:
        settings = get_settings()
        self.max_file_size_mb = max_file_size_mb if max_file_size_mb is not None else settings.max_file_size_mb
        self.max_pages = max_pages if max_pages is not None else settings.max_pages
        self.page_render_dpi = page_render_dpi if page_render_dpi is not None else settings.page_render_dpi

    def process(
        self,
        file_bytes: bytes,
        *,
        original_filename: str,
        document_id: str | None = None,
    ) -> PreprocessedDocument:
        """Validate, render, and preprocess an uploaded PDF end to end.

        Order of operations matters: cheap checks first. We validate file size and
        extension before touching PyMuPDF at all, and validate page count (a cheap
        metadata read) before rendering a single pixel — so a 200-page file gets
        rejected in milliseconds rather than after minutes of full-resolution
        rendering.
        """
        self._validate_extension(original_filename)
        validate_file_size(file_bytes, max_size_mb=self.max_file_size_mb)

        page_count = get_pdf_page_count(file_bytes)
        validate_page_count(page_count, max_pages=self.max_pages)

        raw_pages = render_pdf_pages(file_bytes, dpi=self.page_render_dpi)

        pages: list[PageImage] = []
        for page_number, image, source_dpi, rotation_applied, was_rotated in raw_pages:
            processed_image = preprocess_page_image(image)
            pages.append(
                PageImage(
                    page_number=page_number,
                    image=processed_image,
                    source_dpi=source_dpi,
                    original_width=image.width,
                    original_height=image.height,
                    rotation_applied_degrees=rotation_applied,
                    was_rotation_detected=was_rotated,
                )
            )

        return PreprocessedDocument(
            document_id=document_id or str(uuid.uuid4()),
            original_filename=original_filename,
            page_count=len(pages),
            file_size_bytes=len(file_bytes),
            pages=pages,
        )

    @staticmethod
    def _validate_extension(filename: str) -> None:
        lower = filename.lower()
        if not any(lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            raise UnsupportedFileTypeError(
                f"Unsupported file type for '{filename}'. Supported: {SUPPORTED_EXTENSIONS}",
                details={"filename": filename, "supported_extensions": list(SUPPORTED_EXTENSIONS)},
            )
