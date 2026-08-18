"""
PDF -> page images.

Why PyMuPDF (fitz) rather than pdf2image/poppler: PyMuPDF renders pages directly to
raster images without shelling out to an external `pdftoppm` binary, which keeps the
Docker image smaller (no poppler-utils system dependency) and rendering faster for
multi-page documents. It also exposes each page's ``rotation`` metadata directly,
which we need for automatic rotation correction (Section 5 of the brief).

Trade-off worth naming in an interview: PyMuPDF's license (AGPL, with a commercial
option) is more restrictive than poppler's (permissive). For a portfolio project this
is a non-issue; for a real commercial product you'd weigh AGPL obligations against
buying a commercial PyMuPDF license or switching to poppler-based rendering.
"""

from __future__ import annotations

import io

import pymupdf  # PyMuPDF — the `fitz` alias is the legacy import name, deprecated upstream
from PIL import Image

from app.core.exceptions import CorruptedDocumentError


def get_pdf_page_count(file_bytes: bytes) -> int:
    """Open a PDF just far enough to read its page count, without rendering anything.
    Used to validate page-count limits BEFORE paying the cost of rendering every page.
    """
    try:
        with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
            return doc.page_count
    except Exception as exc:  # PyMuPDF raises its own exception types for bad files
        raise CorruptedDocumentError(
            "Could not open PDF to read page count — the file may be corrupted, "
            "password-protected, or not a valid PDF.",
            details={"underlying_error": str(exc)},
        ) from exc


def _detect_and_normalize_rotation(page: "pymupdf.Page") -> int:
    """Return the page's declared rotation (0/90/180/270) as recorded in the PDF's
    own page object. Many scanners/statement generators write this metadata directly
    rather than physically rotating pixel content, so honoring it is the cheap, exact
    way to fix rotated pages — no image-analysis heuristics needed for the common case.
    """
    rotation = page.rotation % 360
    if rotation not in (0, 90, 180, 270):
        return 0
    return rotation


def render_pdf_pages(
    file_bytes: bytes,
    *,
    dpi: int = 200,
) -> list[tuple[int, Image.Image, int, int, int]]:
    """
    Render every page of a PDF to a PIL Image.

    Returns a list of tuples:
        (page_number [1-indexed], image, source_dpi, rotation_applied_degrees, was_rotated)

    Rotation handling: pymupdf.Page.get_pixmap() already honors the page's ``/Rotate``
    entry when rendering (i.e. the output pixmap is already upright) — we still record
    the rotation that was applied so it's visible in PageImage metadata for debugging
    and for the human-review UI.
    """
    zoom = dpi / 72.0  # PDF native resolution is 72 DPI; fitz scales via a zoom matrix
    matrix = pymupdf.Matrix(zoom, zoom)

    results: list[tuple[int, Image.Image, int, int, int]] = []
    try:
        with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
            for index, page in enumerate(doc):
                rotation = _detect_and_normalize_rotation(page)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                results.append((index + 1, image, dpi, rotation, rotation != 0))
    except CorruptedDocumentError:
        raise
    except Exception as exc:
        raise CorruptedDocumentError(
            "Failed while rendering PDF pages — the file may be corrupted or "
            "password-protected.",
            details={"underlying_error": str(exc)},
        ) from exc

    if not results:
        raise CorruptedDocumentError("PDF contains no renderable pages.")

    return results


def bytes_to_pil_image(image_bytes: bytes) -> Image.Image:
    """Helper used by tests/fixtures to load a standalone image file as a PIL Image."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
