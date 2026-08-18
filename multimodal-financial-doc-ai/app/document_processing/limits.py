"""
Enforced limits on uploaded documents.

Why these limits exist (asked for explicitly in the brief):

- ``MAX_FILE_SIZE_MB``: without a cap, a single malicious or accidental upload (a
  500MB scanned PDF) can exhaust API worker memory or disk, since the whole file is
  read into memory before page extraction. A 25MB default comfortably covers a
  100+ page statement scanned at reasonable DPI while bounding worst-case memory use
  per request.
- ``MAX_PAGES``: each page is a separate VLM inference call. Without a cap, a single
  upload could trigger hundreds of expensive (and, for local GPU inference, slow)
  model calls, turning one request into a de facto denial-of-service against the
  inference backend and blowing through per-document cost/latency budgets. 50 pages
  covers effectively all real bank/credit-card/loan statements; anything larger is
  far more likely to be a corrupted file or a wrong-file upload than a legitimate
  single statement.
- Both limits fail fast, before any expensive processing (page rendering, VLM calls)
  happens — checked immediately on the raw file, not discovered halfway through a
  50-page rendering loop.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.exceptions import FileTooLargeError, TooManyPagesError


def validate_file_size(file_bytes: bytes, *, max_size_mb: int | None = None) -> None:
    """Raise FileTooLargeError if the raw upload exceeds the configured limit."""
    settings = get_settings()
    limit_mb = max_size_mb if max_size_mb is not None else settings.max_file_size_mb
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > limit_mb:
        raise FileTooLargeError(
            f"Uploaded file is {size_mb:.2f}MB, which exceeds the {limit_mb}MB limit.",
            details={"file_size_mb": round(size_mb, 2), "limit_mb": limit_mb},
        )


def validate_page_count(page_count: int, *, max_pages: int | None = None) -> None:
    """Raise TooManyPagesError if the document has more pages than allowed."""
    settings = get_settings()
    limit = max_pages if max_pages is not None else settings.max_pages
    if page_count > limit:
        raise TooManyPagesError(
            f"Document has {page_count} pages, which exceeds the {limit}-page limit.",
            details={"page_count": page_count, "limit": limit},
        )
    if page_count < 1:
        raise TooManyPagesError(
            "Document has zero readable pages.",
            details={"page_count": page_count},
        )
