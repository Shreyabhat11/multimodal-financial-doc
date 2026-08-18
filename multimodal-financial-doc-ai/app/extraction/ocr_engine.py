"""
Tesseract OCR wrapper.

This module's ONLY job is "pixels -> raw text". It deliberately does not attempt to
produce structured JSON — that's the job of the text-completion step in
ocr_fallback.py, which takes this raw text and asks a language model to structure it.
Keeping OCR itself dumb (text-in, text-out) is what makes the fallback swappable: you
could replace Tesseract with a cloud OCR API here without touching anything else in
the fallback pipeline.
"""

from __future__ import annotations

from PIL import Image

from app.core.config import get_settings
from app.core.exceptions import VisionModelError


def run_tesseract_ocr(image: Image.Image) -> str:
    """Run Tesseract OCR on a page image and return the extracted raw text.

    Raises VisionModelError (not a bare exception) on failure, so callers can handle
    "OCR itself is unavailable/misconfigured" uniformly with other extraction-layer
    failures rather than needing a separate except clause.
    """
    try:
        import pytesseract
    except ImportError as exc:
        raise VisionModelError(
            "pytesseract is not installed — OCR fallback is unavailable.",
        ) from exc

    settings = get_settings()
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    try:
        # PSM 6 ("assume a single uniform block of text") is a reasonable default for
        # statement pages, which are dense but not multi-column in the way a
        # newspaper layout is. `--oem 3` uses the default (LSTM) OCR engine.
        text = pytesseract.image_to_string(image, config="--psm 6 --oem 3")
    except Exception as exc:
        raise VisionModelError(f"Tesseract OCR failed: {exc}") from exc

    return text.strip()
