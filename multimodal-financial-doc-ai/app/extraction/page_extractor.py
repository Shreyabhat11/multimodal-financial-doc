"""
PageExtractor — orchestrates the "vision-first -> confidence evaluation -> OCR
fallback if necessary" flow (brief, Section 7) for a single page.

This is the class the LangGraph `extract_page_information` node (Phase 7) calls once
per page. It doesn't know about PDFs, LangGraph state, or multi-page merging — its
entire job is "given one page image, produce the best structured JSON we can for it,
and tell the caller how confident we are and whether we had to fall back." That
narrow scope is what makes it independently unit-testable (see the tests exercised
below) without any of the graph or document-processing machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.document_processing.models import PageImage
from app.extraction.base_vision_model import BaseVisionModel, RawVLMResponse
from app.extraction.confidence_estimator import estimate_page_confidence
from app.extraction.ocr_engine import run_tesseract_ocr
from app.extraction.prompts import (
    SYSTEM_INSTRUCTION,
    build_ocr_structuring_prompt,
    build_page_extraction_prompt,
)
from app.schemas.enums import DocumentType


@dataclass
class PageExtractionOutcome:
    page_number: int
    parsed_json: dict | None
    confidence: float
    source: str  # "vlm" | "ocr_fallback" | "failed"
    used_fallback: bool
    vlm_response: RawVLMResponse
    ocr_response: RawVLMResponse | None = None

    @property
    def succeeded(self) -> bool:
        return self.parsed_json is not None


class PageExtractor:
    def __init__(
        self,
        vision_model: BaseVisionModel,
        *,
        ocr_fallback_model: BaseVisionModel | None = None,
        ocr_enabled: bool | None = None,
        ocr_trigger_confidence_below: float | None = None,
    ) -> None:
        settings = get_settings()
        self.vision_model = vision_model
        self.ocr_fallback_model = ocr_fallback_model
        self.ocr_enabled = settings.ocr_enabled if ocr_enabled is None else ocr_enabled
        self.ocr_trigger_confidence_below = (
            settings.ocr_trigger_confidence_below
            if ocr_trigger_confidence_below is None
            else ocr_trigger_confidence_below
        )

    def extract_page(
        self,
        page_image: PageImage,
        *,
        total_pages: int,
        expected_document_type: DocumentType | None = None,
    ) -> PageExtractionOutcome:
        prompt = build_page_extraction_prompt(
            page_number=page_image.page_number,
            total_pages=total_pages,
            expected_document_type=expected_document_type,
        )
        vlm_response = self.vision_model.extract(
            page_image.image,
            prompt,
            page_number=page_image.page_number,
            system_instruction=SYSTEM_INSTRUCTION,
        )
        confidence = estimate_page_confidence(vlm_response)

        needs_fallback = (not vlm_response.succeeded) or (confidence < self.ocr_trigger_confidence_below)

        if not (needs_fallback and self.ocr_enabled and self.ocr_fallback_model is not None):
            return PageExtractionOutcome(
                page_number=page_image.page_number,
                parsed_json=vlm_response.parsed_json,
                confidence=confidence,
                source="vlm" if vlm_response.succeeded else "failed",
                used_fallback=False,
                vlm_response=vlm_response,
            )

        ocr_response = self._run_ocr_fallback(
            page_image,
            total_pages=total_pages,
            expected_document_type=expected_document_type,
        )
        ocr_confidence = estimate_page_confidence(ocr_response)

        # Prefer whichever of the two actually succeeded; if both succeeded, prefer
        # the one with higher estimated confidence rather than unconditionally
        # trusting the fallback — OCR fallback is a recovery path, not automatically
        # "more correct."
        if ocr_response.succeeded and (not vlm_response.succeeded or ocr_confidence >= confidence):
            return PageExtractionOutcome(
                page_number=page_image.page_number,
                parsed_json=ocr_response.parsed_json,
                confidence=ocr_confidence,
                source="ocr_fallback",
                used_fallback=True,
                vlm_response=vlm_response,
                ocr_response=ocr_response,
            )

        return PageExtractionOutcome(
            page_number=page_image.page_number,
            parsed_json=vlm_response.parsed_json,
            confidence=confidence,
            source="vlm" if vlm_response.succeeded else "failed",
            used_fallback=True,
            vlm_response=vlm_response,
            ocr_response=ocr_response,
        )

    def _run_ocr_fallback(
        self,
        page_image: PageImage,
        *,
        total_pages: int,
        expected_document_type: DocumentType | None,
    ) -> RawVLMResponse:
        try:
            ocr_text = run_tesseract_ocr(page_image.image)
        except Exception as exc:  # OCR engine itself failing is a valid, expected outcome
            return RawVLMResponse(
                page_number=page_image.page_number,
                model_name="ocr-fallback",
                raw_text="",
                parsed_json=None,
                latency_seconds=0.0,
                succeeded=False,
                error_message=f"OCR engine failed: {exc}",
            )

        if not ocr_text.strip():
            return RawVLMResponse(
                page_number=page_image.page_number,
                model_name="ocr-fallback",
                raw_text="",
                parsed_json=None,
                latency_seconds=0.0,
                succeeded=False,
                error_message="OCR produced no text (likely a blank or unreadable page).",
            )

        ocr_prompt = build_ocr_structuring_prompt(
            ocr_text=ocr_text,
            page_number=page_image.page_number,
            total_pages=total_pages,
            expected_document_type=expected_document_type,
        )
        return self.ocr_fallback_model.extract(
            page_image.image,
            ocr_prompt,
            page_number=page_image.page_number,
            system_instruction=SYSTEM_INSTRUCTION,
        )
