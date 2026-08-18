"""Confidence scoring schemas.

Field-level confidence comes from two sources combined: (1) the VLM's own reported
confidence for that field, when the prompt asks it to self-report one, and (2) signals
from deterministic checks (e.g. a field that passed schema validation and financial
reconciliation gets a boost; a field OCR had to rescue after VLM failure gets a
penalty). The aggregation logic that combines per-field scores into one document-level
score lives in app/validation/confidence.py (Phase 11) — this module only defines the
data shapes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FieldConfidence(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    field_name: str
    value: Any
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: str = Field(
        default="vlm", description="Where this value came from: 'vlm', 'ocr_fallback', or 'computed'."
    )


class DocumentConfidence(BaseModel):
    """Document-level confidence, aggregated from field-level scores via configured weights."""

    field_scores: dict[str, FieldConfidence] = Field(default_factory=dict)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    below_threshold: bool = Field(
        default=False, description="True if overall_confidence < settings.confidence_threshold."
    )
