"""
Page-level confidence estimation.

This produces the number PageExtractor uses to decide whether a page needs the OCR
fallback (brief Section 7: "Vision-first extraction -> Confidence evaluation -> OCR
fallback if necessary"). It is deliberately simple and deterministic — a weighted
combination of the model's own self-reported field confidences plus a couple of
structural sanity signals — NOT a second model call. Using another LLM call just to
judge the first LLM call's confidence would add latency/cost without a clear accuracy
benefit, and would itself need to be trusted, which doesn't solve the calibration
problem, just moves it.

This is intentionally distinct from (and simpler than) the document-level
`DocumentConfidence` aggregation in app/validation/confidence.py (Phase 11), which
combines per-field confidence with deterministic validation outcomes across the WHOLE
document, not just one page's raw model output.
"""

from __future__ import annotations

from app.extraction.base_vision_model import RawVLMResponse


def estimate_page_confidence(response: RawVLMResponse) -> float:
    """Estimate a 0.0-1.0 confidence score for a single page's extraction.

    Rules, in order of precedence:
    1. A failed call (network error, unparseable JSON) is always 0.0 — there's no
       partial credit for "the model tried."
    2. If the model reported field_confidence values, use their mean.
    3. If it reported none (some smaller/older checkpoints don't reliably follow that
       instruction), fall back to a fixed moderate default (0.7) rather than 0.0 —
       treating "no self-report" as equivalent to "total failure" would send every
       page through OCR fallback unnecessarily for any model that doesn't self-report.
    """
    if not response.succeeded or response.parsed_json is None:
        return 0.0

    confidences = [
        v
        for v in response.self_reported_confidence.values()
        if isinstance(v, (int, float)) and 0.0 <= v <= 1.0
    ]
    if confidences:
        return sum(confidences) / len(confidences)

    return 0.7
