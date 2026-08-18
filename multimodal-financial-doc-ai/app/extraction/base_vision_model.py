"""
BaseVisionModel — the Strategy-pattern interface every VLM backend implements.

    BaseVisionModel (ABC)
         |
         ├── QwenVLModel        (local transformers OR HF Inference API — see backend param)
         ├── LLaVAModel         (Phase 5)
         └── OCRFallbackModel   (Phase 6 — same interface, used as a fallback, not a "VLM")

The LangGraph node that calls into this layer (`extract_page_information`, Phase 7)
depends only on this interface, never on a concrete class. Swapping
Qwen2-VL for LLaVA, or local inference for the HF Inference API, is a one-line config
change (`MODEL_PROVIDER` in .env) — no pipeline code changes. That's the entire point
of the abstraction, and it's worth being able to say precisely in an interview: this
is the Strategy pattern, applied so the orchestration layer is decoupled from any
specific model's API surface (different backends have wildly different call
signatures — local `transformers.generate()` vs. an HTTP inference call — and none of
that should leak into the graph).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class RawVLMResponse:
    """The uniform output shape every backend returns, regardless of how it produced it."""

    page_number: int
    model_name: str
    raw_text: str
    parsed_json: dict | None
    latency_seconds: float
    succeeded: bool
    error_message: str | None = None
    self_reported_confidence: dict = field(default_factory=dict)


class BaseVisionModel(ABC):
    """Every concrete backend must implement `_call_model`. The public `extract`
    method wraps it with timing and uniform error capture so subclasses only need to
    worry about the model call itself, not bookkeeping."""

    #: Human-readable name used in logs, DB rows, and evaluation reports.
    name: str = "base_vision_model"

    #: Whether this backend can process multiple pages in a single call. All current
    #: backends process one page at a time (simpler error isolation per page — a
    #: single garbled page doesn't cost you the rest of the document's extraction),
    #: but the flag exists so a future batched backend doesn't require an interface
    #: change.
    supports_batch: bool = False

    @abstractmethod
    def _call_model(self, image: Image.Image, prompt: str, system_instruction: str) -> str:
        """Backend-specific call. Must return the model's raw text output.
        Must raise on failure (timeout, API error, etc.) — do not return an empty
        string / None to signal failure, since `extract()` distinguishes "model
        returned text that failed to parse as JSON" from "the call itself failed",
        and callers (Phase 6/7) handle those two cases differently (parsing failure
        can retry with a stricter prompt; call failure retries the call itself).
        """
        raise NotImplementedError

    def extract(
        self,
        image: Image.Image,
        prompt: str,
        *,
        page_number: int,
        system_instruction: str = "",
    ) -> RawVLMResponse:
        """Call the model and return a uniform RawVLMResponse.

        Deliberately does NOT parse the JSON here — that's response_parser.py's job,
        called by the extraction pipeline (Phase 6). Keeping this method's
        responsibility to "call the model, capture text + timing + errors" keeps the
        interface stable even as parsing strategy evolves.
        """
        from app.extraction.response_parser import parse_vlm_json_response
        from app.core.exceptions import VisionModelResponseParsingError

        start = time.monotonic()
        try:
            raw_text = self._call_model(image, prompt, system_instruction)
        except Exception as exc:  # backend-specific exceptions all funnel through here
            latency = time.monotonic() - start
            return RawVLMResponse(
                page_number=page_number,
                model_name=self.name,
                raw_text="",
                parsed_json=None,
                latency_seconds=latency,
                succeeded=False,
                error_message=str(exc),
            )

        latency = time.monotonic() - start

        try:
            parsed = parse_vlm_json_response(raw_text)
        except VisionModelResponseParsingError as exc:
            return RawVLMResponse(
                page_number=page_number,
                model_name=self.name,
                raw_text=raw_text,
                parsed_json=None,
                latency_seconds=latency,
                succeeded=False,
                error_message=str(exc),
            )

        return RawVLMResponse(
            page_number=page_number,
            model_name=self.name,
            raw_text=raw_text,
            parsed_json=parsed,
            latency_seconds=latency,
            succeeded=True,
            self_reported_confidence=parsed.get("field_confidence", {}) if isinstance(parsed, dict) else {},
        )
