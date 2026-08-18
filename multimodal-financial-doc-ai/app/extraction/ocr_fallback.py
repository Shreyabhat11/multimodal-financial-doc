"""
OCRFallbackModel — completes the vision-first -> confidence-check -> OCR-fallback
design (brief, Section 7).

Implementation note on responsibility split: this class implements BaseVisionModel
(so PageExtractor, Phase 6 below, can treat it uniformly alongside QwenVLModel /
LLaVAModel), but it is a TEXT completion backend, not a second vision model — the
`image` argument to `extract()` is accepted for interface compatibility but is not
sent anywhere; by the time this class is invoked, the caller (PageExtractor) has
already run Tesseract on the image and baked the resulting OCR text into `prompt` via
`build_ocr_structuring_prompt`. This keeps the OCR step itself (pixels -> text,
ocr_engine.py) decoupled from the structuring step (text -> JSON, here) — you could
swap Tesseract for a different OCR engine without touching this class at all, since
this class never touches pixels.

--- OCR vs. VLM vs. hybrid trade-offs (brief asks this be explained explicitly) ---

Pure OCR:
  + Fast, cheap, no GPU/API cost, deterministic.
  - Loses table structure entirely (returns text in reading order, not rows/columns) —
    a bank statement's debit/credit columns can get scrambled into a single stream of
    numbers with no reliable way to tell which column a number came from.
  - No semantic understanding — can't distinguish a running balance from a transaction
    amount by position/context the way a human (or VLM) can from layout.

Pure VLM:
  + Understands layout, tables, and semantics jointly — this is why it's the primary
    path in this system.
  - Slower and more expensive per page than OCR.
  - Can still fail on very low-quality scans, unusual fonts, or dense small print
    where OCR's per-character recognition can actually outperform a VLM's holistic
    reading.

Hybrid (this system's approach):
  Vision-first for the primary pass (best average-case accuracy on layout-heavy
  documents), OCR text-structuring as a fallback specifically for low-confidence
  pages — not as a competing pass run on every page. This gets most of the VLM's
  layout understanding on the common case while providing a cheaper, different-failure-
  mode recovery path for the pages where the VLM itself signals (or its output implies)
  it struggled.
"""

from __future__ import annotations

from typing import Callable

from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import VisionModelTimeoutError
from app.extraction.base_vision_model import BaseVisionModel

#: Signature for an injectable text-completion function — (prompt, system_instruction) -> raw_text.
#: Used both for the real HF Inference API call and for unit tests, which inject a fake.
TextCompletionFn = Callable[[str, str], str]


class OCRFallbackModel(BaseVisionModel):
    name = "ocr-fallback"

    def __init__(
        self,
        *,
        text_model_name: str | None = None,
        timeout_seconds: int | None = None,
        text_completion_fn: TextCompletionFn | None = None,
    ) -> None:
        settings = get_settings()
        self.text_model_name = text_model_name or settings.model_name
        self.timeout_seconds = timeout_seconds or settings.vlm_request_timeout_seconds
        self.hf_token = settings.hf_token

        # Dependency injection point: production code leaves this None and gets the
        # real HF Inference API text-completion call; tests inject a fake function so
        # they can exercise the retry/error-handling logic without network access.
        self._text_completion_fn = text_completion_fn or self._default_text_completion

        self._hf_client = None
        if text_completion_fn is None:
            self._init_hf_client()

    def _init_hf_client(self) -> None:
        from huggingface_hub import InferenceClient

        self._hf_client = InferenceClient(model=self.text_model_name, token=self.hf_token, timeout=self.timeout_seconds)

    def _default_text_completion(self, prompt: str, system_instruction: str) -> str:
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]
        completion = self._hf_client.chat_completion(messages=messages, max_tokens=2048, temperature=0.1)
        return completion.choices[0].message.content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True,
    )
    def _call_model(self, image: Image.Image, prompt: str, system_instruction: str) -> str:
        # `image` intentionally unused — see module docstring. Accepted only to
        # satisfy the BaseVisionModel interface.
        try:
            return self._text_completion_fn(prompt, system_instruction)
        except TimeoutError as exc:
            raise VisionModelTimeoutError(
                f"OCR-fallback text structuring call timed out after {self.timeout_seconds}s"
            ) from exc
