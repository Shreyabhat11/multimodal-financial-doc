"""
QwenVLModel — Qwen2-VL backend, with two interchangeable call strategies selected by
`backend`:

  - backend="local"        -> runs transformers.Qwen2VLForConditionalGeneration locally
  - backend="hf-inference" -> calls the Hugging Face Inference API (no local GPU needed)

Both strategies live in one class (rather than two separate classes) because they
share everything except the actual model call: same prompt building, same retry
policy, same RawVLMResponse shape. The `backend` value comes straight from
`settings.model_provider` ("qwen-vl" implies local; "hf-inference" implies the API
path with whatever `settings.model_name` is configured) — see model_factory.py.

--- Hardware requirements (stated plainly, not glossed over) ---

Local inference (backend="local"):
  - Qwen2-VL-7B-Instruct in bf16: ~16GB VRAM comfortably (weights + activations +
    KV cache for a multi-page prompt).
  - With 4-bit quantization (bitsandbytes, MODEL_LOAD_IN_4BIT=true): ~6-7GB VRAM.
    Quality degrades somewhat, most noticeably on fine table-structure reasoning —
    acceptable for a portfolio project, a real production system would benchmark
    this trade-off explicitly (see evaluation/, Phase 15).
  - CPU-only inference of a 7B VLM is not practical for anything beyond a single-page
    smoke test — expect multiple minutes per page. This is stated explicitly rather
    than pretended away: if you don't have a >=8GB CUDA GPU, use backend="hf-inference".

HF Inference API (backend="hf-inference"):
  - No local GPU required. Requires HF_TOKEN and (for larger/gated models) either a
    Pro account or a dedicated Inference Endpoint, since free-tier serverless
    inference does not host every model at all times — check model availability
    before relying on it for a demo.
"""

from __future__ import annotations

import base64
import io
from typing import Literal

from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import VisionModelTimeoutError
from app.extraction.base_vision_model import BaseVisionModel

QwenBackend = Literal["local", "hf-inference"]


def _image_to_base64_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


class QwenVLModel(BaseVisionModel):
    name = "qwen2-vl"

    def __init__(
        self,
        *,
        backend: QwenBackend = "hf-inference",
        model_name: str | None = None,
        device: str | None = None,
        load_in_4bit: bool | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        timeout_seconds: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        settings = get_settings()
        self.backend: QwenBackend = backend
        self.model_name = model_name or settings.model_name
        self.device = device or settings.model_device
        self.load_in_4bit = settings.model_load_in_4bit if load_in_4bit is None else load_in_4bit
        self.max_new_tokens = max_new_tokens or settings.model_max_new_tokens
        self.temperature = settings.model_temperature if temperature is None else temperature
        self.timeout_seconds = timeout_seconds or settings.vlm_request_timeout_seconds
        self.max_retries = settings.vlm_max_retries if max_retries is None else max_retries
        self.hf_token = settings.hf_token

        self._local_model = None
        self._local_processor = None
        self._hf_client = None

        if self.backend == "local":
            self._load_local_model()
        else:
            self._init_hf_client()

    # ------------------------------------------------------------------
    # Local transformers backend
    # ------------------------------------------------------------------

    def _load_local_model(self) -> None:
        """Lazily import heavy ML deps so importing this module doesn't require torch
        to be installed when only the hf-inference backend is used (e.g. in the API
        service container, which may intentionally not ship torch/CUDA at all —
        see docker-compose.yml, Phase 17, where the API service is CPU-only and a
        separate optional GPU-enabled worker handles local inference)."""
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        quantization_config = None
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self._local_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            quantization_config=quantization_config,
        )
        self._local_processor = AutoProcessor.from_pretrained(self.model_name)

    def _call_local(self, image: Image.Image, prompt: str, system_instruction: str) -> str:
        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text_prompt = self._local_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._local_processor(text=[text_prompt], images=[image], return_tensors="pt").to(
            self._local_model.device
        )
        output_ids = self._local_model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = self._local_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return decoded[0]

    # ------------------------------------------------------------------
    # Hugging Face Inference API backend
    # ------------------------------------------------------------------

    def _init_hf_client(self) -> None:
        from huggingface_hub import InferenceClient

        self._hf_client = InferenceClient(model=self.model_name, token=self.hf_token, timeout=self.timeout_seconds)

    def _call_hf_inference(self, image: Image.Image, prompt: str, system_instruction: str) -> str:
        image_url = _image_to_base64_data_url(image)
        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        completion = self._hf_client.chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=max(self.temperature, 1e-5),
        )
        return completion.choices[0].message.content

    # ------------------------------------------------------------------
    # BaseVisionModel interface
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True,
    )
    def _call_model(self, image: Image.Image, prompt: str, system_instruction: str) -> str:
        """Retries transient network/timeout errors with exponential backoff (2s, 4s,
        8s...). Does NOT retry on e.g. auth errors or malformed-input errors — those
        will fail identically on retry, so retrying just burns latency and quota.
        `tenacity`'s `retry_if_exception_type` filter is what enforces that
        distinction; local inference errors (OOM, etc.) are also not retried since a
        4th retry of an out-of-memory call will not somehow have memory next time.
        """
        try:
            if self.backend == "local":
                return self._call_local(image, prompt, system_instruction)
            return self._call_hf_inference(image, prompt, system_instruction)
        except TimeoutError as exc:
            raise VisionModelTimeoutError(
                f"Qwen-VL ({self.backend}) call timed out after {self.timeout_seconds}s"
            ) from exc
