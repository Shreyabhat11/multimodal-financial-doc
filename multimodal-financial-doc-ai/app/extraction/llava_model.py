"""
LLaVAModel — LLaVA-NeXT (LLaVA-1.6) backend, completing the
BaseVisionModel / QwenVLModel / LLaVAModel trio from the architecture diagram.

Structurally this file deliberately mirrors qwen_vl_model.py closely — same
`backend="local"|"hf-inference"` split, same lazy-import strategy, same retry policy.
That symmetry is not accidental: it's what makes both backends genuinely
interchangeable behind BaseVisionModel rather than "technically implementing the same
interface but with different runtime behavior/failure modes." When two
implementations of a Strategy pattern diverge in structure, they usually also diverge
in reliability characteristics — keeping them parallel is a deliberate consistency
discipline, not just style preference.

--- Hardware requirements ---

Local inference (backend="local"):
  - llava-v1.6-mistral-7b in bf16: ~15-16GB VRAM, comparable to Qwen2-VL-7B.
  - 4-bit quantized: ~7-8GB VRAM.
  - llava-v1.6-vicuna-13b (higher quality on complex table layouts in informal
    benchmarking) needs proportionally more: ~26GB bf16, ~10-11GB 4-bit.
  - Same CPU caveat as Qwen: not practical for real document volumes.

HF Inference API (backend="hf-inference"):
  - Same trade-offs as the Qwen HF-inference path — no local GPU, but subject to
    model availability on the serverless tier and to your account's rate limits.

--- Why offer both Qwen-VL and LLaVA at all? ---

This is a fair interview question and worth having a real answer, not just "the brief
asked for it": different VLM families have different failure modes on dense tabular
documents (a bank statement is much closer to "a data table" than to the natural
images most VLM benchmarks are built around). Having two backends behind one interface
lets you A/B them empirically on your own document set via the evaluation harness
(Phase 15) rather than picking one on vibes — and in a real production system, the
right move is often exactly this: evaluate multiple candidate models on your actual
data before committing.
"""

from __future__ import annotations

from typing import Literal

from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import VisionModelTimeoutError
from app.extraction.base_vision_model import BaseVisionModel

LLaVABackend = Literal["local", "hf-inference"]


class LLaVAModel(BaseVisionModel):
    name = "llava-next"

    def __init__(
        self,
        *,
        backend: LLaVABackend = "hf-inference",
        model_name: str | None = None,
        device: str | None = None,
        load_in_4bit: bool | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        timeout_seconds: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        settings = get_settings()
        self.backend: LLaVABackend = backend
        self.model_name = model_name or settings.llava_model_name
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
        """Lazy import — see qwen_vl_model.py for why (avoids a hard torch dependency
        for deployments that only use the hf-inference backend)."""
        import torch
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        quantization_config = None
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self._local_model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            quantization_config=quantization_config,
        )
        self._local_processor = LlavaNextProcessor.from_pretrained(self.model_name)

    def _call_local(self, image: Image.Image, prompt: str, system_instruction: str) -> str:
        # LLaVA-NeXT chat models use the same processor.apply_chat_template pattern as
        # Qwen2-VL in recent transformers versions, but LLaVA has no distinct "system"
        # role in its default chat template — we fold the system instruction into the
        # user turn as a prefix, which is the documented workaround for LLaVA chat
        # templates that don't define a system slot.
        combined_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": combined_prompt},
                ],
            }
        ]
        text_prompt = self._local_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._local_processor(text=text_prompt, images=image, return_tensors="pt").to(
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
        from app.extraction.qwen_vl_model import _image_to_base64_data_url

        image_url = _image_to_base64_data_url(image)
        combined_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": combined_prompt},
                ],
            }
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
        try:
            if self.backend == "local":
                return self._call_local(image, prompt, system_instruction)
            return self._call_hf_inference(image, prompt, system_instruction)
        except TimeoutError as exc:
            raise VisionModelTimeoutError(
                f"LLaVA ({self.backend}) call timed out after {self.timeout_seconds}s"
            ) from exc
