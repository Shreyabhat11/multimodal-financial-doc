"""
Vision model factory.

This is the ONLY place in the codebase that branches on `settings.model_provider`.
Every other component (the LangGraph extraction node, the evaluation harness, tests)
depends on `BaseVisionModel` and asks this factory for an instance — they never
import a concrete backend class directly. That's what makes "switch models through
configuration" (brief, Section 4) literally true rather than aspirational: change
MODEL_PROVIDER in .env, restart, done.

Note on `model_provider="hf-inference"`: this always resolves to a Qwen2-VL backend
called via the Hugging Face Inference API. LLaVA's local-only local backend is
selected explicitly via `model_provider="llava"`. If you want LLaVA over the HF
Inference API instead of locally, instantiate `LLaVAModel(backend="hf-inference", ...)`
directly rather than through this factory — the factory encodes the two most common
configurations, not every backend x provider combination.
"""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.extraction.base_vision_model import BaseVisionModel


def get_vision_model(settings: Settings | None = None) -> BaseVisionModel:
    """Instantiate the vision model backend configured via MODEL_PROVIDER."""
    settings = settings or get_settings()

    if settings.model_provider == "qwen-vl":
        from app.extraction.qwen_vl_model import QwenVLModel

        return QwenVLModel(backend="local", model_name=settings.model_name)

    if settings.model_provider == "hf-inference":
        from app.extraction.qwen_vl_model import QwenVLModel

        return QwenVLModel(backend="hf-inference", model_name=settings.model_name)

    if settings.model_provider == "llava":
        from app.extraction.llava_model import LLaVAModel

        return LLaVAModel(backend="local", model_name=settings.llava_model_name)

    raise ValueError(
        f"Unknown MODEL_PROVIDER '{settings.model_provider}'. "
        "Expected one of: 'qwen-vl', 'llava', 'hf-inference'."
    )
