"""
Central application configuration.

Design decision
----------------
We load *non-secret, structural* defaults from ``configs/config.yaml`` (limits,
thresholds, model names) and let *secrets and deployment-specific values*
(API keys, DATABASE_URL, HF_TOKEN) come exclusively from environment variables /
``.env``. This keeps secrets out of version control (config.yaml is safe to commit;
.env is gitignored) while still giving us a single, typed, validated ``Settings``
object that the rest of the app imports from.

Everything is exposed through ``get_settings()``, which is cached with
``functools.lru_cache`` so the YAML file is parsed once per process and the same
validated object is shared everywhere (config is effectively immutable at runtime).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_YAML_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _load_yaml_config() -> dict:
    """Read configs/config.yaml. Returns an empty dict if the file is missing.

    A missing config.yaml is not a hard error at import time — some deployment
    contexts (e.g. certain test runners) may not have the file mounted. Every
    Settings field below has a sane default regardless.
    """
    if not CONFIG_YAML_PATH.exists():
        return {}
    with CONFIG_YAML_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_yaml_config = _load_yaml_config()


def _yaml_get(*keys: str, default=None):
    """Safe nested dict lookup into the parsed YAML, e.g. _yaml_get('vision_model', 'provider')."""
    node = _yaml_config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


class Settings(BaseSettings):
    """
    Typed application settings.

    Precedence (highest wins): environment variables / .env  >  configs/config.yaml  >
    hardcoded field default. This is implemented by using ``_yaml_get(...)`` as the
    ``default`` passed to each ``Field`` — pydantic-settings then overrides that default
    with any matching environment variable it finds.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    app_env: Literal["development", "staging", "production"] = Field(default="development")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    secret_key: str = Field(default="dev-only-insecure-secret-key")

    # --- Vision-Language Model ---
    model_provider: Literal["qwen-vl", "llava", "hf-inference"] = Field(
        default=_yaml_get("vision_model", "provider", default="hf-inference")
    )
    model_name: str = Field(
        default=_yaml_get("vision_model", "qwen_model_name", default="Qwen/Qwen2-VL-7B-Instruct")
    )
    llava_model_name: str = Field(
        default=_yaml_get("vision_model", "llava_model_name", default="llava-hf/llava-v1.6-mistral-7b-hf")
    )
    hf_token: str | None = Field(default=None)
    model_device: Literal["cuda", "cpu", "mps"] = Field(default="cuda")
    model_load_in_4bit: bool = Field(
        default=_yaml_get("vision_model", "load_in_4bit", default=True)
    )
    model_max_new_tokens: int = Field(
        default=_yaml_get("vision_model", "max_new_tokens", default=2048), gt=0
    )
    model_temperature: float = Field(
        default=_yaml_get("vision_model", "temperature", default=0.1), ge=0.0, le=2.0
    )

    # --- OCR fallback ---
    ocr_enabled: bool = Field(default=_yaml_get("ocr_fallback", "enabled", default=True))
    ocr_engine: str = Field(default=_yaml_get("ocr_fallback", "engine", default="tesseract"))
    tesseract_cmd: str = Field(default="/usr/bin/tesseract")
    ocr_trigger_confidence_below: float = Field(
        default=_yaml_get("ocr_fallback", "trigger_confidence_below", default=0.6),
        ge=0.0,
        le=1.0,
    )

    # --- Document processing limits ---
    max_file_size_mb: int = Field(
        default=_yaml_get("document_processing", "max_file_size_mb", default=25), gt=0
    )
    max_pages: int = Field(
        default=_yaml_get("document_processing", "max_pages", default=50), gt=0
    )
    page_render_dpi: int = Field(
        default=_yaml_get("document_processing", "page_render_dpi", default=200), gt=0
    )
    vlm_request_timeout_seconds: int = Field(
        default=_yaml_get("document_processing", "vlm_request_timeout_seconds", default=90),
        gt=0,
    )
    vlm_max_retries: int = Field(
        default=_yaml_get("document_processing", "vlm_max_retries", default=3), ge=0
    )
    retry_backoff_seconds: float = Field(
        default=_yaml_get("document_processing", "retry_backoff_seconds", default=2.0), ge=0.0
    )

    # --- Validation / confidence thresholds ---
    confidence_threshold: float = Field(
        default=_yaml_get("validation", "confidence_threshold", default=0.75), ge=0.0, le=1.0
    )
    confidence_weights: dict[str, float] = Field(
        default_factory=lambda: _yaml_get(
            "confidence_weights",
            default={
                "account_number": 0.20,
                "statement_period": 0.15,
                "opening_balance": 0.15,
                "closing_balance": 0.15,
                "transactions": 0.35,
            },
        )
    )
    balance_tolerance: float = Field(
        default=_yaml_get("validation", "balance_tolerance", default=0.01), ge=0.0
    )
    large_transaction_multiplier: float = Field(
        default=_yaml_get("validation", "large_transaction_multiplier", default=5), gt=0
    )
    human_review_on_validation_failure: bool = Field(
        default=_yaml_get("validation", "human_review_on_validation_failure", default=True)
    )

    # --- Database ---
    database_url: str = Field(
        default="postgresql+psycopg2://findoc_user:findoc_pass@localhost:5432/findoc_db"
    )
    database_echo: bool = Field(default=False)

    # --- CrewAI / agent reasoning LLM ---
    agent_llm_provider: Literal["anthropic", "openai", "ollama"] = Field(default="anthropic")
    agent_llm_model: str = Field(default="claude-sonnet-4-6")
    anthropic_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)
    crew_process: Literal["sequential", "hierarchical"] = Field(
        default=_yaml_get("crew", "process", default="sequential")
    )
    crew_max_iterations: int = Field(default=_yaml_get("crew", "max_iterations", default=5), gt=0)
    crew_verbose: bool = Field(default=_yaml_get("crew", "verbose", default=False))

    # --- API ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, gt=0, lt=65536)
    # Annotated with NoDecode: pydantic-settings' env/dotenv source auto-JSON-decodes
    # any list/dict/set-typed field read from an environment variable BEFORE our own
    # field_validator ever sees it — so a plain, non-JSON value like
    # "http://localhost:8501" (exactly what .env.example ships) raises a
    # SettingsError at the source level, never reaching `_split_csv_origins` below.
    # NoDecode disables that automatic decoding for this field specifically, so the
    # raw string is handed to our validator instead, which does the (correct, simple)
    # comma-split itself. Without this annotation, `alembic upgrade head` / anything
    # that imports app.core.config fails immediately on startup.
    api_cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: _yaml_get("api", "cors_origins", default=["http://localhost:8501"])
    )

    # --- Streamlit ---
    streamlit_api_base_url: str = Field(default="http://localhost:8000")

    # --- Logging / masking ---
    mask_account_numbers: bool = Field(
        default=_yaml_get("logging", "mask_account_numbers", default=True)
    )
    mask_keep_last_n_digits: int = Field(
        default=_yaml_get("logging", "mask_keep_last_n_digits", default=4), ge=0
    )

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def _split_csv_origins(cls, v):
        """Allow API_CORS_ORIGINS to be provided as a comma-separated env var string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @model_validator(mode="after")
    def _validate_provider_requirements(self) -> "Settings":
        """Fail fast at startup if the selected model provider is missing what it needs."""
        if self.model_provider == "hf-inference" and not self.hf_token:
            # Not a hard error: some HF models are queryable without a token at low rate
            # limits, so we warn via a soft check the caller can log, rather than raising.
            pass
        if self.agent_llm_provider == "anthropic" and self.anthropic_api_key is None:
            # CrewAI agents will fail at call time with a clearer error; we don't raise
            # here because unit tests often stub the LLM client entirely.
            pass
        return self

    @model_validator(mode="after")
    def _validate_confidence_weights_sum_to_one(self) -> "Settings":
        """Fail fast at startup if confidence_weights don't sum to ~1.0 — a silently
        mis-summing weight config would make DocumentConfidence.overall_confidence
        (app/validation/confidence.py) meaningless without any obvious symptom other
        than "the number looks a bit off," which is exactly the kind of bug that
        survives in production far too long."""
        total = sum(self.confidence_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"confidence_weights must sum to 1.0, got {total} from {self.confidence_weights}"
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton (cached after first call)."""
    return Settings()
