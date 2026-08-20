"""
Unit tests for app/core/config.py.

The api_cors_origins tests are a direct regression test for a real bug: pydantic-
settings' env/dotenv source auto-JSON-decodes any list-typed field read from an
environment variable BEFORE our own field_validator runs. A plain, non-JSON value —
exactly what .env.example ships (API_CORS_ORIGINS=http://localhost:8501) — used to
raise `SettingsError` at import time, breaking every entry point that imports
app.core.config (uvicorn, alembic, pytest itself). Fixed via the `NoDecode`
annotation. See app/core/config.py's `api_cors_origins` field comment for the full
explanation.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings


class TestApiCorsOriginsParsing:
    def test_plain_single_origin_from_env_does_not_raise(self, monkeypatch):
        """This is the exact scenario that used to crash `alembic upgrade head`."""
        monkeypatch.setenv("API_CORS_ORIGINS", "http://localhost:8501")
        settings = Settings()
        assert settings.api_cors_origins == ["http://localhost:8501"]

    def test_comma_separated_origins_from_env(self, monkeypatch):
        monkeypatch.setenv("API_CORS_ORIGINS", "http://localhost:8501,https://myapp.example.com")
        settings = Settings()
        assert settings.api_cors_origins == ["http://localhost:8501", "https://myapp.example.com"]

    def test_falls_back_to_default_when_env_var_absent(self, monkeypatch):
        monkeypatch.delenv("API_CORS_ORIGINS", raising=False)
        settings = Settings()
        assert settings.api_cors_origins == ["http://localhost:8501"]

    def test_whitespace_around_commas_is_trimmed(self, monkeypatch):
        monkeypatch.setenv("API_CORS_ORIGINS", " http://a.com , http://b.com ")
        settings = Settings()
        assert settings.api_cors_origins == ["http://a.com", "http://b.com"]


class TestConfidenceWeightsValidation:
    def test_default_weights_sum_to_one(self):
        settings = Settings()
        assert abs(sum(settings.confidence_weights.values()) - 1.0) < 1e-9

    def test_rejects_weights_that_do_not_sum_to_one(self):
        with pytest.raises(Exception):
            Settings(confidence_weights={"account_number": 0.5, "transactions": 0.6})


class TestSettingsSingleton:
    def test_get_settings_returns_cached_instance(self):
        from app.core.config import get_settings

        first = get_settings()
        second = get_settings()
        assert first is second
