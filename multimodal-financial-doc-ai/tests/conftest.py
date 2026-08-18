"""
Shared fixtures for the whole test suite.

`fake_good_vlm` / `fake_wrong_balance_vlm` / `crashing_vlm` etc. are the same fake
backend pattern used throughout interactive development (Phases 4-10) — reused here
rather than reinvented, since they're already proven to correctly exercise
BaseVisionModel's contract.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-for-pytest")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

from app.extraction.base_vision_model import BaseVisionModel, RawVLMResponse  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_pdf_bytes() -> bytes:
    """A real, valid 3-page PDF (1 page rotated), generated fresh per test session
    rather than checked in as a binary fixture — keeps the repo binary-free and the
    fixture trivially reviewable as code."""
    import pymupdf

    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), f"Bank Statement - Page {i + 1}", fontsize=16)
        page.insert_text((72, 110), "Account Holder: Jane Doe", fontsize=11)
        page.insert_text((72, 128), "Account Number: 9988776655", fontsize=11)
    doc[1].set_rotation(90)
    buffer = doc.tobytes()
    doc.close()
    return buffer


@pytest.fixture
def fake_good_vlm() -> BaseVisionModel:
    """A fake VLM that returns a correct, internally-consistent 3-page bank statement."""

    class GoodFakeVLM(BaseVisionModel):
        name = "fake-good"

        def _call_model(self, image, prompt, system_instruction):
            if "page 1 of 3" in prompt:
                return json.dumps(
                    {
                        "account_holder": "Jane Doe",
                        "account_number": "9988776655",
                        "bank_name": "First National",
                        "statement_period": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
                        "opening_balance": 1000.00,
                        "currency": "USD",
                        "transactions": [
                            {"date": "2024-01-05", "description": "Salary", "credit": 500.00, "debit": 0}
                        ],
                        "field_confidence": {"account_holder": 0.95, "account_number": 0.95, "opening_balance": 0.9},
                    }
                )
            if "page 2 of 3" in prompt:
                return json.dumps(
                    {
                        "closing_balance": 1300.00,
                        "transactions": [
                            {"date": "2024-01-10", "description": "Groceries", "debit": 200.00, "credit": 0}
                        ],
                        "totals": {"total_debits": 200.00, "total_credits": 500.00},
                        "field_confidence": {"closing_balance": 0.92},
                    }
                )
            return json.dumps({"transactions": [], "field_confidence": {}})

    return GoodFakeVLM()


@pytest.fixture
def fake_wrong_balance_vlm() -> BaseVisionModel:
    """A fake VLM that reports a closing balance that does NOT reconcile with the
    transaction list — used to test that financial_validation actually catches it."""

    class WrongBalanceVLM(BaseVisionModel):
        name = "fake-wrong-balance"

        def _call_model(self, image, prompt, system_instruction):
            if "page 1 of 3" in prompt:
                return json.dumps(
                    {
                        "account_holder": "Jane Doe",
                        "account_number": "9988776655",
                        "statement_period": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
                        "opening_balance": 1000.00,
                        "currency": "USD",
                        "transactions": [
                            {"date": "2024-01-05", "description": "Salary", "credit": 500.00, "debit": 0}
                        ],
                        "field_confidence": {},
                    }
                )
            if "page 2 of 3" in prompt:
                return json.dumps(
                    {
                        "closing_balance": 9999.00,  # WRONG: should be 1300.00
                        "transactions": [
                            {"date": "2024-01-10", "description": "Groceries", "debit": 200.00, "credit": 0}
                        ],
                        "field_confidence": {},
                    }
                )
            return json.dumps({"transactions": [], "field_confidence": {}})

    return WrongBalanceVLM()


@pytest.fixture
def crashing_vlm() -> BaseVisionModel:
    """A fake VLM that always raises, for testing failure/retry/fallback paths."""

    class CrashingVLM(BaseVisionModel):
        name = "fake-crashing"

        def _call_model(self, image, prompt, system_instruction):
            raise ConnectionError("simulated VLM outage")

    return CrashingVLM()


@pytest.fixture
def malformed_output_vlm() -> BaseVisionModel:
    """A fake VLM that returns text that cannot be parsed as JSON at all."""

    class MalformedVLM(BaseVisionModel):
        name = "fake-malformed"

        def _call_model(self, image, prompt, system_instruction):
            return "I'm sorry, I cannot read this page clearly. It appears to be blank."

    return MalformedVLM()


@pytest.fixture(autouse=True)
def _fake_crew_validation(monkeypatch):
    """Monkeypatch CrewAI validation to a deterministic pass by default across the
    whole test suite — no test in this repo should make a real Anthropic API call.
    Individual tests that need a specific crew outcome override this again locally.
    """
    import app.agents.crew as crew_module
    from app.schemas.validation import ValidationResult

    def _fake(**kwargs):
        return ValidationResult.passed("crew_validation", ["extraction_review"])

    monkeypatch.setattr(crew_module, "run_crew_validation", _fake)


@pytest.fixture
def sqlite_db_url(tmp_path: Path) -> str:
    db_file = tmp_path / "test.db"
    return f"sqlite:///{db_file}"


@pytest.fixture
def db_session(sqlite_db_url):
    """A real SQLAlchemy session against a fresh temp-file SQLite database, tables
    created via Base.metadata.create_all — see PROGRESS.md for why SQLite (not
    Postgres) is used in this test environment, and why the ORM's use of only
    cross-dialect-portable column types makes this a legitimate correctness test."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models.orm import Base

    engine = create_engine(sqlite_db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
