"""
Integration tests: the full LangGraph pipeline, invoked end to end via
build_document_graph — real PDF preprocessing, real merge/normalize/validate logic,
fake VLM backends (no network), and the crew_validation node monkeypatched by the
autouse `_fake_crew_validation` fixture in conftest.py.
"""

from __future__ import annotations

from app.core.config import Settings
from app.graph.graph_builder import build_document_graph
from app.schemas.enums import DocumentType, ProcessingStatus


class TestFullPipelineHappyPath:
    def test_multi_page_document_completes_successfully(self, synthetic_pdf_bytes, fake_good_vlm):
        settings = Settings(anthropic_api_key="fake")
        graph = build_document_graph(fake_good_vlm, settings=settings)

        final_state = graph.invoke(
            {
                "document_id": "doc_multipage",
                "file_bytes": synthetic_pdf_bytes,
                "original_filename": "statement.pdf",
                "expected_document_type": DocumentType.BANK_STATEMENT,
            }
        )

        assert final_state["status"] == ProcessingStatus.COMPLETED
        assert final_state["final_document"]["account"]["account_number"] == "9988776655"
        assert len(final_state["final_document"]["transactions"]) == 2
        assert "confidence" in final_state["final_document"]

    def test_page_source_metadata_recorded_for_every_page(self, synthetic_pdf_bytes, fake_good_vlm):
        settings = Settings(anthropic_api_key="fake")
        graph = build_document_graph(fake_good_vlm, settings=settings)
        final_state = graph.invoke(
            {"document_id": "doc_meta", "file_bytes": synthetic_pdf_bytes, "original_filename": "statement.pdf"}
        )
        assert len(final_state["page_outcomes"]) == 3
        assert all(o.source == "vlm" for o in final_state["page_outcomes"])


class TestFinancialValidationCannotBeOverriddenByCrew:
    def test_wrong_balance_routes_to_human_review_even_when_crew_passes(
        self, synthetic_pdf_bytes, fake_wrong_balance_vlm
    ):
        """The single most important integration test in this project: proves a
        deterministic validation failure forces human review REGARDLESS of what the
        (mocked, always-passing) CrewAI validation reports."""
        settings = Settings(anthropic_api_key="fake")
        graph = build_document_graph(fake_wrong_balance_vlm, settings=settings)

        final_state = graph.invoke(
            {"document_id": "doc_wrong_balance", "file_bytes": synthetic_pdf_bytes, "original_filename": "s.pdf"}
        )

        assert final_state["status"] == ProcessingStatus.NEEDS_HUMAN_REVIEW
        financial_result = next(
            r for r in final_state["validation_results"] if r.validator_name == "financial_validation"
        )
        assert financial_result.status.value == "failed"
        crew_result = next(r for r in final_state["validation_results"] if r.validator_name == "crew_validation")
        assert crew_result.status.value == "passed"  # crew said fine -- didn't matter


class TestCorruptedDocumentHandling:
    def test_corrupted_pdf_fails_before_ever_calling_the_vlm(self, crashing_vlm):
        """crashing_vlm raises on every call -- if the graph ever reached
        extract_page_information, this test would fail with a ConnectionError
        instead of asserting FAILED, proving preprocessing correctly short-circuits."""
        settings = Settings(anthropic_api_key="fake")
        graph = build_document_graph(crashing_vlm, settings=settings)

        final_state = graph.invoke(
            {"document_id": "doc_corrupt", "file_bytes": b"not a real pdf", "original_filename": "bad.pdf"}
        )

        assert final_state["status"] == ProcessingStatus.FAILED
        assert "page_outcomes" not in final_state


class TestExtractionFailureAndRetry:
    def test_total_extraction_failure_retries_then_routes_to_human_review(self, synthetic_pdf_bytes, crashing_vlm):
        settings = Settings(anthropic_api_key="fake")
        graph = build_document_graph(crashing_vlm, settings=settings)

        final_state = graph.invoke(
            {"document_id": "doc_all_fail", "file_bytes": synthetic_pdf_bytes, "original_filename": "s.pdf"}
        )

        assert final_state["status"] == ProcessingStatus.NEEDS_HUMAN_REVIEW
        assert final_state["page_extraction_retry_count"] == 1  # confirms the retry loop actually fired
        assert len(final_state["schema_validation_errors"]) > 0


class TestMalformedVlmOutput:
    def test_malformed_json_output_does_not_crash_the_pipeline(self, synthetic_pdf_bytes, malformed_output_vlm):
        settings = Settings(anthropic_api_key="fake")
        graph = build_document_graph(malformed_output_vlm, settings=settings)

        final_state = graph.invoke(
            {"document_id": "doc_malformed", "file_bytes": synthetic_pdf_bytes, "original_filename": "s.pdf"}
        )

        # every page fails to parse -> same end state as total extraction failure
        assert final_state["status"] == ProcessingStatus.NEEDS_HUMAN_REVIEW
        assert all(o.source == "failed" for o in final_state["page_outcomes"])
