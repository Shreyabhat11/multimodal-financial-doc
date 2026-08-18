"""Unit tests for app/extraction/merge.py and response_parser.py."""

from __future__ import annotations

from app.extraction.base_vision_model import RawVLMResponse
from app.extraction.merge import merge_page_results
from app.extraction.page_extractor import PageExtractionOutcome
from app.extraction.response_parser import parse_vlm_json_response
from app.core.exceptions import VisionModelResponseParsingError

import pytest


def _outcome(page_number, parsed_json, succeeded=True):
    response = RawVLMResponse(
        page_number=page_number, model_name="fake", raw_text="", parsed_json=parsed_json,
        latency_seconds=0.01, succeeded=succeeded,
    )
    return PageExtractionOutcome(
        page_number=page_number, parsed_json=parsed_json, confidence=0.9 if succeeded else 0.0,
        source="vlm" if succeeded else "failed", used_fallback=False, vlm_response=response,
    )


class TestMergePageResults:
    def test_document_level_fields_taken_from_first_page_that_has_them(self):
        outcomes = [
            _outcome(1, {"account_holder": "Jane Doe", "transactions": []}),
            _outcome(2, {"closing_balance": 1300.0, "transactions": []}),
        ]
        merged = merge_page_results(outcomes)
        assert merged["account_holder"] == "Jane Doe"
        assert merged["closing_balance"] == 1300.0

    def test_transactions_concatenated_with_source_page(self):
        outcomes = [
            _outcome(1, {"transactions": [{"date": "2024-01-05", "description": "A"}]}),
            _outcome(2, {"transactions": [{"date": "2024-01-10", "description": "B"}]}),
        ]
        merged = merge_page_results(outcomes)
        assert len(merged["transactions"]) == 2
        assert merged["transactions"][0]["source_page"] == 1
        assert merged["transactions"][1]["source_page"] == 2

    def test_failed_page_contributes_nothing_but_is_recorded_in_metadata(self):
        outcomes = [
            _outcome(1, {"account_holder": "Jane Doe", "transactions": []}),
            _outcome(2, None, succeeded=False),
        ]
        merged = merge_page_results(outcomes)
        assert merged["account_holder"] == "Jane Doe"
        assert len(merged["page_extraction_metadata"]) == 2
        assert merged["page_extraction_metadata"][1]["source"] == "failed"


class TestParseVlmJsonResponse:
    def test_clean_json(self):
        assert parse_vlm_json_response('{"a": 1}') == {"a": 1}

    def test_code_fenced_json(self):
        assert parse_vlm_json_response('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_surrounding_prose(self):
        result = parse_vlm_json_response('Here is the data:\n{"a": 1}\nLet me know if you need more.')
        assert result == {"a": 1}

    def test_trailing_comma_recovered(self):
        result = parse_vlm_json_response('{"a": 1, "list": [1, 2, 3,],}')
        assert result == {"a": 1, "list": [1, 2, 3]}

    def test_unparseable_text_raises(self):
        with pytest.raises(VisionModelResponseParsingError):
            parse_vlm_json_response("I could not read this page clearly.")

    def test_empty_response_raises(self):
        with pytest.raises(VisionModelResponseParsingError):
            parse_vlm_json_response("   ")
