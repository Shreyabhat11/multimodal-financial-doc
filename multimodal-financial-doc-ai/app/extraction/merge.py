"""
merge_page_results - combines a list of per-page PageExtractionOutcome objects into
one raw (still-unvalidated) document-level dict.

"Raw" and "still-unvalidated" are doing real work in that sentence: this function
does NOT construct Pydantic schema objects. It produces a plain dict shaped like the
target schema, which the LangGraph `normalize_transactions` and `validate_schema`
nodes (Phase 7) then parse into real Transaction/FinalExtractionResult objects. Keeping
this function schema-agnostic (dicts in, dict out) means a malformed page doesn't
raise here and abort the whole document - it just contributes nothing to the merge,
and schema validation later reports exactly which fields were missing/invalid, which
is much more actionable for the human-review flow than a merge-time crash.
"""

from __future__ import annotations

from app.extraction.page_extractor import PageExtractionOutcome

# Document-level fields we look for on each page's parsed JSON, in the order we
# prefer to trust them (first page that has a non-null value for a field wins) - most
# statements put this metadata on page 1, but multi-page statements occasionally
# repeat it in a footer on later pages too, so we don't hard-code "only page 1."
_DOCUMENT_LEVEL_FIELDS = (
    "document_type",
    "account_holder",
    "account_number",
    "bank_name",
    "statement_period",
    "opening_balance",
    "closing_balance",
    "currency",
)


def merge_page_results(outcomes: list[PageExtractionOutcome]) -> dict:
    """Merge per-page parsed JSON into one document-level raw dict.

    - Document-level fields: first non-null value encountered, in page order.
    - transactions: concatenated across pages, in page order, each tagged with
      `source_page` so downstream review UI / debugging can trace a transaction back
      to the page it was read from.
    - totals: collected the same way as other document-level fields (whatever a page
      explicitly reported) - this is the VLM's own read of a totals line, kept
      distinct from the deterministically COMPUTED totals that financial_validation
      (Phase 8) will calculate from the merged transaction list. Both exist so
      validation can compare "what the document says" against "what the numbers add
      up to."
    - page_extraction_metadata: per-page confidence/source bookkeeping, useful for
      the confidence aggregation step (Phase 11) and for debugging/human review.
    """
    merged: dict = {field: None for field in _DOCUMENT_LEVEL_FIELDS}
    merged["transactions"] = []
    merged["totals"] = None
    merged["page_extraction_metadata"] = []

    sorted_outcomes = sorted(outcomes, key=lambda o: o.page_number)

    for outcome in sorted_outcomes:
        merged["page_extraction_metadata"].append(
            {
                "page_number": outcome.page_number,
                "source": outcome.source,
                "confidence": outcome.confidence,
                "used_fallback": outcome.used_fallback,
            }
        )

        if not outcome.succeeded or outcome.parsed_json is None:
            continue

        page_data = outcome.parsed_json

        for field_name in _DOCUMENT_LEVEL_FIELDS:
            if merged[field_name] is None and page_data.get(field_name):
                merged[field_name] = page_data[field_name]

        if merged["totals"] is None and page_data.get("totals"):
            merged["totals"] = page_data["totals"]

        for txn in page_data.get("transactions") or []:
            if isinstance(txn, dict):
                txn_with_source = dict(txn)
                txn_with_source["source_page"] = outcome.page_number
                merged["transactions"].append(txn_with_source)

    return merged
