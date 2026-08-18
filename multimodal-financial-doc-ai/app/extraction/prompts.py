"""
Prompt templates for VLM-based financial document extraction.

Design decisions worth calling out:

1. The prompt asks for JSON matching our Pydantic schema shape directly (field names
   line up with app.schemas.Transaction / Account / etc.), so response_parser.py can
   feed the parsed dict straight into schema validation with minimal remapping.
2. The prompt explicitly asks the model to self-report a confidence score per major
   field. VLMs are not calibrated probability estimators, so we do NOT treat this as a
   ground-truth probability — it's one signal fed into DocumentConfidence alongside
   deterministic signals (schema validity, reconciliation success). See
   app/validation/confidence.py (Phase 11).
3. The prompt tells the model to leave a field as null/empty rather than guess when
   illegible — this is the single highest-leverage instruction for reducing
   hallucination in extraction tasks: an empty field is safe (it routes to human
   review), a confidently wrong number is not.
4. Temperature is controlled by the caller (settings.model_temperature, default 0.1) —
   the prompt itself doesn't encode sampling params, since those belong to the
   inference call, not the instructions.
"""

from __future__ import annotations

from app.schemas.enums import DocumentType

_JSON_SHAPE = """{
  "document_type": "bank_statement | credit_card_statement | invoice | financial_report | loan_statement",
  "account_holder": "string",
  "account_number": "string, exactly as printed (do not mask it)",
  "bank_name": "string",
  "statement_period": {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"},
  "opening_balance": "number, no currency symbol or thousands separators",
  "closing_balance": "number, no currency symbol or thousands separators",
  "currency": "3-letter ISO code, e.g. USD",
  "transactions": [
    {
      "date": "YYYY-MM-DD",
      "description": "string, exactly as printed",
      "reference": "string or null",
      "debit": "number, 0 if this row is not a debit",
      "credit": "number, 0 if this row is not a credit",
      "balance": "number or null, the running balance if printed on this row"
    }
  ],
  "totals": {"total_debits": "number", "total_credits": "number"},
  "field_confidence": {
    "account_number": "float 0.0-1.0",
    "opening_balance": "float 0.0-1.0",
    "closing_balance": "float 0.0-1.0",
    "transactions": "float 0.0-1.0, your confidence in the completeness/accuracy of the full transaction list for THIS page"
  },
  "extraction_notes": "string or null — mention anything illegible, ambiguous, or unusual about this page"
}"""


SYSTEM_INSTRUCTION = """You are a meticulous financial document analyst. You read financial statement \
page images (bank statements, credit card statements, invoices, loan statements) and extract \
their content into strict JSON. You understand tables, transaction rows, running balances, \
headers, logos, stamps, and page layout — you do not rely on OCR, you read the page visually \
as a human analyst would.

Rules you always follow:
- Output ONLY valid JSON. No markdown code fences, no commentary before or after the JSON.
- If a field is illegible, not present on this page, or you are not confident, use null \
(or an empty list for transactions) rather than guessing. A missing value is safe; a \
confidently wrong value is not.
- Numbers must be plain numbers (e.g. 1234.56), never containing currency symbols, commas, \
or parentheses for negatives — use a leading minus sign instead if the source shows a \
negative/accounting-style figure.
- Every transaction row you see on this page must appear in the "transactions" array, in the \
same order as printed. Do not summarize, merge, or omit rows.
- Populate "field_confidence" honestly based on print clarity, table structure, and how \
certain you are you read each value correctly — not on how important the field is."""


def build_page_extraction_prompt(
    *,
    page_number: int,
    total_pages: int,
    expected_document_type: DocumentType | None = None,
) -> str:
    """Build the user-turn prompt sent alongside a single page image.

    Each page is processed independently (see extraction pipeline, Phase 6) — the
    prompt is deliberately page-scoped ("account_holder"/"opening_balance" will often
    be null on pages after page 1, which is expected and handled by merge_page_results).
    """
    doc_type_hint = (
        f'You have been told this document is very likely a "{expected_document_type.value}" — '
        "use that as a prior, but if what you see clearly contradicts it, report what you actually see."
        if expected_document_type
        else "Determine the document type from what you see on the page."
    )

    return f"""This is page {page_number} of {total_pages} of a financial statement. {doc_type_hint}

Extract everything you can find on THIS page into the following JSON shape. Leave \
document-level fields (account_holder, account_number, bank_name, statement_period, \
opening_balance, closing_balance, currency, totals) null if they are not shown on this \
specific page — do not guess values you can't see. The "transactions" array should only \
contain the transaction rows visible on this page.

JSON shape to fill in:
{_JSON_SHAPE}

Respond with ONLY the JSON object."""


def build_ocr_structuring_prompt(
    *,
    ocr_text: str,
    page_number: int,
    total_pages: int,
    expected_document_type: DocumentType | None = None,
) -> str:
    """Build the text-only prompt used by the OCR fallback path.

    Used when vision-first extraction failed or scored low confidence for a page.
    Rather than re-showing the image, we hand the model Tesseract's raw OCR text and
    ask it to structure that instead — cheaper and more reliable than a second image
    call when the underlying problem is likely image quality (in which case a second
    VLM pass over the same pixels would probably fail the same way), and OCR text,
    while lossy about layout, is still readable.
    """
    doc_type_hint = (
        f'This document is very likely a "{expected_document_type.value}".'
        if expected_document_type
        else "Infer the document type from the text."
    )

    return f"""This is OCR-extracted raw text from page {page_number} of {total_pages} of a \
financial statement. {doc_type_hint} OCR text loses table structure and can contain \
recognition errors (e.g. "O" vs "0", misjoined columns) — use your judgement to reconstruct \
the likely correct structure, but if a value is genuinely ambiguous, use null rather than guessing.

--- RAW OCR TEXT START ---
{ocr_text}
--- RAW OCR TEXT END ---

Structure this into the following JSON shape (same rules as normal extraction: null for \
anything not present or too ambiguous to trust, plain numbers with no symbols/commas):

{_JSON_SHAPE}

Respond with ONLY the JSON object."""
