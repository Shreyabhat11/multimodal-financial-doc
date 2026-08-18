"""
Shared parsing/validation helpers for schema fields.

Why this module exists: VLM output is free-text-ish JSON — amounts might arrive as
"1,234.56", "$1,234.56", "(1,234.56)" (accounting negative notation), or already-numeric.
Dates might arrive as "12/31/2024", "2024-12-31", or "31 Dec 2024". Every schema that
touches money or dates needs the same normalization, so it lives here once instead of
being copy-pasted into five ``field_validator`` methods.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

_AMOUNT_CLEAN_RE = re.compile(r"[^\d.\-()]")

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%Y/%m/%d",
)


def parse_amount(value) -> Decimal:
    """Parse a monetary value into a Decimal.

    Accepts Decimal/int/float/str. Handles thousands separators, currency symbols,
    and accounting-style negatives written as "(123.45)". Raises ValueError on
    anything that can't be confidently parsed — callers (Pydantic field_validators)
    convert that into a schema validation error rather than silently defaulting to 0,
    since silently defaulting a money field to zero is exactly the kind of bug that
    would corrupt a balance reconciliation without anyone noticing.
    """
    if value is None:
        raise ValueError("Amount value is None")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))

    text = str(value).strip()
    if text == "":
        raise ValueError("Amount value is an empty string")

    is_negative = text.startswith("(") and text.endswith(")")
    cleaned = _AMOUNT_CLEAN_RE.sub("", text)
    cleaned = cleaned.replace("(", "").replace(")", "")

    if cleaned.count("-") > 1:
        raise ValueError(f"Cannot parse amount: {value!r}")

    try:
        amount = Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError(f"Cannot parse amount: {value!r}") from exc

    if is_negative:
        amount = -abs(amount)
    return amount


def parse_flexible_date(value) -> date:
    """Parse a date from any of the common financial-document date formats.

    Raises ValueError if no known format matches, which is deliberate — an
    unparseable date should surface as an anomaly (Phase 11: MISSING_DATE /
    INVALID_DATE), not get coerced into some arbitrary default date.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    if not text:
        raise ValueError("Date value is an empty string")

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    raise ValueError(f"Cannot parse date: {value!r}")


def mask_account_number(account_number: str, keep_last_n: int = 4) -> str:
    """Mask all but the last N digits of an account number, e.g. '1234567890' -> '******7890'.

    Used both by the logging masking utility (Phase 20) and by API responses where a
    caller has requested a masked view. Kept here (not just in the logging module) so
    schemas can expose a ``masked_account_number`` computed field without importing
    the logging subsystem.
    """
    if not account_number:
        return account_number
    if len(account_number) <= keep_last_n:
        return "*" * len(account_number)
    return "*" * (len(account_number) - keep_last_n) + account_number[-keep_last_n:]
