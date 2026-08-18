"""
Pure formatting helpers — no Streamlit imports here on purpose, so these are
directly unit-testable without a Streamlit runtime (see the smoke tests run during
development). The UI module imports these rather than inlining formatting logic
into widget calls.
"""

from __future__ import annotations

STATUS_COLORS = {
    "uploaded": "🔵",
    "processing": "🟡",
    "validating": "🟡",
    "needs_human_review": "🟠",
    "completed": "🟢",
    "failed": "🔴",
}

STATUS_LABELS = {
    "uploaded": "Uploaded",
    "processing": "Processing",
    "validating": "Validating",
    "needs_human_review": "Needs Human Review",
    "completed": "Completed",
    "failed": "Failed",
}

SEVERITY_COLORS = {
    "low": "🟢",
    "medium": "🟡",
    "high": "🟠",
    "critical": "🔴",
}

VALIDATION_STATUS_COLORS = {
    "passed": "🟢",
    "passed_with_warnings": "🟡",
    "failed": "🔴",
}


def status_badge(status: str) -> str:
    icon = STATUS_COLORS.get(status, "⚪")
    label = STATUS_LABELS.get(status, status.replace("_", " ").title())
    return f"{icon} {label}"


def severity_badge(severity: str) -> str:
    icon = SEVERITY_COLORS.get(severity, "⚪")
    return f"{icon} {severity.upper()}"


def validation_status_badge(status: str) -> str:
    icon = VALIDATION_STATUS_COLORS.get(status, "⚪")
    return f"{icon} {status.replace('_', ' ').title()}"


def format_currency(value, currency: str = "USD") -> str:
    """Format a numeric-or-string amount as a currency string. Accepts strings
    (as returned by the API, since amounts are serialized as strings to preserve
    Decimal precision over JSON) as well as numbers."""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)
    sign = "-" if amount < 0 else ""
    return f"{sign}{currency} {abs(amount):,.2f}"


def format_confidence(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def confidence_color(value: float | None, threshold: float = 0.75) -> str:
    if value is None:
        return "⚪"
    if value >= threshold:
        return "🟢"
    if value >= threshold * 0.7:
        return "🟡"
    return "🔴"


def is_terminal_status(status: str) -> bool:
    return status in ("completed", "failed", "needs_human_review")
