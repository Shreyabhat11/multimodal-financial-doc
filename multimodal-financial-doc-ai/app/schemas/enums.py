"""Shared enums.

Kept in their own module (rather than inline in each schema file) because they're
referenced from schemas, SQLAlchemy models, the LangGraph state, and the CrewAI tools —
having a single import source avoids four slightly different copies of "what counts as
high severity" drifting apart.
"""

from __future__ import annotations

from enum import Enum


class DocumentType(str, Enum):
    BANK_STATEMENT = "bank_statement"
    CREDIT_CARD_STATEMENT = "credit_card_statement"
    INVOICE = "invoice"
    FINANCIAL_REPORT = "financial_report"
    LOAN_STATEMENT = "loan_statement"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Lifecycle of a single uploaded document as it moves through the pipeline."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    VALIDATING = "validating"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationStatus(str, Enum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(str, Enum):
    DUPLICATE_TRANSACTION = "duplicate_transaction"
    UNUSUALLY_LARGE_TRANSACTION = "unusually_large_transaction"
    IMPOSSIBLE_BALANCE = "impossible_balance"
    MISSING_DATE = "missing_date"
    INVALID_DATE = "invalid_date"
    DATE_OUT_OF_STATEMENT_PERIOD = "date_out_of_statement_period"
    DEBIT_CREDIT_INCONSISTENCY = "debit_credit_inconsistency"
    INCONSISTENT_TOTALS = "inconsistent_totals"
    BALANCE_RECONCILIATION_MISMATCH = "balance_reconciliation_mismatch"
    RUNNING_BALANCE_INCONSISTENCY = "running_balance_inconsistency"


class RecommendedAction(str, Enum):
    AUTO_APPROVE = "auto_approve"
    HUMAN_REVIEW = "human_review"
    REPROCESS = "reprocess"
    REJECT = "reject"


class VisionModelProvider(str, Enum):
    QWEN_VL = "qwen-vl"
    LLAVA = "llava"
    HF_INFERENCE = "hf-inference"
    OCR_FALLBACK = "ocr-fallback"
