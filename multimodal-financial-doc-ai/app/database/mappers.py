"""
Mapping between Pydantic schemas (app.schemas, the API/pipeline-facing shape) and
SQLAlchemy ORM models (app.models.orm, the storage shape).

Kept as its own module rather than inlined in the repository, so the repository's
job stays "talk to the database" and this module's job stays "translate between two
different representations of the same data" — two different kinds of complexity that
are easier to get right (and to unit test) separately.
"""

from __future__ import annotations

from app.database.encryption import FieldEncryptor
from app.models.orm import AccountORM, AnomalyORM, DocumentORM, TransactionORM, ValidationResultORM
from app.schemas.document import Account
from app.schemas.transaction import Transaction
from app.schemas.validation import ValidationIssue, ValidationResult


def account_to_orm(account: Account, *, document_id: str, encryptor: FieldEncryptor) -> AccountORM:
    return AccountORM(
        document_id=document_id,
        account_holder=account.account_holder,
        account_number_encrypted=encryptor.encrypt(account.account_number),
        account_number_last4=account.account_number[-4:] if len(account.account_number) >= 4 else account.account_number,
        bank_name=account.bank_name,
        branch=account.branch,
    )


def account_from_orm(orm: AccountORM, *, encryptor: FieldEncryptor) -> Account:
    return Account(
        account_holder=orm.account_holder,
        account_number=encryptor.decrypt(orm.account_number_encrypted),
        bank_name=orm.bank_name,
        branch=orm.branch,
    )


def transaction_to_orm(txn: Transaction, *, document_id: str) -> TransactionORM:
    return TransactionORM(
        document_id=document_id,
        transaction_date=txn.date,
        description=txn.description,
        reference=txn.reference,
        debit=txn.debit,
        credit=txn.credit,
        balance=txn.balance,
        currency=txn.currency,
        source_page=txn.source_page,
    )


def transaction_from_orm(orm: TransactionORM) -> Transaction:
    return Transaction(
        date=orm.transaction_date,
        description=orm.description,
        reference=orm.reference,
        debit=orm.debit,
        credit=orm.credit,
        balance=orm.balance,
        currency=orm.currency,
        source_page=orm.source_page,
    )


def validation_result_to_orm(result: ValidationResult, *, document_id: str) -> ValidationResultORM:
    return ValidationResultORM(
        document_id=document_id,
        validator_name=result.validator_name,
        status=result.status.value,
        checks_performed=result.checks_performed,
        issues=[issue.model_dump(mode="json") for issue in result.issues],
        recommendation=result.recommendation.value,
    )


def validation_result_from_orm(orm: ValidationResultORM) -> ValidationResult:
    return ValidationResult(
        validator_name=orm.validator_name,
        status=orm.status,
        checks_performed=orm.checks_performed,
        issues=[ValidationIssue(**issue) for issue in orm.issues],
        recommendation=orm.recommendation,
    )


def anomaly_to_orm(anomaly, *, document_id: str) -> AnomalyORM:
    return AnomalyORM(
        document_id=document_id,
        anomaly_type=anomaly.anomaly_type.value,
        severity=anomaly.severity.value,
        message=anomaly.message,
        affected_transaction_indices=anomaly.affected_transaction_indices,
        field=anomaly.field,
    )


def document_to_public_dict(orm: DocumentORM, *, encryptor: FieldEncryptor) -> dict:
    """Build the API-facing dict for a DocumentORM row, with the account number
    masked — this is what GET /documents/{id}/result (Phase 12) returns. Full
    decryption only happens where genuinely needed (e.g. an internal reconciliation
    export), never on the default read path."""
    from app.schemas.parsing import mask_account_number

    account_data = None
    if orm.account is not None:
        full_number = encryptor.decrypt(orm.account.account_number_encrypted)
        account_data = {
            "account_holder": orm.account.account_holder,
            "account_number": mask_account_number(full_number),
            "bank_name": orm.account.bank_name,
            "branch": orm.account.branch,
        }

    return {
        "document_id": orm.id,
        "document_type": orm.document_type,
        "status": orm.status,
        "original_filename": orm.original_filename,
        "page_count": orm.page_count,
        "currency": orm.currency,
        "opening_balance": str(orm.opening_balance) if orm.opening_balance is not None else None,
        "closing_balance": str(orm.closing_balance) if orm.closing_balance is not None else None,
        "statement_period": (
            {"start_date": str(orm.statement_start_date), "end_date": str(orm.statement_end_date)}
            if orm.statement_start_date and orm.statement_end_date
            else None
        ),
        "account": account_data,
        "transactions": [transaction_from_orm(t).model_dump(mode="json") for t in orm.transactions],
        "validation_results": [validation_result_from_orm(v).model_dump(mode="json") for v in orm.validation_results],
        "overall_confidence": orm.overall_confidence,
        "uploaded_at": orm.uploaded_at.isoformat() if orm.uploaded_at else None,
    }
