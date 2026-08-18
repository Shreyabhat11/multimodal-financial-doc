"""
Integration tests: repositories against a real (temp-file SQLite) database,
verifying actual persistence — not in-memory object state — by re-fetching through a
fresh session, and verifying account-number encryption at rest.
"""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy.orm import sessionmaker

from app.database.encryption import FieldEncryptor
from app.database.mappers import document_to_public_dict
from app.database.repositories import DocumentRepository, ProcessingRunRepository
from app.schemas.document import Account, StatementPeriod
from app.schemas.transaction import Transaction
from app.schemas.validation import ValidationResult


class TestDocumentPersistence:
    def test_full_round_trip_across_sessions(self, sqlite_db_url, db_session):
        encryptor = FieldEncryptor("test-secret-key")
        Session = sessionmaker(bind=db_session.get_bind())

        # --- write, in one session ---
        write_session = Session()
        doc_repo = DocumentRepository(write_session, encryptor)
        run_repo = ProcessingRunRepository(write_session)

        doc_repo.create(
            document_id="doc_persist_test", original_filename="s.pdf", document_type="bank_statement",
            page_count=2, file_size_bytes=5000,
        )
        run = run_repo.start_run("doc_persist_test", model_provider="hf-inference")

        account = Account(account_holder="Jane Doe", account_number="9988776655", bank_name="First National")
        period = StatementPeriod(start_date="2024-01-01", end_date="2024-01-31")
        transactions = [
            Transaction(date="2024-01-05", description="Salary", credit="500.00", debit=0, source_page=1),
            Transaction(date="2024-01-10", description="Groceries", debit="200.00", credit=0, source_page=2),
        ]
        doc_repo.save_pipeline_result(
            document_id="doc_persist_test", status="completed", account=account, statement_period=period,
            opening_balance=Decimal("1000.00"), closing_balance=Decimal("1300.00"), currency="USD",
            transactions=transactions, validation_results=[ValidationResult.passed("financial_validation", [])],
            overall_confidence=0.91,
        )
        run_repo.complete_run(run.id, status="completed")
        write_session.commit()
        write_session.close()

        # --- read, in a completely different session ---
        read_session = Session()
        read_repo = DocumentRepository(read_session, encryptor)
        fetched = read_repo.get_by_id("doc_persist_test")

        assert fetched.status == "completed"
        assert len(fetched.transactions) == 2
        assert fetched.account.account_holder == "Jane Doe"

        # encryption at rest -- the raw stored ciphertext must not contain the plaintext
        assert "9988776655" not in fetched.account.account_number_encrypted

        # decryption round-trips correctly
        assert encryptor.decrypt(fetched.account.account_number_encrypted) == "9988776655"

        # public dict masks it
        public = document_to_public_dict(fetched, encryptor=encryptor)
        assert public["account"]["account_number"] == "******6655"

        runs = ProcessingRunRepository(read_session).list_for_document("doc_persist_test")
        assert len(runs) == 1
        assert runs[0].status == "completed"

        read_session.close()

    def test_reprocessing_replaces_transactions_not_appends(self, sqlite_db_url, db_session):
        encryptor = FieldEncryptor("test-secret-key")
        doc_repo = DocumentRepository(db_session, encryptor)

        doc_repo.create(
            document_id="doc_reprocess_test", original_filename="s.pdf", document_type="bank_statement",
            page_count=1, file_size_bytes=1000,
        )
        account = Account(account_holder="Jane Doe", account_number="1111111111")
        period = StatementPeriod(start_date="2024-01-01", end_date="2024-01-31")

        # first save: 2 transactions
        doc_repo.save_pipeline_result(
            document_id="doc_reprocess_test", status="completed", account=account, statement_period=period,
            opening_balance=Decimal("100"), closing_balance=Decimal("100"), currency="USD",
            transactions=[
                Transaction(date="2024-01-05", description="A", debit="1", credit=0),
                Transaction(date="2024-01-06", description="B", debit="1", credit=0),
            ],
            validation_results=[], overall_confidence=0.9,
        )
        db_session.commit()

        # reprocess: only 1 transaction this time
        doc_repo.save_pipeline_result(
            document_id="doc_reprocess_test", status="completed", account=account, statement_period=period,
            opening_balance=Decimal("100"), closing_balance=Decimal("100"), currency="USD",
            transactions=[Transaction(date="2024-01-05", description="A only", debit="1", credit=0)],
            validation_results=[], overall_confidence=0.9,
        )
        db_session.commit()

        fetched = doc_repo.get_by_id("doc_reprocess_test")
        assert len(fetched.transactions) == 1  # replaced, not accumulated to 3
        assert fetched.transactions[0].description == "A only"

    def test_document_not_found_raises(self, db_session):
        from app.core.exceptions import DocumentNotFoundError

        encryptor = FieldEncryptor("test-secret-key")
        doc_repo = DocumentRepository(db_session, encryptor)
        try:
            doc_repo.get_by_id("does-not-exist")
            assert False, "should have raised"
        except DocumentNotFoundError as exc:
            assert exc.error_code == "document_not_found"
