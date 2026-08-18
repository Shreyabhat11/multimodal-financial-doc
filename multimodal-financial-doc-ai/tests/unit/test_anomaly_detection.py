"""Unit tests for app/validation/{duplicates,dates,balance_consistency,outliers,anomaly_detection}.py."""

from __future__ import annotations

from app.schemas.document import StatementPeriod
from app.schemas.transaction import Transaction
from app.validation.anomaly_detection import run_anomaly_detection
from app.validation.balance_consistency import validate_running_balance_consistency
from app.validation.dates import find_transactions_outside_period
from app.validation.duplicates import find_duplicate_transactions
from app.validation.outliers import find_dual_sided_transactions, find_unusually_large_transactions

PERIOD = StatementPeriod(start_date="2024-01-01", end_date="2024-01-31")


class TestDuplicateDetection:
    def test_detects_exact_duplicate(self):
        txns = [
            Transaction(date="2024-01-05", description="Coffee Shop", debit="4.50", credit=0),
            Transaction(date="2024-01-05", description="Coffee Shop", debit="4.50", credit=0),
        ]
        anomalies = find_duplicate_transactions(txns)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type.value == "duplicate_transaction"
        assert set(anomalies[0].affected_transaction_indices) == {0, 1}

    def test_no_false_positive_on_similar_but_distinct_transactions(self):
        txns = [
            Transaction(date="2024-01-05", description="Coffee Shop", debit="4.50", credit=0),
            Transaction(date="2024-01-06", description="Coffee Shop", debit="4.50", credit=0),  # different date
        ]
        assert find_duplicate_transactions(txns) == []

    def test_groups_three_identical_rows_as_one_anomaly(self):
        txns = [Transaction(date="2024-01-05", description="Same", debit="1.00", credit=0) for _ in range(3)]
        anomalies = find_duplicate_transactions(txns)
        assert len(anomalies) == 1
        assert len(anomalies[0].affected_transaction_indices) == 3


class TestDateValidation:
    def test_detects_transaction_outside_period(self):
        txns = [Transaction(date="2024-02-15", description="Out of range", debit="10.00", credit=0)]
        anomalies = find_transactions_outside_period(txns, PERIOD)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type.value == "date_out_of_statement_period"

    def test_no_anomaly_when_all_dates_in_range(self):
        txns = [Transaction(date="2024-01-15", description="In range", debit="10.00", credit=0)]
        assert find_transactions_outside_period(txns, PERIOD) == []


class TestBalanceConsistency:
    def test_detects_broken_running_balance(self):
        txns = [
            Transaction(date="2024-01-01", description="Opening", credit="1000.00", debit=0, balance="1000.00"),
            Transaction(date="2024-01-05", description="Groceries", debit="200.00", credit=0, balance="800.00"),
            Transaction(date="2024-01-10", description="Rent", debit="300.00", credit=0, balance="999.00"),  # WRONG
        ]
        anomalies = validate_running_balance_consistency(txns)
        assert len(anomalies) == 1
        assert anomalies[0].severity.value == "high"
        assert 2 in anomalies[0].affected_transaction_indices

    def test_consistent_running_balance_produces_no_anomaly(self):
        txns = [
            Transaction(date="2024-01-01", description="Opening", credit="1000.00", debit=0, balance="1000.00"),
            Transaction(date="2024-01-05", description="Groceries", debit="200.00", credit=0, balance="800.00"),
        ]
        assert validate_running_balance_consistency(txns) == []

    def test_sparse_balance_reporting_does_not_false_positive(self):
        """Only some rows report a balance -- should compare against the nearest
        prior anchor, not assume every row must report one."""
        txns = [
            Transaction(date="2024-01-01", description="Opening", credit="1000.00", debit=0, balance="1000.00"),
            Transaction(date="2024-01-03", description="No balance shown", debit="50.00", credit=0, balance=None),
            Transaction(date="2024-01-05", description="Balance shown again", debit="50.00", credit=0, balance="900.00"),
        ]
        assert validate_running_balance_consistency(txns) == []


class TestOutlierDetection:
    def test_detects_large_transaction(self):
        txns = [
            Transaction(date="2024-01-01", description="Coffee", debit="5.00", credit=0),
            Transaction(date="2024-01-02", description="Coffee", debit="4.50", credit=0),
            Transaction(date="2024-01-03", description="Coffee", debit="5.50", credit=0),
            Transaction(date="2024-01-04", description="Coffee", debit="4.00", credit=0),
            Transaction(date="2024-01-05", description="Huge wire", debit="50000.00", credit=0),
        ]
        anomalies = find_unusually_large_transactions(txns, multiplier=5.0)
        assert len(anomalies) == 1
        assert 4 in anomalies[0].affected_transaction_indices

    def test_no_false_positive_on_uniform_amounts(self):
        txns = [
            Transaction(date="2024-01-0" + str(i), description="Coffee", debit=str(4 + i), credit=0)
            for i in range(1, 6)
        ]
        assert find_unusually_large_transactions(txns, multiplier=5.0) == []

    def test_skips_check_with_too_few_transactions(self):
        txns = [Transaction(date="2024-01-01", description="Solo huge txn", debit="99999.00", credit=0)]
        assert find_unusually_large_transactions(txns, multiplier=5.0) == []

    def test_detects_dual_sided_transaction(self):
        txns = [Transaction(date="2024-01-01", description="Odd row", debit="10.00", credit="5.00")]
        anomalies = find_dual_sided_transactions(txns)
        assert len(anomalies) == 1


class TestRunAnomalyDetection:
    def test_clean_statement_passes(self):
        txns = [
            Transaction(date="2024-01-0" + str(i), description=f"Item {i}", debit=str(4 + i), credit=0)
            for i in range(1, 6)
        ]
        result = run_anomaly_detection(transactions=txns, statement_period=PERIOD)
        assert result.status.value == "passed"
        assert result.anomalies == []

    def test_high_severity_anomaly_fails_the_result(self):
        txns = [
            Transaction(date="2024-01-01", description="Opening", credit="1000.00", debit=0, balance="1000.00"),
            Transaction(date="2024-01-05", description="Broken", debit="300.00", credit=0, balance="999.00"),
        ]
        result = run_anomaly_detection(transactions=txns, statement_period=PERIOD)
        assert result.status.value == "failed"
        assert result.recommendation.value == "human_review"

    def test_medium_severity_only_produces_warning_not_failure(self):
        txns = [Transaction(date="2024-02-15", description="Out of range only", debit="10.00", credit=0)] + [
            Transaction(date="2024-01-0" + str(i), description=f"Item {i}", debit=str(4 + i), credit=0)
            for i in range(1, 5)
        ]
        result = run_anomaly_detection(transactions=txns, statement_period=PERIOD)
        assert result.status.value == "passed_with_warnings"
        assert result.recommendation.value == "auto_approve"
