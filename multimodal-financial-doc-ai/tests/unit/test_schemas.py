"""Unit tests for app/schemas/ — parsing, validation, and rejection of bad input."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.document import Account, FinancialTotals, StatementPeriod
from app.schemas.parsing import mask_account_number, parse_amount, parse_flexible_date
from app.schemas.transaction import Transaction


class TestAmountParsing:
    def test_parses_plain_number(self):
        assert parse_amount("1234.56") == Decimal("1234.56")

    def test_parses_comma_separated(self):
        assert parse_amount("1,234.56") == Decimal("1234.56")

    def test_parses_currency_symbol(self):
        assert parse_amount("$1,234.56") == Decimal("1234.56")

    def test_parses_accounting_negative(self):
        assert parse_amount("(50.00)") == Decimal("-50.00")

    def test_parses_numeric_types_directly(self):
        assert parse_amount(1234.56) == Decimal("1234.56")
        assert parse_amount(100) == Decimal("100")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError):
            parse_amount("")

    def test_rejects_none(self):
        with pytest.raises(ValueError):
            parse_amount(None)

    def test_rejects_double_negative(self):
        with pytest.raises(ValueError):
            parse_amount("--50.00")


class TestDateParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2024-12-31", "2024-12-31"),
            ("12/31/2024", "2024-12-31"),
            ("31 Dec 2024", "2024-12-31"),
            ("December 31, 2024", "2024-12-31"),
        ],
    )
    def test_parses_common_formats(self, raw, expected):
        assert str(parse_flexible_date(raw)) == expected

    def test_rejects_unparseable_date(self):
        with pytest.raises(ValueError):
            parse_flexible_date("not a date at all")


class TestMasking:
    def test_masks_all_but_last_four(self):
        assert mask_account_number("9988776655") == "******6655"

    def test_short_number_fully_masked(self):
        assert mask_account_number("123") == "***"


class TestStatementPeriod:
    def test_valid_period(self):
        p = StatementPeriod(start_date="2024-01-01", end_date="2024-01-31")
        assert p.num_days == 30

    def test_rejects_end_before_start(self):
        with pytest.raises(ValidationError):
            StatementPeriod(start_date="2024-02-01", end_date="2024-01-01")


class TestAccount:
    def test_masked_account_number_property(self):
        account = Account(account_holder="Jane Doe", account_number="9988776655", bank_name="Test Bank")
        assert account.masked_account_number == "******6655"

    def test_repr_never_exposes_full_account_number(self):
        account = Account(account_holder="Jane Doe", account_number="9988776655")
        assert "9988776655" not in repr(account)
        assert "******6655" in repr(account)

    def test_rejects_empty_account_number(self):
        with pytest.raises(ValidationError):
            Account(account_holder="Jane Doe", account_number="")


class TestTransaction:
    def test_basic_construction(self):
        txn = Transaction(date="2024-01-05", description="Salary", credit="500.00", debit=0)
        assert txn.net_amount == Decimal("500.00")

    def test_debit_and_credit_stored_as_positive_magnitude(self):
        txn = Transaction(date="2024-01-05", description="Purchase", debit="-50.00", credit=0)
        assert txn.debit == Decimal("50.00")  # sign carried by field, not value

    def test_is_dual_sided_detection(self):
        txn = Transaction(date="2024-01-05", description="Odd row", debit="10.00", credit="5.00")
        assert txn.is_dual_sided is True

    def test_dedup_key_ignores_case_and_whitespace_in_description(self):
        t1 = Transaction(date="2024-01-05", description="  Coffee Shop  ", debit="4.50", credit=0)
        t2 = Transaction(date="2024-01-05", description="coffee shop", debit="4.50", credit=0)
        assert t1.dedup_key == t2.dedup_key

    def test_rejects_missing_description(self):
        with pytest.raises(ValidationError):
            Transaction(date="2024-01-05", description="", debit="1.00", credit=0)


class TestFinancialTotals:
    def test_rejects_negative_totals(self):
        with pytest.raises(ValidationError):
            FinancialTotals(total_debits="-5.00", total_credits="100.00")

    def test_net_change(self):
        totals = FinancialTotals(total_debits="200.00", total_credits="500.00")
        assert totals.net_change == Decimal("300.00")
