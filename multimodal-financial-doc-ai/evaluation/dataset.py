"""
Synthetic evaluation dataset.

Real bank/credit-card statements can't be freely redistributed (privacy, licensing),
so this generates SYNTHETIC PDFs with known ground truth — the honest alternative
the brief explicitly allows ("create a small synthetic/sample financial document
dataset"). Each sample is a real PDF (via PyMuPDF, same rendering path
document_processing uses) paired with the exact structured JSON that should be
extracted from it, since we authored both.

This is intentionally a small (3-sample) dataset for demonstrating the framework
works end to end — a real evaluation run should use dozens-to-hundreds of documents,
ideally including real (de-identified) statements, for statistically meaningful
numbers. See evaluation/model_comparison.py's docstring for how to point this
framework at a larger/real dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


@dataclass
class EvaluationSample:
    sample_id: str
    pdf_bytes: bytes
    ground_truth: dict  # shaped like the raw merged-extraction dict (app.extraction.merge output)


def _render_statement_pdf(
    *,
    account_holder: str,
    account_number: str,
    bank_name: str,
    opening_balance: float,
    transactions: list[dict],
    rotate_page: bool = False,
) -> bytes:
    """Render a synthetic bank statement PDF with known content, using the same
    PyMuPDF primitives app.document_processing.pdf_loader consumes on the way back
    in — so a sample generated here is a realistic input to the real pipeline, not a
    format the pipeline has never seen."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    y = 72
    page.insert_text((72, y), f"Statement — {bank_name}", fontsize=16)
    y += 30
    page.insert_text((72, y), f"Account Holder: {account_holder}", fontsize=11)
    y += 18
    page.insert_text((72, y), f"Account Number: {account_number}", fontsize=11)
    y += 18
    page.insert_text((72, y), f"Opening Balance: {opening_balance:.2f}", fontsize=11)
    y += 30
    page.insert_text((72, y), "Date        Description                 Debit     Credit", fontsize=10)
    y += 16
    for txn in transactions:
        debit = f"{txn['debit']:.2f}" if txn.get("debit") else ""
        credit = f"{txn['credit']:.2f}" if txn.get("credit") else ""
        page.insert_text((72, y), f"{txn['date']}  {txn['description']:<24} {debit:>8} {credit:>8}", fontsize=9)
        y += 14

    if rotate_page:
        page.set_rotation(90)

    buffer = doc.tobytes()
    doc.close()
    return buffer


def build_synthetic_dataset() -> list[EvaluationSample]:
    """Build the 3-sample synthetic evaluation set: a clean statement, a statement
    with a rotated page, and a statement with a deliberately larger transaction
    count (to exercise the transaction-matching metric more thoroughly)."""

    samples: list[EvaluationSample] = []

    # --- Sample 1: clean, straightforward ---
    txns_1 = [
        {"date": "2024-01-05", "description": "Salary", "debit": 0, "credit": 500.00},
        {"date": "2024-01-10", "description": "Groceries", "debit": 200.00, "credit": 0},
    ]
    pdf_1 = _render_statement_pdf(
        account_holder="Jane Doe", account_number="9988776655", bank_name="First National",
        opening_balance=1000.00, transactions=txns_1,
    )
    samples.append(
        EvaluationSample(
            sample_id="clean_statement",
            pdf_bytes=pdf_1,
            ground_truth={
                "account_holder": "Jane Doe", "account_number": "9988776655", "bank_name": "First National",
                "opening_balance": 1000.00, "closing_balance": 1300.00, "currency": "USD",
                "statement_period": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
                "transactions": txns_1,
            },
        )
    )

    # --- Sample 2: rotated page ---
    txns_2 = [
        {"date": "2024-02-03", "description": "Freelance Payment", "debit": 0, "credit": 750.00},
        {"date": "2024-02-14", "description": "Rent", "debit": 1200.00, "credit": 0},
        {"date": "2024-02-20", "description": "Utilities", "debit": 85.50, "credit": 0},
    ]
    pdf_2 = _render_statement_pdf(
        account_holder="Bob Smith", account_number="1122334455", bank_name="Second Trust",
        opening_balance=2000.00, transactions=txns_2, rotate_page=True,
    )
    samples.append(
        EvaluationSample(
            sample_id="rotated_page_statement",
            pdf_bytes=pdf_2,
            ground_truth={
                "account_holder": "Bob Smith", "account_number": "1122334455", "bank_name": "Second Trust",
                "opening_balance": 2000.00, "closing_balance": 1464.50, "currency": "USD",
                "statement_period": {"start_date": "2024-02-01", "end_date": "2024-02-29"},
                "transactions": txns_2,
            },
        )
    )

    # --- Sample 3: larger transaction volume ---
    txns_3 = [{"date": f"2024-03-{i:02d}", "description": f"Purchase {i}", "debit": 10.00 + i, "credit": 0} for i in range(1, 16)]
    txns_3.append({"date": "2024-03-28", "description": "Paycheck", "debit": 0, "credit": 2500.00})
    total_debits_3 = sum(t["debit"] for t in txns_3)
    total_credits_3 = sum(t["credit"] for t in txns_3)
    pdf_3 = _render_statement_pdf(
        account_holder="Carla Nguyen", account_number="6677889900", bank_name="Third Community Bank",
        opening_balance=500.00, transactions=txns_3,
    )
    samples.append(
        EvaluationSample(
            sample_id="high_volume_statement",
            pdf_bytes=pdf_3,
            ground_truth={
                "account_holder": "Carla Nguyen", "account_number": "6677889900", "bank_name": "Third Community Bank",
                "opening_balance": 500.00, "closing_balance": round(500.00 - total_debits_3 + total_credits_3, 2),
                "currency": "USD",
                "statement_period": {"start_date": "2024-03-01", "end_date": "2024-03-31"},
                "transactions": txns_3,
            },
        )
    )

    return samples


def save_dataset_to_disk(samples: list[EvaluationSample]) -> None:
    """Persist the generated PDFs and ground truth to data/samples/, so they can be
    inspected directly or reused without regenerating them (e.g. `git diff` on the
    ground truth JSON when the dataset changes)."""
    import json

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        (SAMPLES_DIR / f"{sample.sample_id}.pdf").write_bytes(sample.pdf_bytes)
        (SAMPLES_DIR / f"{sample.sample_id}.ground_truth.json").write_text(json.dumps(sample.ground_truth, indent=2))
