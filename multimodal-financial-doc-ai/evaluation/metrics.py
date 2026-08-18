"""
Evaluation metrics (brief, Section 16).

Every metric here operates on plain dicts, not Pydantic schemas — evaluation needs
to score PARTIAL/malformed extraction output too (a model that got the JSON shape
wrong shouldn't crash the evaluator, it should just score badly), so metrics work at
the same "raw dict" level as merge.py's output rather than requiring a fully
schema-valid FinalExtractionResult to even begin scoring.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


def _to_decimal(value) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _values_equal(predicted, expected) -> bool:
    """Exact-match comparison with light normalization: string-trim and case-fold
    for strings, so 'Jane Doe' vs 'jane doe ' isn't scored as a miss over
    formatting noise that has nothing to do with extraction correctness."""
    if predicted is None and expected is None:
        return True
    if predicted is None or expected is None:
        return False
    if isinstance(predicted, str) and isinstance(expected, str):
        return predicted.strip().lower() == expected.strip().lower()
    return predicted == expected


@dataclass
class FieldMetric:
    field: str
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0  # both predicted and expected are None/absent — correct by omission

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def exact_match_rate(self) -> float:
        total = self.true_positive + self.false_positive + self.false_negative + self.true_negative
        correct = self.true_positive + self.true_negative
        return correct / total if total else 1.0


def compute_field_metrics(
    predictions: list[dict],
    ground_truths: list[dict],
    fields: list[str],
) -> dict[str, FieldMetric]:
    """Compute per-field precision/recall/F1/exact-match across a dataset of
    (prediction, ground_truth) document pairs.

    TP: field present in both and values match.
    FP: field present in prediction but wrong value, OR present when ground truth has none.
    FN: field missing/wrong in prediction when ground truth has a value.
    TN: both prediction and ground truth agree the field is absent (correct omission).
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must be the same length (one pair per document)")

    metrics = {field: FieldMetric(field=field) for field in fields}

    for pred, truth in zip(predictions, ground_truths):
        for field in fields:
            p_val = pred.get(field)
            t_val = truth.get(field)
            m = metrics[field]

            if t_val is None and p_val is None:
                m.true_negative += 1
            elif t_val is not None and p_val is not None and _values_equal(p_val, t_val):
                m.true_positive += 1
            elif t_val is not None and (p_val is None or not _values_equal(p_val, t_val)):
                m.false_negative += 1
                if p_val is not None:
                    m.false_positive += 1  # a wrong (non-null) value is also a false positive
            elif t_val is None and p_val is not None:
                m.false_positive += 1

    return metrics


def numeric_match_within_tolerance(predicted, expected, tolerance: Decimal = Decimal("0.01")) -> bool:
    """Tolerance-based match for numeric fields (brief, Section 16: 'Use
    tolerance-based accuracy' for numeric fields) — an extraction of 1300.00 against
    a ground truth of 1300.005 (a rounding artifact in how the ground truth itself
    was authored) shouldn't be scored as wrong."""
    p = _to_decimal(predicted)
    e = _to_decimal(expected)
    if p is None or e is None:
        return p == e  # both None -> True; one None -> False
    return abs(p - e) <= tolerance


def numeric_field_accuracy(
    predictions: list[dict],
    ground_truths: list[dict],
    fields: list[str],
    tolerance: Decimal = Decimal("0.01"),
) -> dict[str, float]:
    """Accuracy (fraction correct within tolerance) per numeric field across the dataset."""
    accuracy: dict[str, float] = {}
    for field in fields:
        correct = 0
        for pred, truth in zip(predictions, ground_truths):
            if numeric_match_within_tolerance(pred.get(field), truth.get(field), tolerance):
                correct += 1
        accuracy[field] = correct / len(predictions) if predictions else 1.0
    return accuracy


def _transaction_similarity(a: dict, b: dict) -> float:
    """Similarity score in [0, 1] used only as a matching TIE-BREAKER, never as the
    primary match criterion (date + amount agreement is) — description text from a
    VLM/OCR read can have minor wording differences from the ground truth even for a
    genuinely correct match (e.g. extra whitespace, a truncated merchant name)."""
    return difflib.SequenceMatcher(None, str(a.get("description", "")).lower(), str(b.get("description", "")).lower()).ratio()


def match_transactions(
    predicted: list[dict],
    ground_truth: list[dict],
    *,
    amount_tolerance: Decimal = Decimal("0.01"),
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy bipartite matching between predicted and ground-truth transactions.

    A candidate pair is eligible to match only if: same date, AND (debit matches
    within tolerance AND credit matches within tolerance). Among eligible candidates
    for a given ground-truth transaction, the one with the highest description
    similarity is chosen — this handles the common case of two same-day,
    same-amount transactions (e.g. two identical coffee purchases) where date+amount
    alone can't disambiguate, without letting description similarity alone create a
    false match between transactions that don't actually agree on date/amount.

    Returns (matched_pairs, unmatched_predicted_indices, unmatched_ground_truth_indices).
    """
    unmatched_gt = set(range(len(ground_truth)))
    unmatched_pred = set(range(len(predicted)))
    matched_pairs: list[tuple[int, int]] = []

    for gt_idx, gt_txn in enumerate(ground_truth):
        best_pred_idx = None
        best_score = -1.0
        for pred_idx in unmatched_pred:
            pred_txn = predicted[pred_idx]
            if str(pred_txn.get("date")) != str(gt_txn.get("date")):
                continue
            if not numeric_match_within_tolerance(pred_txn.get("debit", 0), gt_txn.get("debit", 0), amount_tolerance):
                continue
            if not numeric_match_within_tolerance(pred_txn.get("credit", 0), gt_txn.get("credit", 0), amount_tolerance):
                continue
            score = _transaction_similarity(pred_txn, gt_txn)
            if score > best_score:
                best_score = score
                best_pred_idx = pred_idx

        if best_pred_idx is not None:
            matched_pairs.append((best_pred_idx, gt_idx))
            unmatched_pred.discard(best_pred_idx)
            unmatched_gt.discard(gt_idx)

    return matched_pairs, sorted(unmatched_pred), sorted(unmatched_gt)


@dataclass
class TransactionMetrics:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def compute_transaction_metrics(
    predictions: list[list[dict]],
    ground_truths: list[list[dict]],
    *,
    amount_tolerance: Decimal = Decimal("0.01"),
) -> TransactionMetrics:
    """Transaction-level precision/recall/F1 (brief, Section 16), aggregated across
    an entire dataset of documents (not per-document averages — pooling raw
    TP/FP/FN counts across all documents first, then computing P/R/F1 once, so a
    50-transaction document isn't diluted to the same weight as a 2-transaction one)."""
    total_tp = total_fp = total_fn = 0
    for pred_txns, gt_txns in zip(predictions, ground_truths):
        matched, unmatched_pred, unmatched_gt = match_transactions(pred_txns, gt_txns, amount_tolerance=amount_tolerance)
        total_tp += len(matched)
        total_fp += len(unmatched_pred)
        total_fn += len(unmatched_gt)
    return TransactionMetrics(true_positive=total_tp, false_positive=total_fp, false_negative=total_fn)


def schema_validity_rate(parse_results: list[bool]) -> float:
    """Fraction of documents whose raw extraction output successfully parsed into
    valid Pydantic schema objects (brief, Section 16: 'Measure schema validity').
    Callers pass a list of booleans (one per document) — True if
    FinalExtractionResult construction succeeded, False if it raised."""
    if not parse_results:
        return 1.0
    return sum(parse_results) / len(parse_results)
