"""
Model comparison harness (brief, Section 16: "Compare Qwen-VL vs LLaVA vs OCR
baseline").

HOW TO RUN THIS FOR REAL:

    from evaluation.dataset import build_synthetic_dataset
    from evaluation.model_comparison import run_comparison
    from app.extraction.qwen_vl_model import QwenVLModel
    from app.extraction.llava_model import LLaVAModel
    from app.extraction.ocr_fallback import OCRFallbackModel

    samples = build_synthetic_dataset()  # or load a larger/real dataset (see dataset.py)
    models = {
        "qwen-vl": QwenVLModel(backend="hf-inference"),   # or backend="local" with a GPU
        "llava": LLaVAModel(backend="hf-inference"),
        "ocr_baseline": OCRFallbackModel(),
    }
    report = run_comparison(samples, models)
    print(report.to_markdown())

This module itself never constructs a real VLM — it's given one (or several) via
dependency injection (the `models: dict[str, BaseVisionModel]` parameter), exactly
like every other layer in this project. That's what makes it possible to prove the
FRAMEWORK is correct (metrics computed correctly, reports assembled correctly) using
fake backends in an environment with no GPU/API access, while the exact same code
path works unmodified against real Qwen-VL/LLaVA models wherever those ARE available
— nothing about the harness itself needs to change to go from "verified with a fake"
to "run for real."

On the OCR baseline specifically: pure OCR (raw pixels -> raw text, no structuring)
has no way to produce the structured fields the other metrics compare against, so a
fair three-way comparison necessarily gives the OCR baseline the same text-
structuring step OCRFallbackModel already implements (Tesseract -> text-completion
LLM). This isolates the variable actually being compared — vision-based reading vs.
OCR-based reading — while holding the structuring step constant across all three.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from app.document_processing.pipeline import DocumentPreprocessor
from app.extraction.base_vision_model import BaseVisionModel
from app.extraction.merge import merge_page_results
from app.extraction.page_extractor import PageExtractor
from evaluation.dataset import EvaluationSample
from evaluation.metrics import (
    compute_field_metrics,
    compute_transaction_metrics,
    numeric_field_accuracy,
    schema_validity_rate,
)

DOCUMENT_LEVEL_FIELDS = ["account_holder", "account_number", "bank_name"]
NUMERIC_FIELDS = ["opening_balance", "closing_balance"]


@dataclass
class ModelEvaluationResult:
    model_name: str
    field_metrics: dict
    numeric_accuracy: dict[str, float]
    transaction_metrics: object
    schema_validity: float
    avg_latency_seconds: float
    sample_count: int


@dataclass
class ComparisonReport:
    results: list[ModelEvaluationResult] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# Model Comparison Report", ""]
        lines.append(f"Evaluated on {self.results[0].sample_count if self.results else 0} sample document(s).")
        lines.append("")
        lines.append("| Model | Schema Validity | Txn Precision | Txn Recall | Txn F1 | Avg Latency (s) |")
        lines.append("|---|---|---|---|---|---|")
        for r in self.results:
            tm = r.transaction_metrics
            lines.append(
                f"| {r.model_name} | {r.schema_validity:.0%} | {tm.precision:.2f} | {tm.recall:.2f} | "
                f"{tm.f1:.2f} | {r.avg_latency_seconds:.2f} |"
            )
        lines.append("")
        lines.append("## Field-level metrics")
        for r in self.results:
            lines.append(f"### {r.model_name}")
            lines.append("| Field | Precision | Recall | F1 | Exact Match |")
            lines.append("|---|---|---|---|---|")
            for field_name, m in r.field_metrics.items():
                lines.append(f"| {field_name} | {m.precision:.2f} | {m.recall:.2f} | {m.f1:.2f} | {m.exact_match_rate:.2f} |")
            lines.append("")
            lines.append("| Numeric field | Accuracy (within tolerance) |")
            lines.append("|---|---|")
            for field_name, acc in r.numeric_accuracy.items():
                lines.append(f"| {field_name} | {acc:.0%} |")
            lines.append("")
        return "\n".join(lines)


def _extract_with_model(model: BaseVisionModel, sample: EvaluationSample) -> tuple[dict, bool, float]:
    """Run one model against one sample's PDF and return (raw_merged_dict,
    schema_valid, total_latency_seconds)."""
    preprocessor = DocumentPreprocessor()
    preprocessed = preprocessor.process(sample.pdf_bytes, original_filename=f"{sample.sample_id}.pdf")

    extractor = PageExtractor(model, ocr_fallback_model=None, ocr_enabled=False)
    outcomes = [
        extractor.extract_page(page, total_pages=preprocessed.page_count) for page in preprocessed.pages
    ]
    total_latency = sum(o.vlm_response.latency_seconds for o in outcomes)

    raw_merged = merge_page_results(outcomes)

    schema_valid = True
    try:
        # Best-effort structural check — evaluation cares whether extraction is
        # USABLE (would pass Phase 7's validate_schema), not whether it's perfect;
        # so we do a light shape check rather than constructing the full
        # FinalExtractionResult (which needs fields evaluation doesn't set, like
        # document_id/metadata). Checked with `is None`, not truthiness, since an
        # opening_balance of legitimately 0 is falsy but perfectly valid.
        if raw_merged.get("account_number") is None or raw_merged.get("opening_balance") is None:
            schema_valid = False
    except Exception:
        schema_valid = False

    return raw_merged, schema_valid, total_latency


def run_comparison(samples: list[EvaluationSample], models: dict[str, BaseVisionModel]) -> ComparisonReport:
    report = ComparisonReport()

    for model_name, model in models.items():
        predictions: list[dict] = []
        ground_truths: list[dict] = []
        pred_transactions: list[list[dict]] = []
        gt_transactions: list[list[dict]] = []
        schema_valid_flags: list[bool] = []
        latencies: list[float] = []

        for sample in samples:
            raw_merged, schema_valid, latency = _extract_with_model(model, sample)
            predictions.append(raw_merged)
            ground_truths.append(sample.ground_truth)
            pred_transactions.append(raw_merged.get("transactions", []))
            gt_transactions.append(sample.ground_truth.get("transactions", []))
            schema_valid_flags.append(schema_valid)
            latencies.append(latency)

        field_metrics = compute_field_metrics(predictions, ground_truths, DOCUMENT_LEVEL_FIELDS)
        numeric_accuracy = numeric_field_accuracy(predictions, ground_truths, NUMERIC_FIELDS, Decimal("0.01"))
        transaction_metrics = compute_transaction_metrics(pred_transactions, gt_transactions)
        validity = schema_validity_rate(schema_valid_flags)

        report.results.append(
            ModelEvaluationResult(
                model_name=model_name,
                field_metrics=field_metrics,
                numeric_accuracy=numeric_accuracy,
                transaction_metrics=transaction_metrics,
                schema_validity=validity,
                avg_latency_seconds=sum(latencies) / len(latencies) if latencies else 0.0,
                sample_count=len(samples),
            )
        )

    return report
