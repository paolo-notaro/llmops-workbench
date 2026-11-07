"""Run the local LLMOps evaluation demo."""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmops_workbench.config import load_settings
from llmops_workbench.dataset import build_dataset_profile, load_evaluation_examples
from llmops_workbench.evaluators import evaluate_examples
from llmops_workbench.providers import provider_from_env
from llmops_workbench.rag import LocalTfidfRAGIndex
from llmops_workbench.report import write_report


def main() -> None:
    """Run the complete local evaluation workflow."""

    console = Console()
    settings = load_settings()
    index = LocalTfidfRAGIndex.from_directory(settings.docs_dir)
    provider = provider_from_env(settings.llm_provider)
    examples = load_evaluation_examples(settings.eval_dir)
    profile = build_dataset_profile(examples, settings.eval_dir)
    report = evaluate_examples(
        examples,
        index,
        provider,
        top_k=3,
        max_latency_ms=settings.max_latency_ms,
        config={
            "provider": provider.name,
            "retriever": "tfidf",
            "dataset_id": profile.dataset_id,
            "dataset_version": profile.version,
            "top_k": 3,
            "max_latency_ms": settings.max_latency_ms,
            "mode": "offline_snapshot",
        },
        dataset_id=profile.dataset_id,
        dataset_version=profile.version,
    )
    json_path, markdown_path = write_report(report, settings.reports_dir)
    print_summary(console, report)
    console.print(f"\nReports written: {json_path} and {markdown_path}")
    console.print(f"Predictions: {settings.reports_dir / 'evaluation_predictions.jsonl'}")
    console.print(f"Metrics: {settings.reports_dir / 'evaluation_metrics.json'}")


def print_summary(console: Console, report) -> None:
    """Print a concise terminal summary."""

    table = Table(title="LLMOps Evaluation Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Threshold", justify="right")
    for metric in report.summary.quality_metrics:
        table.add_row(metric.label, f"{metric.value:.3f}", f"{metric.threshold:.3f}")
    console.print(table)
    console.print(f"Total examples: {report.summary.total_examples}")
    console.print(f"Quality gates: {report.summary.gates_passed}/{report.summary.gates_total}")
    console.print(f"Average latency: {report.summary.average_latency_ms:.3f} ms")
    console.print(f"p95 latency: {report.summary.latency_p95_ms:.3f} ms")
    console.print(f"Examples requiring review: {report.summary.review_count}")


if __name__ == "__main__":
    main()
