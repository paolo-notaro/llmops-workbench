"""Report generation for evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path

from llmops_workbench.models import EvaluationReport


def write_report(report: EvaluationReport, reports_dir: Path) -> tuple[Path, Path]:
    """Write JSON, Markdown, prediction, and aggregate metric artifacts."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "evaluation_report.json"
    markdown_path = reports_dir / "evaluation_report.md"
    predictions_path = reports_dir / "evaluation_predictions.jsonl"
    metrics_path = reports_dir / "evaluation_metrics.json"
    report.artifact_paths.update(
        {
            "report_json": str(json_path),
            "report_markdown": str(markdown_path),
            "predictions_jsonl": str(predictions_path),
            "metrics_json": str(metrics_path),
        }
    )
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    predictions_path.write_text(render_predictions_jsonl(report), encoding="utf-8")
    metrics_path.write_text(json.dumps(render_metrics_payload(report), indent=2), encoding="utf-8")
    return json_path, markdown_path


def render_predictions_jsonl(report: EvaluationReport) -> str:
    """Render one JSON prediction/evaluation record per line."""

    lines = []
    for record in report.records:
        payload = {
            "example_id": record.example_id,
            "category": record.category,
            "query": record.query,
            "expected_action": record.expected_action,
            "predicted_action": record.predicted_action,
            "expected_source_docs": record.expected_source_docs,
            "retrieved_docs": [
                {"doc_id": doc.doc_id, "score": doc.score, "chunk_id": doc.chunk_id}
                for doc in record.retrieved_docs
            ],
            "cited_docs": record.cited_docs,
            "support_terms": record.support_terms,
            "metric_scores": record.metric_scores,
            "requires_review": record.requires_review,
            "notes": record.notes,
            "answer": record.answer,
        }
        lines.append(json.dumps(payload))
    return "\n".join(lines) + "\n"


def render_metrics_payload(report: EvaluationReport) -> dict[str, object]:
    """Render compact aggregate metrics for dashboards and CI gates."""

    return {
        "timestamp": report.timestamp,
        "config": report.config,
        "total_examples": report.summary.total_examples,
        "unweighted_quality_mean": report.summary.pass_rate,
        "gates_passed": report.summary.gates_passed,
        "gates_total": report.summary.gates_total,
        "review_count": report.summary.review_count,
        "latency": {
            "average_ms": report.summary.average_latency_ms,
            "p50_ms": report.summary.latency_p50_ms,
            "p95_ms": report.summary.latency_p95_ms,
            "slo_passed": report.summary.operational_slo_passed,
        },
        "quality_metrics": [metric.model_dump() for metric in report.summary.quality_metrics],
        "insights": [insight.model_dump() for insight in report.insights],
    }


def render_markdown_report(report: EvaluationReport) -> str:
    """Render a human-readable Markdown report."""

    summary = report.summary
    lines = [
        "# LLMOps Evaluation Report",
        "",
        f"Generated: `{report.timestamp}`",
        "",
        "## Configuration",
        "",
        "| Key | Value |",
        "| --- | --- |",
    ]
    for key, value in report.config.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Evaluation Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Total examples | {summary.total_examples} |",
            f"| Quality gates passed | {summary.gates_passed} / {summary.gates_total} |",
            f"| Unweighted quality mean | {summary.pass_rate:.3f} |",
            f"| Average latency ms | {summary.average_latency_ms:.3f} |",
            f"| p95 latency ms | {summary.latency_p95_ms:.3f} |",
            f"| Examples requiring review | {summary.review_count} |",
            "",
            "## Quality Metrics",
            "",
            "| Metric | Value | Threshold | Population | n | Formula |",
            "| --- | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for metric in summary.quality_metrics:
        lines.append(
            f"| {metric.label} | {metric.value:.3f} | {metric.threshold:.3f} | "
            f"{metric.population} | {metric.sample_count} | {metric.formula} |"
        )
    lines.extend(["", "## Insights", ""])
    for insight in report.insights:
        lines.append(f"- **{insight.severity.upper()} / {insight.dimension}:** {insight.finding} Recommendation: {insight.recommendation}")
    lines.extend(["", "## Top Examples Requiring Review", ""])
    failures = [record for record in report.records if record.requires_review][:8]
    if failures:
        for record in failures:
            lines.append(
                f"- `{record.example_id}` ({record.category}): expected `{record.expected_action}`, "
                f"predicted `{record.predicted_action}`; {'; '.join(record.notes)}"
            )
    else:
        lines.append("No examples required review in this run.")
    lines.extend(["", "## Sample Predictions", ""])
    for record in report.records[:5]:
        status = "PASS" if not record.requires_review else "REVIEW"
        lines.extend(
            [
                f"### {record.example_id} - {status}",
                "",
                f"Query: {record.query}",
                "",
                f"Expected action: `{record.expected_action}`; predicted action: `{record.predicted_action}`",
                "",
                f"Metrics: `{record.metric_scores}`",
                "",
                f"Answer: {record.answer}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "The metrics are transparent local heuristics over synthetic data. They are designed to make evaluation behavior inspectable: source-set retrieval, citation evidence, answer grounding, policy routing, output contract adherence, and paired-prompt robustness. Scores below threshold are intentional review signals, not production claims.",
        ]
    )
    return "\n".join(lines) + "\n"
