"""FastAPI application for the local LLMOps demo."""

from __future__ import annotations

import json
from collections import deque
from datetime import UTC, datetime
from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from llmops_workbench.config import REPO_ROOT, Settings, load_settings
from llmops_workbench.dataset import build_dataset_profile, load_evaluation_examples
from llmops_workbench.evaluators import evaluate_examples
from llmops_workbench.execution import execute_request
from llmops_workbench.live_evaluation import evaluate_live_trace, summarize_live_requests
from llmops_workbench.models import (
    DatasetProfile,
    DocumentSummary,
    EvaluationExample,
    EvaluationReport,
    LiveEvaluationRecord,
    LiveMonitoringReport,
    ObservabilitySummary,
    QueryRequest,
    QueryResponse,
)
from llmops_workbench.observability import metrics_registry
from llmops_workbench.providers import LLMProvider, provider_from_env
from llmops_workbench.rag import LocalTfidfRAGIndex
from llmops_workbench.security import PublicDemoSecurityMiddleware


FRONTEND_DIR = REPO_ROOT / "frontend"
DOCS_DIR = REPO_ROOT / "docs"
LIVE_WINDOW_LIMIT = 50

app = FastAPI(
    title="LLMOps Workbench API",
    version="0.2.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(PublicDemoSecurityMiddleware)
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")
app.mount("/workbench-docs", StaticFiles(directory=DOCS_DIR), name="workbench-docs")

_live_records: deque[LiveEvaluationRecord] = deque(maxlen=LIVE_WINDOW_LIMIT)
_live_total_requests = 0
_offline_report: EvaluationReport | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once for the API process."""

    return load_settings()


@lru_cache(maxsize=1)
def get_index() -> LocalTfidfRAGIndex:
    """Build the local retrieval index once."""

    return LocalTfidfRAGIndex.from_directory(get_settings().docs_dir)


@lru_cache(maxsize=1)
def get_provider() -> LLMProvider:
    """Create the configured provider once."""

    return provider_from_env(get_settings().llm_provider)


@lru_cache(maxsize=1)
def get_evaluation_examples() -> tuple[EvaluationExample, ...]:
    """Load local JSONL evaluation examples."""

    return tuple(load_evaluation_examples(get_settings().eval_dir))


@lru_cache(maxsize=1)
def get_dataset_profile() -> DatasetProfile:
    """Return stable metadata for the annotated synthetic benchmark."""

    return build_dataset_profile(list(get_evaluation_examples()), get_settings().eval_dir)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the workbench surface selector."""

    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/app", include_in_schema=False)
def customer_app() -> FileResponse:
    """Serve the customer-facing RAG assistant."""

    return FileResponse(FRONTEND_DIR / "app.html")


@app.get("/ops", include_in_schema=False)
def ops_console() -> FileResponse:
    """Serve the LLMOps / DevOps console."""

    return FileResponse(FRONTEND_DIR / "ops.html")


@app.get("/case-studies", include_in_schema=False)
def case_studies() -> FileResponse:
    """Serve the rendered case-study index."""

    return FileResponse(FRONTEND_DIR / "case-studies.html")


@app.get("/case-studies/{slug}", include_in_schema=False)
def case_study(slug: str) -> FileResponse:
    """Serve the rendered case-study detail shell."""

    return FileResponse(FRONTEND_DIR / "case-study.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Avoid noisy favicon 404s in browser demos."""

    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a dependency-free health signal for deployment probes."""

    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentSummary])
def documents() -> list[DocumentSummary]:
    """Return metadata for the synthetic documents in the local index."""

    return get_index().document_summaries


@app.get("/evaluation/dataset", response_model=DatasetProfile)
def evaluation_dataset() -> DatasetProfile:
    """Describe the annotations and populations behind offline metrics."""

    return get_dataset_profile()


@app.get("/evaluation/offline", response_model=EvaluationReport)
def offline_benchmark() -> EvaluationReport:
    """Return the stable offline benchmark snapshot."""

    return _get_offline_report()


@app.get("/evaluation/report", response_model=EvaluationReport)
def evaluation_report() -> EvaluationReport:
    """Compatibility alias for the stable offline benchmark."""

    return _get_offline_report()


@app.get("/evaluation/live", response_model=LiveMonitoringReport)
def live_monitoring() -> LiveMonitoringReport:
    """Return rolling reference-free signals for interactive requests."""

    return summarize_live_requests(
        list(_live_records),
        _live_total_requests,
        window_limit=LIVE_WINDOW_LIMIT,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Retrieve context, generate an answer, and record live quality proxies."""

    global _live_total_requests
    _live_total_requests += 1
    trace = execute_request(
        request.query,
        get_index(),
        get_provider(),
        mode="live",
        top_k=request.top_k,
        trace_id=f"req-{_live_total_requests:04d}",
    )
    live_record = evaluate_live_trace(trace)
    _live_records.append(live_record)
    quality_checks = {
        **live_record.checks,
        "latency_under_threshold": trace.timings.generation_ms <= get_settings().max_latency_ms,
    }
    metrics_registry.record_request(
        trace.timings.generation_ms,
        evaluation_passed=all(quality_checks.values()),
        retrieval_hit=live_record.retrieved_count > 0,
        requires_review=live_record.requires_review,
        refusal="cannot help" in trace.answer.lower(),
        retrieval_confidence=live_record.metrics["retrieval_confidence"],
    )
    return QueryResponse(
        answer=trace.answer,
        provider=trace.provider,
        latency_ms=trace.timings.generation_ms,
        retrieved_docs=trace.retrieval.documents(),
        quality_checks=quality_checks,
        live_metrics=live_record.metrics,
        trace_id=trace.trace_id,
        config_id=trace.config_id,
    )


@app.get("/observability/summary", response_model=ObservabilitySummary)
def observability_summary() -> ObservabilitySummary:
    """Return structured process-local telemetry for the Ops console."""

    return metrics_registry.summary()


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Return Prometheus exposition for runtime and offline quality signals."""

    profile = get_dataset_profile()
    report = _get_offline_report()
    lines = [metrics_registry.render_prometheus().rstrip(), ""]
    lines.extend([
        "# HELP llmops_dataset_info Version and size of the annotated synthetic dataset",
        "# TYPE llmops_dataset_info gauge",
        f'llmops_dataset_info{{dataset_id="{profile.dataset_id}",version="{profile.version}",examples="{profile.total_examples}"}} 1',
        "# HELP llmops_offline_metric_score Latest versioned offline metric score",
        "# TYPE llmops_offline_metric_score gauge",
    ])
    for metric in report.summary.quality_metrics:
        lines.append(
            f'llmops_offline_metric_score{{metric="{metric.name}",version="{profile.version}"}} {metric.value:.3f}'
        )
    lines.extend([
        "# HELP llmops_offline_gate_pass Latest versioned offline gate outcome",
        "# TYPE llmops_offline_gate_pass gauge",
    ])
    for metric in report.summary.quality_metrics:
        lines.append(
            f'llmops_offline_gate_pass{{metric="{metric.name}",version="{profile.version}"}} {int(metric.passed)}'
        )
    return "\n".join(lines) + "\n"


def _get_offline_report() -> EvaluationReport:
    global _offline_report
    if _offline_report is not None:
        return _offline_report
    stored_path = get_settings().reports_dir / "evaluation_report.json"
    if stored_path.exists():
        try:
            candidate = EvaluationReport.model_validate(json.loads(stored_path.read_text(encoding="utf-8")))
            if (
                candidate.config.get("mode") == "offline_snapshot"
                and candidate.config.get("dataset_version") == get_dataset_profile().version
                and all(record.trace is not None for record in candidate.records)
            ):
                _offline_report = candidate
                return _offline_report
        except (json.JSONDecodeError, ValueError):
            pass
    _offline_report = _build_offline_report()
    return _offline_report


def _build_offline_report() -> EvaluationReport:
    settings = get_settings()
    provider = get_provider()
    profile = get_dataset_profile()
    report = evaluate_examples(
        list(get_evaluation_examples()),
        get_index(),
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
    return report
