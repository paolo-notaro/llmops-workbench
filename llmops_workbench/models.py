"""Shared Pydantic models for LLMOps Workbench."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Action = Literal["answer", "refuse", "abstain"]


class DocumentChunk(BaseModel):
    """A chunk of a synthetic source document."""

    doc_id: str
    title: str
    path: str
    chunk_id: str
    text: str


class RetrievedDocument(DocumentChunk):
    """A retrieved document chunk with a similarity score."""

    score: float = Field(ge=0.0)


class DocumentSummary(BaseModel):
    """Public metadata for one indexed synthetic document."""

    doc_id: str
    title: str
    description: str
    chunk_count: int = Field(ge=1)
    indexed: bool = True


class EvaluationExample(BaseModel):
    """One ground-truth evaluation example."""

    id: str
    category: str
    query: str
    expected_source_docs: list[str] = Field(default_factory=list)
    support_terms: list[str] = Field(default_factory=list)
    expected_action: Action = "answer"
    required_format: str = "answer_with_citations"
    pair_id: str | None = None
    perturbation: str | None = None
    risk_tags: list[str] = Field(default_factory=list)

    # Backward-compatible fields for the original small JSONL examples/tests.
    expected_terms: list[str] = Field(default_factory=list)
    required_citations: bool = True
    unsafe: bool = False
    expected_refusal: bool = False
    expected_format: str = "answer_with_citations"


class LLMResponse(BaseModel):
    """Provider response with measured latency."""

    answer: str
    provider: str
    latency_ms: float = Field(ge=0.0)


class DimensionResult(BaseModel):
    """Result for one evaluation dimension."""

    name: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    details: str
    formula: str | None = None
    threshold: float | None = None


class MetricDefinition(BaseModel):
    """Definition and aggregate value for a workbench quality metric."""

    name: str
    label: str
    value: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    formula: str
    formula_latex: str
    formula_notation: str
    formula_notation_latex: list[str]
    explanation: str
    passed: bool
    sample_count: int = Field(default=0, ge=0)
    population: str = ""


class EvaluationInsight(BaseModel):
    """Actionable interpretation generated from metric outcomes."""

    severity: Literal["low", "medium", "high"]
    dimension: str
    finding: str
    recommendation: str


class EvaluationRecord(BaseModel):
    """Evaluation outcome for one example."""

    example_id: str
    category: str
    query: str
    answer: str
    provider: str
    latency_ms: float
    retrieved_docs: list[RetrievedDocument]
    dimensions: list[DimensionResult]
    passed: bool
    requires_review: bool
    notes: list[str] = Field(default_factory=list)
    expected_action: Action = "answer"
    predicted_action: Action = "answer"
    expected_source_docs: list[str] = Field(default_factory=list)
    cited_docs: list[str] = Field(default_factory=list)
    support_terms: list[str] = Field(default_factory=list)
    pair_id: str | None = None
    perturbation: str | None = None
    risk_tags: list[str] = Field(default_factory=list)
    metric_scores: dict[str, float] = Field(default_factory=dict)


class SummaryMetrics(BaseModel):
    """Aggregate report metrics."""

    total_examples: int
    average_latency_ms: float
    pass_rate: float
    dimension_pass_rates: dict[str, float]
    retrieval_hit_rate: float
    review_count: int
    quality_metrics: list[MetricDefinition] = Field(default_factory=list)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    operational_slo_passed: bool = True
    gates_passed: int = 0
    gates_total: int = 0


class EvaluationReport(BaseModel):
    """Machine-readable evaluation report."""

    timestamp: str
    config: dict[str, Any]
    summary: SummaryMetrics
    records: list[EvaluationRecord]
    insights: list[EvaluationInsight] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Request body for the query API."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=8)


class QueryResponse(BaseModel):
    """Response body for the query API."""

    answer: str
    provider: str
    latency_ms: float
    retrieved_docs: list[RetrievedDocument]
    quality_checks: dict[str, bool] = Field(default_factory=dict)
    live_metrics: dict[str, float] = Field(default_factory=dict)


class QueryTrace(BaseModel):
    """Latest query trace exposed to the Ops console."""

    query: str | None = None
    answer: str | None = None
    provider: str | None = None
    latency_ms: float = 0.0
    retrieved_docs: list[RetrievedDocument] = Field(default_factory=list)
    quality_checks: dict[str, bool] = Field(default_factory=dict)


class DatasetFieldDefinition(BaseModel):
    """Meaning of one human-authored ground-truth field."""

    name: str
    purpose: str


class DatasetExampleSummary(BaseModel):
    """Compact annotated example for methodology inspection."""

    id: str
    query: str
    expected_action: Action
    expected_source_docs: list[str]
    support_terms: list[str]
    pair_id: str | None = None
    perturbation: str | None = None
    risk_tags: list[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    """Versioned description of the synthetic offline benchmark dataset."""

    dataset_id: str
    version: str
    source_path: str
    synthetic: bool = True
    annotation_method: str
    total_examples: int
    pair_count: int
    metric_populations: dict[str, int]
    distributions: dict[str, dict[str, int]]
    fields: list[DatasetFieldDefinition]
    sample_examples: list[DatasetExampleSummary]


class LiveMetricDefinition(BaseModel):
    """Aggregate reference-free signal over recent interactive requests."""

    name: str
    label: str
    value: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    passed: bool | None
    sample_count: int = Field(ge=0)
    explanation: str
    formula: str


class LiveEvaluationRecord(BaseModel):
    """Reference-free evaluation of one interactive request."""

    request_id: str
    timestamp: str
    query: str
    provider: str
    latency_ms: float = Field(ge=0.0)
    retrieved_count: int = Field(ge=0)
    metrics: dict[str, float]
    checks: dict[str, bool]
    requires_review: bool


class LiveMonitoringReport(BaseModel):
    """Rolling live signals kept in memory for the local demo."""

    total_requests: int
    window_size: int
    review_count: int
    pass_rate: float = Field(ge=0.0, le=1.0)
    average_latency_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    quality_metrics: list[LiveMetricDefinition]
    recent_records: list[LiveEvaluationRecord]


class ObservabilityBucket(BaseModel):
    """One non-cumulative latency interval for the Ops visualization."""

    label: str
    upper_bound_ms: float | None = None
    count: int = Field(ge=0)


class ObservabilitySummary(BaseModel):
    """Structured process-local telemetry for the Ops console."""

    process_started_at: str
    uptime_seconds: float = Field(ge=0.0)
    request_count: int = Field(ge=0)
    average_latency_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    review_rate: float = Field(ge=0.0, le=1.0)
    mean_retrieval_confidence: float = Field(ge=0.0, le=1.0)
    refusal_count: int = Field(ge=0)
    latency_buckets: list[ObservabilityBucket]
