"""Process-local metrics and Prometheus exposition for the demo API."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from llmops_portfolio.models import ObservabilityBucket, ObservabilitySummary


LATENCY_BOUNDS_MS = (0.05, 0.10, 0.25, 1.0, 10.0, 100.0, 500.0)


@dataclass
class MetricsRegistry:
    """Collect bounded request and quality telemetry."""

    process_started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    process_started_monotonic: float = field(default_factory=time.monotonic)
    request_count: int = 0
    latency_ms_total: float = 0.0
    evaluation_pass_count: int = 0
    retrieval_hit_count: int = 0
    review_count: int = 0
    refusal_count: int = 0
    retrieval_confidence_total: float = 0.0
    latency_values_ms: list[float] = field(default_factory=list)
    latency_bucket_counts: dict[float, int] = field(
        default_factory=lambda: {bound: 0 for bound in LATENCY_BOUNDS_MS}
    )

    def record_request(
        self,
        latency_ms: float,
        *,
        evaluation_passed: bool,
        retrieval_hit: bool,
        requires_review: bool = False,
        refusal: bool = False,
        retrieval_confidence: float = 0.0,
    ) -> None:
        """Record metrics for one interactive request."""

        self.request_count += 1
        self.latency_ms_total += latency_ms
        self.evaluation_pass_count += int(evaluation_passed)
        self.retrieval_hit_count += int(retrieval_hit)
        self.review_count += int(requires_review)
        self.refusal_count += int(refusal)
        self.retrieval_confidence_total += retrieval_confidence
        for bound in LATENCY_BOUNDS_MS:
            self.latency_bucket_counts[bound] += int(latency_ms <= bound)
        self.latency_values_ms.append(latency_ms)
        if len(self.latency_values_ms) > 1000:
            self.latency_values_ms = self.latency_values_ms[-1000:]

    @property
    def average_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.latency_ms_total / self.request_count

    @property
    def evaluation_pass_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.evaluation_pass_count / self.request_count

    @property
    def retrieval_hit_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.retrieval_hit_count / self.request_count

    @property
    def review_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.review_count / self.request_count

    @property
    def mean_retrieval_confidence(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.retrieval_confidence_total / self.request_count

    def summary(self) -> ObservabilitySummary:
        """Return structured telemetry for the human-facing console."""

        values = sorted(self.latency_values_ms)
        previous = float("-inf")
        buckets: list[ObservabilityBucket] = []
        for bound in LATENCY_BOUNDS_MS:
            count = sum(previous < value <= bound for value in values)
            buckets.append(ObservabilityBucket(label=f"<= {bound:g} ms", upper_bound_ms=bound, count=count))
            previous = bound
        buckets.append(ObservabilityBucket(
            label=f"> {LATENCY_BOUNDS_MS[-1]:g} ms",
            upper_bound_ms=None,
            count=sum(value > LATENCY_BOUNDS_MS[-1] for value in values),
        ))
        return ObservabilitySummary(
            process_started_at=self.process_started_at.isoformat(),
            uptime_seconds=round(time.monotonic() - self.process_started_monotonic, 3),
            request_count=self.request_count,
            average_latency_ms=round(self.average_latency_ms, 3),
            latency_p95_ms=round(_percentile(values, 0.95), 3),
            pass_rate=round(self.evaluation_pass_rate, 3),
            review_rate=round(self.review_rate, 3),
            mean_retrieval_confidence=round(self.mean_retrieval_confidence, 3),
            refusal_count=self.refusal_count,
            latency_buckets=buckets,
        )

    def render_prometheus(self) -> str:
        """Render valid Prometheus counters, gauges, and a cumulative histogram."""

        lines = [
            "# HELP llmops_query_requests_total Total interactive query requests",
            "# TYPE llmops_query_requests_total counter",
            f"llmops_query_requests_total {self.request_count}",
            "# HELP llmops_query_review_total Interactive requests requiring review",
            "# TYPE llmops_query_review_total counter",
            f"llmops_query_review_total {self.review_count}",
            "# HELP llmops_query_refusal_total Interactive requests routed to refusal",
            "# TYPE llmops_query_refusal_total counter",
            f"llmops_query_refusal_total {self.refusal_count}",
            "# HELP llmops_live_check_pass_ratio Fraction of requests passing all live checks",
            "# TYPE llmops_live_check_pass_ratio gauge",
            f"llmops_live_check_pass_ratio {self.evaluation_pass_rate:.3f}",
            "# HELP llmops_retrieval_hit_ratio Fraction of requests with a non-zero retrieval hit",
            "# TYPE llmops_retrieval_hit_ratio gauge",
            f"llmops_retrieval_hit_ratio {self.retrieval_hit_rate:.3f}",
            "# HELP llmops_retrieval_confidence_mean Mean reference-free retrieval confidence",
            "# TYPE llmops_retrieval_confidence_mean gauge",
            f"llmops_retrieval_confidence_mean {self.mean_retrieval_confidence:.3f}",
            "# HELP llmops_query_latency_ms Interactive provider latency in milliseconds",
            "# TYPE llmops_query_latency_ms histogram",
        ]
        for bound in LATENCY_BOUNDS_MS:
            cumulative = self.latency_bucket_counts[bound]
            lines.append(f'llmops_query_latency_ms_bucket{{le="{bound:g}"}} {cumulative}')
        lines.extend([
            f'llmops_query_latency_ms_bucket{{le="+Inf"}} {self.request_count}',
            f"llmops_query_latency_ms_sum {self.latency_ms_total:.6f}",
            f"llmops_query_latency_ms_count {self.request_count}",
            "# HELP llmops_process_uptime_seconds Uptime of the local API process",
            "# TYPE llmops_process_uptime_seconds gauge",
            f"llmops_process_uptime_seconds {time.monotonic() - self.process_started_monotonic:.3f}",
        ])
        return "\n".join(lines) + "\n"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * percentile)))
    return values[index]


metrics_registry = MetricsRegistry()
