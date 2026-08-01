from pathlib import Path

from llmops_workbench.app import app
from llmops_workbench.case_studies import parse_case_study
from llmops_workbench.observability import MetricsRegistry


def test_observability_summary_uses_calibrated_latency_intervals() -> None:
    registry = MetricsRegistry()
    for latency, review, refusal, confidence in [
        (0.03, False, False, 0.9),
        (0.08, False, False, 0.8),
        (0.20, True, True, 0.4),
    ]:
        registry.record_request(
            latency,
            evaluation_passed=not review,
            retrieval_hit=True,
            requires_review=review,
            refusal=refusal,
            retrieval_confidence=confidence,
        )

    summary = registry.summary()

    assert summary.request_count == 3
    assert summary.review_rate == 0.333
    assert summary.refusal_count == 1
    assert summary.mean_retrieval_confidence == 0.7
    assert [bucket.count for bucket in summary.latency_buckets[:3]] == [1, 1, 1]


def test_prometheus_histogram_is_cumulative_and_bounded() -> None:
    registry = MetricsRegistry()
    registry.record_request(
        0.08,
        evaluation_passed=True,
        retrieval_hit=True,
        retrieval_confidence=0.8,
    )

    metrics = registry.render_prometheus()

    assert 'llmops_query_latency_ms_bucket{le="0.05"} 0' in metrics
    assert 'llmops_query_latency_ms_bucket{le="0.1"} 1' in metrics
    assert 'llmops_query_latency_ms_bucket{le="+Inf"} 1' in metrics
    assert "query=" not in metrics


def test_prometheus_histogram_remains_cumulative_beyond_sample_window() -> None:
    registry = MetricsRegistry()
    for _ in range(1001):
        registry.record_request(0.08, evaluation_passed=True, retrieval_hit=True)

    metrics = registry.render_prometheus()

    assert 'llmops_query_latency_ms_bucket{le="0.1"} 1001' in metrics
    assert 'llmops_query_latency_ms_bucket{le="+Inf"} 1001' in metrics
    assert "llmops_query_latency_ms_count 1001" in metrics
    assert len(registry.latency_values_ms) == 1000


def test_case_study_generator_preserves_expected_sections() -> None:
    study = parse_case_study(Path("docs/case-studies/01-llm-evaluation-platform.md"))

    assert study["slug"] == "llm-evaluation-platform"
    assert len(study["sections"]) == 8
    assert study["sections"][0]["heading"] == "Problem"
    assert any(block["type"] == "list" for block in study["sections"][1]["blocks"])


def test_case_study_and_observability_routes_are_exposed() -> None:
    routes = {route.path: route.methods for route in app.routes if hasattr(route, "methods")}

    assert "GET" in routes["/case-studies"]
    assert "GET" in routes["/case-studies/{slug}"]
    assert "GET" in routes["/health"]
    assert "GET" in routes["/observability/summary"]
