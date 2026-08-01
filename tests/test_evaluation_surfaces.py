from pathlib import Path

from llmops_workbench.app import app, evaluation_dataset, live_monitoring
from llmops_workbench.dataset import build_dataset_profile, dataset_version, load_evaluation_examples
from llmops_workbench.live_evaluation import evaluate_live_request, summarize_live_requests
from llmops_workbench.providers import MockLLMProvider
from llmops_workbench.rag import LocalTfidfRAGIndex


def test_dataset_profile_is_stable_and_explains_metric_populations() -> None:
    eval_dir = Path("datasets/ground_truth")
    examples = load_evaluation_examples(eval_dir)

    profile = build_dataset_profile(examples, eval_dir)

    assert profile.total_examples == 30
    assert profile.version == dataset_version(eval_dir)
    assert profile.distributions["expected_actions"] == {"abstain": 3, "answer": 23, "refuse": 4}
    assert profile.metric_populations["citation_support"] == 23
    assert profile.metric_populations["format_contract_pass_rate"] == 30
    assert profile.fields and profile.sample_examples


def test_live_metrics_are_deterministic_for_a_grounded_query() -> None:
    index = LocalTfidfRAGIndex.from_directory(Path("examples/synthetic_docs"))
    provider = MockLLMProvider()
    query = "How should rollback be handled after a failed GenAI deployment?"
    retrieved = index.query(query, top_k=3)
    response = provider.generate(query, retrieved)

    first = evaluate_live_request("req-1", "2026-01-01T00:00:00+00:00", query, response.answer, response.provider, 1.0, retrieved)
    second = evaluate_live_request("req-2", "2026-01-01T00:00:01+00:00", query, response.answer, response.provider, 2.0, retrieved)
    report = summarize_live_requests([first, second], total_requests=2, window_limit=50)

    assert first.metrics == second.metrics
    assert first.metrics["citation_validity"] == 1.0
    assert first.metrics["evidence_support"] > 0.7
    assert report.window_size == 2
    assert all(metric.sample_count == 2 for metric in report.quality_metrics)


def test_evaluation_surface_routes_are_exposed() -> None:
    routes = {route.path: route.methods for route in app.routes if hasattr(route, "methods")}

    assert "GET" in routes["/evaluation/offline"]
    assert "POST" in routes["/evaluation/offline/run"]
    assert "GET" in routes["/evaluation/live"]
    assert "GET" in routes["/evaluation/dataset"]
    assert evaluation_dataset().total_examples == 30
    assert live_monitoring().window_size >= 0
