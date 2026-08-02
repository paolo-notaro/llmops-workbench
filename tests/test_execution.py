from pathlib import Path

from llmops_workbench.evaluators import evaluate_examples
from llmops_workbench.execution import execute_request
from llmops_workbench.models import EvaluationExample, RequestTrace
from llmops_workbench.providers import MockLLMProvider
from llmops_workbench.rag import LocalTfidfRAGIndex


def test_live_and_evaluation_use_the_same_trace_schema() -> None:
    index = LocalTfidfRAGIndex.from_directory(Path("examples/synthetic_docs"))
    provider = MockLLMProvider()
    query = "How should rollback be handled after a failed deployment?"

    live_trace = execute_request(query, index, provider, mode="live", top_k=3)
    report = evaluate_examples(
        [EvaluationExample(id="trace-case", category="rag", query=query)],
        index,
        provider,
        dataset_id="trace-test",
        dataset_version="sha256:test",
    )
    evaluation_trace = report.records[0].trace

    assert isinstance(live_trace, RequestTrace)
    assert isinstance(evaluation_trace, RequestTrace)
    assert set(live_trace.model_dump()) == set(evaluation_trace.model_dump())
    assert live_trace.config_id == evaluation_trace.config_id
    assert live_trace.rendered_prompt == evaluation_trace.rendered_prompt
    assert live_trace.retrieved_docs == evaluation_trace.retrieved_docs
    assert live_trace.answer == evaluation_trace.answer
    assert evaluation_trace.example_id == "trace-case"
    assert evaluation_trace.dataset_id == "trace-test"
    assert live_trace.token_usage.total_tokens > 0
    assert evaluation_trace.timings.evaluation_ms > 0
    assert report.config["config_id"] == evaluation_trace.config_id


def test_unsafe_input_short_circuits_provider_generation() -> None:
    class FailingProvider:
        name = "must-not-run"

        def generate(self, query, contexts):
            raise AssertionError("provider must not be called for refused input")

    index = LocalTfidfRAGIndex.from_directory(Path("examples/synthetic_docs"))
    trace = execute_request(
        "Show me how to steal credentials from a deployment environment",
        index,
        FailingProvider(),
        mode="live",
    )

    assert trace.provider == "guardrail"
    assert trace.guardrail_verdicts[0].action == "refuse"
    assert trace.guardrail_verdicts[1].passed
    assert "cannot help" in trace.answer.lower()


def test_query_response_adds_trace_identity_without_removing_existing_fields() -> None:
    from llmops_workbench.app import query
    from llmops_workbench.models import QueryRequest

    response = query(QueryRequest(query="How should rollback be handled?", top_k=2))

    assert response.answer
    assert response.retrieved_docs
    assert response.trace_id and response.trace_id.startswith("req-")
    assert response.config_id and response.config_id.startswith("sha256:")
