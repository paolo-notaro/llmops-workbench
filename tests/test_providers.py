from llmops_workbench.models import GenerationRequest, RetrievedDocument
from llmops_workbench.providers import ExternalPlaceholderProvider, MockLLMProvider


def test_mock_provider_returns_deterministic_answer_text() -> None:
    provider = MockLLMProvider()
    contexts = [
        RetrievedDocument(
            doc_id="deployment_guide",
            title="Synthetic Deployment Guide",
            path="examples/synthetic_docs/deployment_guide.md",
            chunk_id="deployment_guide-1",
            text="Rollback should restore the previous application version and retrieval settings.",
            score=0.9,
        )
    ]

    request = GenerationRequest(query="How should rollback work?", rendered_prompt="Rendered rollback prompt", contexts=contexts)
    first = provider.generate(request)
    second = provider.generate(request)

    assert first.answer == second.answer
    assert "[doc:deployment_guide]" in first.answer


def test_mock_provider_excludes_markdown_headings_from_answers() -> None:
    provider = MockLLMProvider()
    contexts = [
        RetrievedDocument(
            doc_id="deployment_guide",
            title="Synthetic Deployment Guide",
            path="examples/synthetic_docs/deployment_guide.md",
            chunk_id="deployment_guide-1",
            text="# Guide\n\n## Rollback\n\nRollback should restore the previous version.",
            score=0.9,
        )
    ]

    response = provider.generate(GenerationRequest(
        query="How should rollback work?",
        rendered_prompt="Rendered rollback prompt",
        contexts=contexts,
    ))

    assert "##" not in response.answer
    assert "Rollback should restore the previous version" in response.answer


def test_placeholder_token_usage_describes_the_final_answer() -> None:
    contexts = [
        RetrievedDocument(
            doc_id="deployment_guide",
            title="Guide",
            path="guide.md",
            chunk_id="deployment_guide-1",
            text="Rollback restores the previous version.",
            score=0.9,
        )
    ]
    request = GenerationRequest(query="Explain rollback", rendered_prompt="Exact rendered prompt", contexts=contexts)

    response = ExternalPlaceholderProvider("openai", "OPENAI_API_KEY").generate(request)

    assert response.token_usage is not None
    assert response.token_usage.input_tokens == len(request.rendered_prompt.split())
    assert response.token_usage.output_tokens == len(response.answer.split())
    assert response.token_usage.total_tokens == response.token_usage.input_tokens + response.token_usage.output_tokens
