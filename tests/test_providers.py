from llmops_workbench.models import RetrievedDocument
from llmops_workbench.providers import MockLLMProvider


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

    first = provider.generate("How should rollback work?", contexts)
    second = provider.generate("How should rollback work?", contexts)

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

    response = provider.generate("How should rollback work?", contexts)

    assert "##" not in response.answer
    assert "Rollback should restore the previous version" in response.answer
