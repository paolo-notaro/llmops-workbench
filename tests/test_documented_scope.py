from pathlib import Path


def test_readme_documents_public_runtime_stance_and_non_goals() -> None:
    readme = Path("README.md").read_text(encoding="utf-8").lower()

    assert "request trace" in readme
    assert "mock-only public runtime" in readme
    for non_goal in (
        "production serving",
        "deep or state-of-the-art retrieval",
        "kubernetes orchestration",
        "multi-tenancy",
        "authentication or authorization",
    ):
        assert non_goal in readme
