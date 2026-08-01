from pathlib import Path

from llmops_workbench.app import app, documents, get_index
from llmops_workbench.rag import LocalTfidfRAGIndex


def test_retrieval_returns_relevant_document_for_known_query() -> None:
    index = LocalTfidfRAGIndex.from_directory(Path("examples/synthetic_docs"))

    results = index.query("rollback deployment retrieval settings", top_k=2)

    assert results
    assert any(result.doc_id == "deployment_guide" for result in results)
    assert results[0].score > 0


def test_document_metadata_is_derived_from_indexed_markdown() -> None:
    index = LocalTfidfRAGIndex.from_directory(Path("examples/synthetic_docs"))

    summaries = index.document_summaries

    assert [summary.doc_id for summary in summaries] == [
        "ai_governance_notes",
        "deployment_guide",
        "incident_runbook",
        "policy_security",
    ]
    assert summaries[0].title == "Synthetic AI Governance Notes"
    assert summaries[0].description.startswith("These notes summarize")
    assert all(summary.chunk_count >= 1 and summary.indexed for summary in summaries)


def test_documents_endpoint_returns_index_metadata() -> None:
    get_index.cache_clear()

    payload = documents()

    assert len(payload) == 4
    assert {document.doc_id for document in payload} == {
        "ai_governance_notes",
        "deployment_guide",
        "incident_runbook",
        "policy_security",
    }
    route = next(route for route in app.routes if getattr(route, "path", None) == "/documents")
    assert "GET" in route.methods
