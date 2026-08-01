"""Local Markdown loading and TF-IDF retrieval."""

from __future__ import annotations

import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llmops_workbench.models import DocumentChunk, DocumentSummary, RetrievedDocument


HEADING_RE = re.compile(r"^#\s+(?P<title>.+)$", re.MULTILINE)
MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+")


def load_markdown_documents(docs_dir: Path) -> list[DocumentChunk]:
    """Load synthetic Markdown documents and split them into readable chunks."""

    chunks: list[DocumentChunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title_match = HEADING_RE.search(text)
        title = title_match.group("title") if title_match else path.stem.replace("_", " ").title()
        doc_id = path.stem
        for index, chunk_text in enumerate(_chunk_markdown(text), start=1):
            chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    title=title,
                    path=str(path),
                    chunk_id=f"{doc_id}-{index}",
                    text=chunk_text,
                )
            )
    return chunks


def _chunk_markdown(text: str, max_chars: int = 900) -> list[str]:
    """Split Markdown into compact paragraph groups."""

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if current and current_len + len(paragraph) > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def build_document_summaries(chunks: list[DocumentChunk]) -> list[DocumentSummary]:
    """Build stable public metadata from the indexed Markdown chunks."""

    grouped: dict[str, list[DocumentChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)

    summaries: list[DocumentSummary] = []
    for doc_chunks in grouped.values():
        first = doc_chunks[0]
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", first.text) if part.strip()]
        description = next(
            (paragraph for paragraph in paragraphs if not MARKDOWN_HEADING_RE.match(paragraph)),
            "Synthetic reference document used by the local retrieval demo.",
        )
        summaries.append(
            DocumentSummary(
                doc_id=first.doc_id,
                title=first.title,
                description=description,
                chunk_count=len(doc_chunks),
            )
        )
    return summaries


class LocalTfidfRAGIndex:
    """Small local retrieval index using scikit-learn TF-IDF."""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("At least one document chunk is required.")
        self._chunks = chunks
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform([chunk.text for chunk in chunks])

    @property
    def document_summaries(self) -> list[DocumentSummary]:
        """Return read-only metadata for documents represented in the index."""

        return build_document_summaries(self._chunks)

    @classmethod
    def from_directory(cls, docs_dir: Path) -> "LocalTfidfRAGIndex":
        """Build an index from Markdown files in a directory."""

        return cls(load_markdown_documents(docs_dir))

    def query(self, query: str, top_k: int = 3) -> list[RetrievedDocument]:
        """Return the top matching synthetic document chunks."""

        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        results: list[RetrievedDocument] = []
        for idx, score in ranked[:top_k]:
            chunk = self._chunks[idx]
            results.append(
                RetrievedDocument(
                    **chunk.model_dump(),
                    score=round(float(score), 4),
                )
            )
        return results
