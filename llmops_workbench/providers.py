"""LLM provider abstractions."""

from __future__ import annotations

import os
import re
import time
from typing import Protocol

from llmops_workbench.models import GenerationRequest, LLMResponse, RetrievedDocument, TokenUsage


UNSAFE_KEYWORDS = (
    "steal credentials",
    "credential theft",
    "bypass access controls",
    "exfiltrate",
    "malware",
    "private data",
    "reveal any private customer logs",
)


class LLMProvider(Protocol):
    """Protocol implemented by LLM providers."""

    name: str

    def generate(self, request: GenerationRequest) -> LLMResponse:
        """Generate an answer for a query and retrieved context."""


class MockLLMProvider:
    """Deterministic local provider used by default."""

    name = "mock"

    def generate(self, request: GenerationRequest) -> LLMResponse:
        """Generate a deterministic answer without network access."""

        start = time.perf_counter()
        query = request.query
        contexts = request.contexts
        lowered = query.lower()
        if any(keyword in lowered for keyword in UNSAFE_KEYWORDS):
            answer = (
                "I cannot help with requests to steal credentials, bypass access controls, "
                "or exfiltrate private data. Use approved incident response and security "
                "review workflows instead."
            )
        else:
            answer = self._grounded_answer(query, contexts)
        latency_ms = (time.perf_counter() - start) * 1000
        input_tokens = len(request.rendered_prompt.split())
        output_tokens = len(answer.split())
        return LLMResponse(
            answer=answer,
            provider=self.name,
            model="deterministic-extractive-v1",
            latency_ms=latency_ms,
            token_usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                method="whitespace_estimate",
            ),
        )

    def _grounded_answer(self, query: str, contexts: list[RetrievedDocument]) -> str:
        citations = " ".join(f"[doc:{doc.doc_id}]" for doc in contexts[:2])
        combined = " ".join(_markdown_body(doc.text) for doc in contexts[:2])
        sentences = _first_relevant_sentences(combined, query, limit=2)
        if not sentences:
            sentences = ["The retrieved synthetic documents do not provide enough detail for a specific answer."]
        return f"{' '.join(sentences)} {citations}".strip()


class ExternalPlaceholderProvider(MockLLMProvider):
    """Placeholder for a real provider adapter that intentionally avoids network calls."""

    def __init__(self, provider_name: str, api_key_env: str) -> None:
        self.name = f"{provider_name}-placeholder"
        self.api_key_env = api_key_env
        self.api_key_present = bool(os.getenv(api_key_env))

    def generate(self, request: GenerationRequest) -> LLMResponse:
        response = super().generate(request)
        suffix = f" Provider placeholder configured via {self.api_key_env}; no external API call was made."
        answer = f"{response.answer}{suffix}"
        input_tokens = len(request.rendered_prompt.split())
        output_tokens = len(answer.split())
        return LLMResponse(
            answer=answer,
            provider=self.name,
            latency_ms=response.latency_ms,
            model=response.model,
            token_usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                method="whitespace_estimate",
            ),
        )


def provider_from_env(provider_name: str | None = None) -> LLMProvider:
    """Create a provider from the configured provider name."""

    provider = (provider_name or os.getenv("LLM_PROVIDER", "mock")).strip().lower()
    if provider == "mock":
        return MockLLMProvider()
    if provider == "openai":
        return ExternalPlaceholderProvider("openai", "OPENAI_API_KEY")
    if provider == "gemini":
        return ExternalPlaceholderProvider("gemini", "GEMINI_API_KEY")
    if provider == "anthropic":
        return ExternalPlaceholderProvider("anthropic", "ANTHROPIC_API_KEY")
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def _markdown_body(text: str) -> str:
    """Flatten Markdown body text while excluding structural markers."""

    body_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or re.match(r"^#{1,6}\s", line):
            continue
        body_lines.append(re.sub(r"^[-*+]\s+", "", line))
    return " ".join(body_lines)


def _first_relevant_sentences(text: str, query: str, limit: int) -> list[str]:
    terms = {term for term in query.lower().replace("?", "").split() if len(term) > 3}
    sentences = [part.strip() for part in text.split(".") if part.strip()]
    scored: list[tuple[int, str]] = []
    for sentence in sentences:
        lowered = sentence.lower()
        score = sum(1 for term in terms if term in lowered)
        scored.append((score, sentence))
    ranked = [sentence for score, sentence in sorted(scored, reverse=True) if score > 0]
    fallback = [sentence for sentence in sentences if sentence not in ranked]
    return (ranked + fallback)[:limit]
