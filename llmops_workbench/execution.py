"""Canonical request execution for live traffic and offline replay."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from llmops_workbench.models import (
    GenerationRequest,
    GuardrailVerdict,
    RequestTrace,
    RetrievalMatch,
    RetrievalTrace,
    TokenUsage,
    TraceTimings,
)
from llmops_workbench.providers import LLMProvider, UNSAFE_KEYWORDS
from llmops_workbench.rag import LocalTfidfRAGIndex


POLICY_ID = "public-demo-keyword-policy"
POLICY_VERSION = "1"
PROMPT_ID = "implicit-grounded-prompt"
PROMPT_VERSION = "1"
REFUSAL_ANSWER = (
    "I cannot help with requests to steal credentials, bypass access controls, "
    "or exfiltrate private data. Use approved incident response and security "
    "review workflows instead."
)


def execute_request(
    query: str,
    index: LocalTfidfRAGIndex,
    provider: LLMProvider,
    *,
    mode: Literal["live", "evaluation"],
    top_k: int = 3,
    dataset_id: str | None = None,
    dataset_version: str | None = None,
    example_id: str | None = None,
    trace_id: str | None = None,
) -> RequestTrace:
    """Run policy, retrieval, and generation once and return their shared trace."""

    total_start = time.perf_counter()
    config_snapshot = {
        "prompt": {"id": PROMPT_ID, "version": PROMPT_VERSION},
        "retriever": {
            "strategy": "tfidf_cosine_similarity",
            "top_k": top_k,
            "minimum_score": None,
        },
        "guardrail": {"policy_id": POLICY_ID, "version": POLICY_VERSION},
        "provider": {"name": provider.name},
    }
    config_id = _stable_id(config_snapshot)

    stage_start = time.perf_counter()
    input_verdict = evaluate_input_guardrail(query)
    guardrail_ms = _elapsed_ms(stage_start)

    stage_start = time.perf_counter()
    retrieved_docs = index.query(query, top_k=top_k)
    retrieval_ms = _elapsed_ms(stage_start)
    rendered_prompt = render_prompt(query, retrieved_docs)
    retrieval = build_retrieval_trace(
        query,
        retrieved_docs,
        top_k=top_k,
        candidate_count=index.candidate_count,
        duration_ms=retrieval_ms,
    )

    if input_verdict.action == "refuse":
        answer = REFUSAL_ANSWER
        generation_ms = 0.0
        provider_name = "guardrail"
        model = f"{POLICY_ID}-v{POLICY_VERSION}"
        token_usage = _estimate_usage(rendered_prompt, answer)
    else:
        response = provider.generate(GenerationRequest(
            query=query,
            rendered_prompt=rendered_prompt,
            contexts=retrieved_docs,
        ))
        answer = response.answer
        generation_ms = response.latency_ms
        provider_name = response.provider
        model = response.model
        token_usage = response.token_usage or _estimate_usage(rendered_prompt, answer)

    stage_start = time.perf_counter()
    output_verdict = evaluate_output_guardrail(query, answer)
    guardrail_ms += _elapsed_ms(stage_start)
    return RequestTrace(
        trace_id=trace_id or f"trace-{uuid4().hex}",
        mode=mode,
        timestamp=datetime.now(UTC).isoformat(),
        query=query,
        config_id=config_id,
        config_snapshot=config_snapshot,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        example_id=example_id,
        guardrail_verdicts=[input_verdict, output_verdict],
        retrieval=retrieval,
        rendered_prompt=rendered_prompt,
        answer=answer,
        provider=provider_name,
        model=model,
        token_usage=token_usage,
        timings=TraceTimings(
            guardrail_ms=round(guardrail_ms, 3),
            retrieval_ms=round(retrieval_ms, 3),
            generation_ms=round(generation_ms, 3),
            total_ms=round(_elapsed_ms(total_start), 3),
        ),
    )


def evaluate_input_guardrail(query: str) -> GuardrailVerdict:
    matches = [keyword for keyword in UNSAFE_KEYWORDS if keyword in query.lower()]
    return GuardrailVerdict(
        stage="input",
        policy_id=POLICY_ID,
        policy_version=POLICY_VERSION,
        action="refuse" if matches else "answer",
        passed=not matches,
        matched_rules=matches,
        reason="Unsafe request pattern detected." if matches else "No input policy rule matched.",
    )


def evaluate_output_guardrail(query: str, answer: str) -> GuardrailVerdict:
    unsafe = any(keyword in query.lower() for keyword in UNSAFE_KEYWORDS)
    refused = "cannot help" in answer.lower()
    passed = unsafe == refused
    return GuardrailVerdict(
        stage="output",
        policy_id=POLICY_ID,
        policy_version=POLICY_VERSION,
        action="refuse" if refused else "answer",
        passed=passed,
        matched_rules=["unsafe_request_refused"] if unsafe and refused else [],
        reason="Response routing matches input policy." if passed else "Response routing conflicts with input policy.",
    )


def render_prompt(query: str, retrieved_docs) -> str:
    context = "\n\n".join(f"[{doc.chunk_id}] {doc.text}" for doc in retrieved_docs)
    return f"Answer only from the supplied synthetic context and cite sources.\n\nQuestion: {query}\n\nContext:\n{context}"


def build_retrieval_trace(
    query: str,
    retrieved_docs,
    *,
    top_k: int,
    candidate_count: int,
    duration_ms: float,
) -> RetrievalTrace:
    """Explain which chunks were selected by the current no-threshold top-k strategy."""

    query_terms = _terms(query)
    results = [
        RetrievalMatch(
            **document.model_dump(),
            rank=rank,
            matched_terms=sorted(query_terms & _terms(document.text)),
            selection_reason="ranked_similarity" if document.score > 0 else "top_k_fill",
        )
        for rank, document in enumerate(retrieved_docs, start=1)
    ]
    return RetrievalTrace(
        strategy="tfidf_cosine_similarity",
        top_k=top_k,
        minimum_score=None,
        candidate_count=candidate_count,
        duration_ms=round(duration_ms, 3),
        results=results,
    )


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower()))


def _stable_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()[:16]}"


def _estimate_usage(rendered_prompt: str, answer: str) -> TokenUsage:
    input_tokens = len(rendered_prompt.split())
    output_tokens = len(answer.split())
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens,
                      total_tokens=input_tokens + output_tokens, method="whitespace_estimate")


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
