# RAG Observability Stack

## Problem

RAG systems can fail even when the language model behaves well. Poor chunking, weak retrieval, missing citations, stale documents, or latency spikes can produce unsupported answers and low user trust.

## Constraints

- Source documents may be private or regulated.
- Retrieval traces can reveal sensitive user intent.
- Product teams need interpretable quality signals, not only embedding scores.
- Monitoring must separate retrieval failures from generation failures.

## Architecture pattern

A sanitized abstraction of this pattern includes document ingestion, chunking, retrieval, optional reranking, answer generation, citation checks, and telemetry across each stage. Product metrics such as successful resolution rate sit alongside system metrics such as retrieval hit rate, citation coverage, latency, and errors.

## Evaluation / monitoring strategy

Evaluation checks compare expected concepts against retrieved chunks, verify citation presence, and track answer grounding. Production equivalents often add click feedback, human review, query clustering, reranker diagnostics, and traces that show which chunks influenced an answer.

## Failure modes

- Chunks are too large to retrieve precisely or too small to preserve context.
- Retrieved documents are topically similar but not actually sufficient.
- Citations are present but point to irrelevant sources.
- Reranking improves relevance but adds latency and operational complexity.
- Retrieval quality drops after document updates.

## Trade-offs

Simple retrieval is cheap and explainable, but may miss semantic matches. Dense retrieval and reranking improve recall, but require more infrastructure and evaluation rigor. Strict grounding checks reduce unsupported answers, but can increase refusals for underspecified questions.

## What is intentionally omitted

The public demo omits private document collections, production embeddings, telemetry backends, exact chunking policies, and internal product analytics.

## Public demo mapping

The demo maps this pattern to `llmops_workbench/rag.py`, `llmops_workbench/evaluators.py`, and the `/metrics` endpoint in `llmops_workbench/app.py`.
