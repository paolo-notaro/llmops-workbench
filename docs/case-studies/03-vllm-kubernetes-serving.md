# vLLM / Kubernetes Serving

## Problem

Serving LLMs reliably requires balancing latency, throughput, cost, model memory, rollout safety, and operational visibility. Changes to model weights, quantization, prompts, or routing can affect user experience and infrastructure stability.

## Constraints

- GPU capacity is limited and expensive.
- Rollouts must protect latency SLOs and error budgets.
- Serving infrastructure may include private model artifacts.
- Public examples should avoid exposing exact cluster topology or production traffic patterns.

## Architecture pattern

A representative pattern uses a model serving layer such as vLLM behind a Kubernetes service, with deployment hygiene around image versioning, resource requests, autoscaling, health checks, canary rollout, and rollback. Traffic can be shifted gradually while SLOs and evaluation metrics are monitored.

## Evaluation / monitoring strategy

Serving changes are evaluated with offline suites, synthetic load, latency percentiles, error rates, token throughput, and canary comparisons. Promotion depends on both answer quality and operational health.

## Failure modes

- Context length or batching changes cause latency spikes.
- GPU memory pressure leads to evictions or degraded throughput.
- Canary metrics look healthy at low traffic but fail at production load.
- Rollback is delayed because model, prompt, and retrieval versions are not tracked together.

## Trade-offs

Self-hosting improves control and can reduce marginal cost, but increases operational burden. Managed providers reduce infrastructure work, but require careful privacy, cost, and reliability controls. Canary rollout reduces blast radius, but adds monitoring and release-management complexity.

## What is intentionally omitted

This public write-up omits exact cluster configuration, provider contracts, model artifacts, production traffic profiles, and internal SLO thresholds.

## Public demo mapping

The local demo uses a `MockLLMProvider` instead of a real vLLM cluster. The same provider boundary in `llmops_workbench/providers.py` is where a real serving adapter would be introduced.
