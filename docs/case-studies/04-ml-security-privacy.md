# ML Security And Privacy

## Problem

LLM systems introduce security and privacy risks including prompt injection, sensitive data leakage, unsafe instruction following, model extraction, and membership inference. These risks must be addressed without overclaiming what simple guardrails can prove.

## Constraints

- Security evaluations may involve sensitive prompts or policy details.
- Privacy controls must be auditable and conservative.
- Guardrails should fail closed for high-risk categories.
- Public demos must avoid operational security details that should remain private.

## Architecture pattern

A representative pattern layers input filtering, retrieval isolation, prompt hardening, output checks, audit logging, and human review. For higher-risk settings, concepts such as trusted execution environments, differential privacy, data minimization, and retention controls may be considered as part of a broader risk program.

## Evaluation / monitoring strategy

The public demo equivalent uses synthetic unsafe prompts and a small set of named input rules, each pairing a harmful capability with the intent to obtain it, so that questions about unsafe behavior are still answered. Production systems can extend this with adversarial prompt suites, privacy leakage tests, red-team findings, policy-based review queues, and incident response playbooks.

## Failure modes

- Prompt injection instructions are retrieved from untrusted documents.
- Outputs include sensitive fields because context filtering is incomplete.
- Guardrails block legitimate use cases due to overly broad rules.
- A model appears safe on static tests but fails under paraphrased attacks.
- Security logs collect more sensitive information than necessary.

## Trade-offs

Strict guardrails reduce risk but can affect usability. Lightweight checks are fast and explainable but incomplete. Advanced privacy technologies can help in specific threat models, but they are not substitutes for data minimization, access control, and operational review.

## What is intentionally omitted

This write-up omits real attack logs, internal policies, sensitive detection rules, private threat models, and exact mitigation thresholds.

## Public demo mapping

The demo maps this pattern to `safety_eval.jsonl`, the named input rules in `llmops_workbench/policy.py`, the guardrail verdicts recorded on every trace in `llmops_workbench/execution.py`, and refusal behavior in `llmops_workbench/providers.py`.
