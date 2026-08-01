"""Profiling helpers for the synthetic annotated evaluation dataset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from llmops_workbench.models import (
    DatasetExampleSummary,
    DatasetFieldDefinition,
    DatasetProfile,
    EvaluationExample,
)


def load_evaluation_examples(eval_dir: Path) -> list[EvaluationExample]:
    """Load deterministic JSONL examples from a directory."""

    examples: list[EvaluationExample] = []
    for path in sorted(eval_dir.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                raise ValueError(f"Blank line in {path}:{line_number}")
            examples.append(EvaluationExample(**json.loads(line)))
    return examples


def dataset_version(eval_dir: Path) -> str:
    """Return a stable short hash over every ground-truth JSONL file."""

    digest = hashlib.sha256()
    for path in sorted(eval_dir.glob("*.jsonl")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()[:12]}"


def build_dataset_profile(examples: list[EvaluationExample], eval_dir: Path) -> DatasetProfile:
    """Describe annotation coverage and metric denominators."""

    groups: dict[str, list[EvaluationExample]] = defaultdict(list)
    for example in examples:
        if example.pair_id:
            groups[example.pair_id].append(example)
    robustness_comparisons = sum(max(0, len(group) - 1) for group in groups.values())
    distributions = {
        "categories": _counter(example.category for example in examples),
        "expected_actions": _counter(example.expected_action for example in examples),
        "perturbations": _counter(example.perturbation or "unpaired" for example in examples),
        "risk_tags": _counter(tag for example in examples for tag in example.risk_tags),
        "expected_documents": _counter(doc for example in examples for doc in example.expected_source_docs),
    }
    answer_count = sum(example.expected_action == "answer" for example in examples)
    return DatasetProfile(
        dataset_id="llmops-eval-ground-truth",
        version=dataset_version(eval_dir),
        source_path="datasets/ground_truth/llmops_eval_ground_truth.jsonl",
        annotation_method=(
            "Manually authored synthetic cases with expected sources, support terms, policy actions, "
            "format contracts, and paired perturbations"
        ),
        total_examples=len(examples),
        pair_count=len(groups),
        metric_populations={
            "evidence_retrieval_f1": sum(bool(example.expected_source_docs) for example in examples),
            "citation_support": answer_count,
            "answer_grounding": answer_count,
            "policy_decision_accuracy": len(examples),
            "format_contract_pass_rate": len(examples),
            "robustness_consistency": robustness_comparisons,
        },
        distributions=distributions,
        fields=[
            DatasetFieldDefinition(name="expected_source_docs", purpose="Documents the retriever should return"),
            DatasetFieldDefinition(name="support_terms", purpose="Evidence phrases expected in supporting documents"),
            DatasetFieldDefinition(name="expected_action", purpose="Required answer, refusal, or abstention route"),
            DatasetFieldDefinition(name="required_format", purpose="Output contract applied by deterministic validation"),
            DatasetFieldDefinition(name="pair_id", purpose="Connects a base prompt with robustness variants"),
            DatasetFieldDefinition(name="perturbation", purpose="Names the paraphrase, injection, or ambiguity transformation"),
            DatasetFieldDefinition(name="risk_tags", purpose="Groups cases by operational and safety risk"),
        ],
        sample_examples=[
            DatasetExampleSummary(
                id=example.id,
                query=example.query,
                expected_action=example.expected_action,
                expected_source_docs=example.expected_source_docs,
                support_terms=example.support_terms,
                pair_id=example.pair_id,
                perturbation=example.perturbation,
                risk_tags=example.risk_tags,
            )
            for example in _representative_examples(examples)
        ],
    )


def _counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _representative_examples(examples: list[EvaluationExample]) -> list[EvaluationExample]:
    selected: list[EvaluationExample] = []
    seen_actions: set[str] = set()
    for example in examples:
        if example.expected_action not in seen_actions:
            selected.append(example)
            seen_actions.add(example.expected_action)
    variant = next((example for example in examples if example.perturbation not in {None, "base"}), None)
    if variant and variant not in selected:
        selected.append(variant)
    return selected[:4]
