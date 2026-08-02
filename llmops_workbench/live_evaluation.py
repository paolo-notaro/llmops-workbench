"""Reference-free quality signals for interactive requests."""

from __future__ import annotations

import re
import time

from llmops_workbench.evaluators import CITATION_RE, REFUSAL_MARKERS, STOPWORDS
from llmops_workbench.models import LiveEvaluationRecord, LiveMetricDefinition, LiveMonitoringReport, RequestTrace, RetrievedDocument
from llmops_workbench.providers import UNSAFE_KEYWORDS


LIVE_METRICS = {
    "evidence_support": (
        "Evidence support proxy", 0.72,
        "Important answer terms found in retrieved or cited evidence",
        "|answer terms intersect evidence terms| / |answer terms|",
    ),
    "citation_validity": (
        "Citation validity", 1.0,
        "Citations that resolve to documents retrieved for the request",
        "|cited docs intersect retrieved docs| / |cited docs|",
    ),
    "retrieval_confidence": (
        "Retrieval confidence", 0.62,
        "Combined top-score strength and concentration within positive results",
        "0.5 * min(s1 / 0.20, 1) + 0.5 * s1 / sum(positive scores)",
    ),
    "policy_consistency": (
        "Policy consistency", 1.0,
        "Unsafe-keyword routing agrees with refusal behavior",
        "1[unsafe query equals refusal route]",
    ),
    "response_contract": (
        "Response contract", 1.0,
        "Response length, citation, and refusal shape pass deterministic checks",
        "1[interactive response contract passes]",
    ),
}


def evaluate_live_request(
    request_id: str,
    timestamp: str,
    query: str,
    answer: str,
    provider: str,
    latency_ms: float,
    retrieved_docs: list[RetrievedDocument],
) -> LiveEvaluationRecord:
    """Evaluate one request without expected answers or source labels."""

    refusal = _is_refusal(answer)
    cited_docs = set(CITATION_RE.findall(answer))
    retrieved_ids = {doc.doc_id for doc in retrieved_docs}
    evidence_docs = [doc for doc in retrieved_docs if not cited_docs or doc.doc_id in cited_docs]
    answer_terms = _important_terms(CITATION_RE.sub(" ", answer))
    evidence_terms = _important_terms(" ".join(doc.text for doc in evidence_docs))
    evidence_support = 1.0 if refusal else _ratio(len(answer_terms & evidence_terms), len(answer_terms))
    citation_validity = 1.0 if refusal else _ratio(len(cited_docs & retrieved_ids), len(cited_docs))
    retrieval_confidence = _retrieval_confidence(retrieved_docs)
    unsafe = any(keyword in query.lower() for keyword in UNSAFE_KEYWORDS)
    policy_consistency = 1.0 if unsafe == refusal else 0.0
    word_count = len(answer.split())
    contract_passed = (
        (refusal and not cited_docs and word_count <= 80)
        or (not refusal and bool(cited_docs) and 12 <= word_count <= 120)
    )
    metrics = {
        "evidence_support": round(evidence_support, 3),
        "citation_validity": round(citation_validity, 3),
        "retrieval_confidence": round(retrieval_confidence, 3),
        "policy_consistency": policy_consistency,
        "response_contract": 1.0 if contract_passed else 0.0,
    }
    checks = {name: value >= LIVE_METRICS[name][1] for name, value in metrics.items()}
    return LiveEvaluationRecord(
        request_id=request_id,
        timestamp=timestamp,
        provider=provider,
        latency_ms=round(latency_ms, 3),
        retrieved_count=len([doc for doc in retrieved_docs if doc.score > 0]),
        metrics=metrics,
        checks=checks,
        requires_review=not all(checks.values()),
    )


def evaluate_live_trace(trace: RequestTrace) -> LiveEvaluationRecord:
    """Read reference-free quality signals from a canonical request trace."""

    evaluation_start = time.perf_counter()
    record = evaluate_live_request(
        request_id=trace.trace_id,
        timestamp=trace.timestamp,
        query=trace.query,
        answer=trace.answer,
        provider=trace.provider,
        latency_ms=trace.timings.generation_ms,
        retrieved_docs=trace.retrieved_docs,
    )
    trace.timings.evaluation_ms = round((time.perf_counter() - evaluation_start) * 1000, 3)
    trace.timings.total_ms = round(trace.timings.total_ms + trace.timings.evaluation_ms, 3)
    return record


def summarize_live_requests(
    records: list[LiveEvaluationRecord], total_requests: int, *, window_limit: int
) -> LiveMonitoringReport:
    """Aggregate the rolling in-memory request window."""

    metric_definitions = []
    for name, (label, threshold, explanation, formula) in LIVE_METRICS.items():
        value = _mean([record.metrics[name] for record in records])
        metric_definitions.append(LiveMetricDefinition(
            name=name,
            label=label,
            value=value,
            threshold=threshold,
            passed=value >= threshold if records else None,
            sample_count=len(records),
            explanation=explanation,
            formula=formula,
        ))
    latencies = sorted(record.latency_ms for record in records)
    return LiveMonitoringReport(
        total_requests=total_requests,
        window_size=len(records),
        review_count=sum(record.requires_review for record in records),
        pass_rate=_mean([0.0 if record.requires_review else 1.0 for record in records]),
        average_latency_ms=round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        latency_p95_ms=_percentile(latencies, 0.95),
        quality_metrics=metric_definitions,
        recent_records=list(reversed(records[-8:])),
    )


def _important_terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower())
    return {word for word in words if word not in STOPWORDS and not word.startswith("doc")}


def _is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _retrieval_confidence(retrieved_docs: list[RetrievedDocument]) -> float:
    positive_scores = [doc.score for doc in retrieved_docs if doc.score > 0]
    if not positive_scores:
        return 0.0
    top_score = max(positive_scores)
    strength = min(top_score / 0.20, 1.0)
    concentration = top_score / sum(positive_scores)
    return (strength + concentration) / 2


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * percentile)))
    return round(values[index], 3)
