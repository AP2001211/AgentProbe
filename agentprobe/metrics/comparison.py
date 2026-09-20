from pydantic import BaseModel

from agentprobe.metrics.aggregator import aggregate_metrics
from agentprobe.models.schemas import EvaluationRun


class MetricDelta(BaseModel):
    baseline: float | None
    candidate: float | None
    delta: float | None


class RunComparison(BaseModel):
    baseline_run_id: str
    candidate_run_id: str

    task_success_rate: MetricDelta
    conditional_task_success_rate: MetricDelta
    completion_rate: MetricDelta
    required_tools_rate: MetricDelta
    allowed_tools_rate: MetricDelta
    forbidden_tool_violation_rate: MetricDelta
    exact_sequence_rate: MetricDelta
    tool_error_rate: MetricDelta
    provider_error_rate: MetricDelta
    retry_rate: MetricDelta
    avg_latency_ms: MetricDelta


def _delta(baseline: float | None, candidate: float | None) -> MetricDelta:
    return MetricDelta(
        baseline=baseline,
        candidate=candidate,
        delta=(candidate - baseline
               if baseline is not None and candidate is not None else None),
    )


def compare_runs(baseline: EvaluationRun, candidate: EvaluationRun) -> RunComparison:
    baseline_metrics = aggregate_metrics(baseline.results)
    candidate_metrics = aggregate_metrics(candidate.results)

    return RunComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        task_success_rate=_delta(
            baseline_metrics.task_success_rate,
            candidate_metrics.task_success_rate,
        ),
        conditional_task_success_rate=_delta(
            baseline_metrics.conditional_task_success_rate,
            candidate_metrics.conditional_task_success_rate,
        ),
        completion_rate=_delta(
            baseline_metrics.completion_rate,
            candidate_metrics.completion_rate,
        ),
        required_tools_rate=_delta(
            baseline_metrics.required_tools_rate,
            candidate_metrics.required_tools_rate,
        ),
        allowed_tools_rate=_delta(
            baseline_metrics.allowed_tools_rate,
            candidate_metrics.allowed_tools_rate,
        ),
        forbidden_tool_violation_rate=_delta(
            baseline_metrics.forbidden_tool_violation_rate,
            candidate_metrics.forbidden_tool_violation_rate,
        ),
        exact_sequence_rate=_delta(
            baseline_metrics.exact_sequence_rate,
            candidate_metrics.exact_sequence_rate,
        ),
        tool_error_rate=_delta(
            baseline_metrics.tool_error_rate,
            candidate_metrics.tool_error_rate,
        ),
        provider_error_rate=_delta(
            baseline_metrics.provider_error_rate,
            candidate_metrics.provider_error_rate,
        ),
        retry_rate=_delta(
            baseline_metrics.retry_rate,
            candidate_metrics.retry_rate,
        ),
        avg_latency_ms=_delta(
            baseline_metrics.avg_latency_ms,
            candidate_metrics.avg_latency_ms,
        ),
    )
