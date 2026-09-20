from agentprobe.models.schemas import (
    AggregateMetrics,
    CategoryMetrics,
    ExecutionStatus,
    TaskEvaluation,
)
from agentprobe.runner.dataset_runner import execution_status


def aggregate_metrics(results: list[TaskEvaluation]) -> AggregateMetrics:
    total = len(results)

    if total == 0:
        raise ValueError("Cannot aggregate an empty evaluation run")

    evaluator_passes = {
        "task_success": 0,
        "completion": 0,
        "required_tools": 0,
        "allowed_tools": 0,
        "forbidden_tools": 0,
        "exact_tool_sequence": 0,
        "tool_errors": 0,
    }
    evaluator_applicable = {name: 0 for name in evaluator_passes}

    provider_errors = 0
    skipped_executions = 0
    eligible_executions = 0
    raw_successes = 0
    raw_applicable = 0
    completion_successes = 0
    total_provider_calls = 0
    total_retries = 0
    total_throttle_wait_ms = 0.0
    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for result in results:
        trace = result.agent_result.trace
        status = execution_status(result)
        if status == ExecutionStatus.SKIPPED:
            skipped_executions += 1
            continue
        total_latency += trace.latency_ms
        total_input_tokens += trace.input_tokens
        total_output_tokens += trace.output_tokens
        total_provider_calls += len(trace.provider_calls)
        total_retries += trace.retries
        total_throttle_wait_ms += sum(
            call.throttle_wait_ms for call in trace.provider_calls
        )

        if status == ExecutionStatus.PROVIDER_FAILED:
            provider_errors += 1
        else:
            eligible_executions += 1

        if trace.termination_reason is not None and status == ExecutionStatus.ELIGIBLE:
            if trace.termination_reason.value == "completed":
                completion_successes += 1

        for evaluation in result.evaluations:
            if evaluation.evaluator == "task_success" and evaluation.passed is not None:
                raw_applicable += 1
                raw_successes += int(evaluation.passed)
            if status != ExecutionStatus.ELIGIBLE:
                continue
            if evaluation.evaluator == "forbidden_tools" and not result.task.forbidden_tools:
                continue
            if evaluation.evaluator in evaluator_passes and evaluation.passed is not None:
                evaluator_applicable[evaluation.evaluator] += 1
                if evaluation.passed:
                    evaluator_passes[evaluation.evaluator] += 1

    def pass_rate(name: str) -> float | None:
        applicable = evaluator_applicable[name]
        return evaluator_passes[name] / applicable if applicable else None

    def failure_rate(name: str) -> float | None:
        rate = pass_rate(name)
        return 1 - rate if rate is not None else None

    return AggregateMetrics(
        total_executions=total,
        task_success_rate=pass_rate("task_success"),
        conditional_task_success_rate=pass_rate("task_success"),
        eligible_executions=eligible_executions,
        provider_failed_executions=provider_errors,
        skipped_executions=skipped_executions,
        raw_task_success_rate=raw_successes / raw_applicable if raw_applicable else None,
        completion_rate=(completion_successes / (total - skipped_executions)
                         if total > skipped_executions else None),
        required_tools_rate=pass_rate("required_tools"),
        allowed_tools_rate=pass_rate("allowed_tools"),
        forbidden_tool_violation_rate=failure_rate("forbidden_tools"),
        exact_sequence_rate=pass_rate("exact_tool_sequence"),
        tool_error_rate=failure_rate("tool_errors"),
        provider_error_rate=(provider_errors / (total - skipped_executions)
                             if total > skipped_executions else None),
        total_provider_calls=total_provider_calls,
        total_retries=total_retries,
        retry_rate=(
            total_retries / total_provider_calls
            if total_provider_calls else None
        ),
        total_throttle_wait_ms=total_throttle_wait_ms,
        avg_latency_ms=total_latency / (total - skipped_executions)
        if total > skipped_executions else 0.0,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
    )


def aggregate_by_category(results: list[TaskEvaluation]) -> list[CategoryMetrics]:
    grouped: dict[str, list[TaskEvaluation]] = {}

    for result in results:
        grouped.setdefault(result.task.category, []).append(result)

    category_metrics = []
    for category, category_results in grouped.items():
        metrics = aggregate_metrics(category_results)
        category_metrics.append(
            CategoryMetrics(
                category=category,
                executions=metrics.total_executions,
                task_success_rate=metrics.task_success_rate,
                completion_rate=metrics.completion_rate,
                required_tools_rate=metrics.required_tools_rate,
                allowed_tools_rate=metrics.allowed_tools_rate,
                forbidden_tool_violation_rate=metrics.forbidden_tool_violation_rate,
            )
        )

    return category_metrics
