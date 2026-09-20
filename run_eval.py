import asyncio
import json

from agentprobe.agents.gemini_agent import GeminiAgent
from agentprobe.datasets.loader import load_dataset
from agentprobe.evaluators.pipeline import EvaluatorPipeline
from agentprobe.evaluators.task_success import (
    CompletionEvaluator,
    TaskSuccessEvaluator,
)
from agentprobe.evaluators.tool_calls import (
    AllowedToolsEvaluator,
    ExactToolSequenceEvaluator,
    ForbiddenToolsEvaluator,
    RequiredToolsEvaluator,
    ToolErrorEvaluator,
)
from agentprobe.metrics.aggregator import aggregate_by_category, aggregate_metrics
from agentprobe.metrics.failure_report import classify_failures
from agentprobe.models.schemas import EvaluationConfig, EvaluationRun, ProviderBudget, RetryConfig
from agentprobe.runner.dataset_runner import DatasetRunner
from agentprobe.runner.rate_limiter import RateLimiter
from agentprobe.runner.request_budget import (
    BenchmarkPreflightError, RequestBudget, estimate_benchmark, require_preflight,
)
from agentprobe.storage.json_store import JSONRunStore


def format_rate(rate: float | None) -> str:
    return "N/A" if rate is None else f"{rate:.1%}"


async def main():
    config = EvaluationConfig(
        repetitions=1,
        requests_per_minute=5,
        max_retries=3,
    )
    tasks = load_dataset("datasets/support_v1.jsonl")
    assert len(tasks) == 25
    provider_budget = ProviderBudget(
        requests_per_minute=config.requests_per_minute,
        requests_per_day=20,
    )
    request_budget = RequestBudget(provider_budget.requests_per_day)
    estimate = estimate_benchmark(
        len(tasks) * config.repetitions, provider_budget, request_budget.used
    )
    print(f"Preflight: {estimate.model_dump()}")
    try:
        require_preflight(estimate)
    except BenchmarkPreflightError as error:
        print(f"Benchmark not started: {error}")
        return
    limiter = (
        RateLimiter(provider_budget.requests_per_minute)
        if provider_budget.requests_per_minute is not None else None
    )
    agent = GeminiAgent(
        retry_config=RetryConfig(
            max_retries=config.max_retries,
            initial_backoff_seconds=config.initial_backoff_seconds,
            max_backoff_seconds=config.max_backoff_seconds,
        ),
        rate_limiter=limiter,
        request_budget=request_budget,
    )
    pipeline = EvaluatorPipeline([
        RequiredToolsEvaluator(),
        AllowedToolsEvaluator(),
        ForbiddenToolsEvaluator(),
        ExactToolSequenceEvaluator(),
        ToolErrorEvaluator(),
        TaskSuccessEvaluator(),
        CompletionEvaluator(),
    ])
    runner = DatasetRunner(agent=agent, evaluator_pipeline=pipeline)
    results = await runner.run(tasks, repetitions=config.repetitions)
    run = EvaluationRun(
        agent_name=agent.model,
        dataset_name="support-v1",
        repetitions=config.repetitions,
        results=results,
    )
    store = JSONRunStore()
    run_path = store.save(run)
    failure_path = store.directory / f"{run.run_id}-failures.json"
    failure_path.write_text(json.dumps(classify_failures(run.results), indent=2), encoding="utf-8")
    metrics = aggregate_metrics(run.results)

    for result in results:
        print(f"\n{'=' * 60}")
        print(f"TASK: {result.task.id}")
        print(f"Status: {result.status.value}")
        print(f"{'=' * 60}")
        print(f"Termination: {result.agent_result.trace.termination_reason}")
        print(f"Latency: {result.agent_result.trace.latency_ms:.0f} ms")
        print(f"Provider calls: {len(result.agent_result.trace.provider_calls)}")
        print(f"Retries: {result.agent_result.trace.retries}")
        print(
            "Throttle wait: "
            f"{sum(call.throttle_wait_ms for call in result.agent_result.trace.provider_calls):.0f} ms"
        )
        print("Provider attempts:")
        for call in result.agent_result.trace.provider_calls:
            status = "OK" if call.error is None else call.error.split(" ", 1)[0]
            print(
                f"  round {call.round} attempt {call.attempt}: "
                f"wait {call.throttle_wait_ms:.0f} ms, "
                f"provider {call.latency_ms:.0f} ms, {status}"
            )
        print("Tools:", [call.name for call in result.agent_result.trace.tool_calls])
        print("\nEvaluations:")

        for evaluation in result.evaluations:
            status = (
                "N/A" if evaluation.passed is None
                else "PASS" if evaluation.passed else "FAIL"
            )
            score = "N/A" if evaluation.score is None else f"{evaluation.score:.2f}"
            print(f"  {evaluation.evaluator:<22}{status:<6} {score}")

    print("\n")
    print("=" * 60)
    print("AGENTPROBE EVALUATION")
    print("=" * 60)
    print(f"Run ID:          {run.run_id}")
    print(f"Agent:           {run.agent_name}")
    print(f"Dataset:         {run.dataset_name}")
    print(f"Executions:      {metrics.total_executions}")
    print(f"Saved:           {run_path}")
    print(f"Failures:        {failure_path}")
    print(f"Eligible:        {metrics.eligible_executions}")
    print(f"Provider failed: {metrics.provider_failed_executions}")
    print(f"Skipped:         {metrics.skipped_executions}")

    print("\nQUALITY")
    print(f"Task success:     {format_rate(metrics.task_success_rate)}")
    print(f"Raw task success: {format_rate(metrics.raw_task_success_rate)}")
    print(f"Eligible runs:    {metrics.eligible_executions}")
    print(f"Conditional:     {format_rate(metrics.conditional_task_success_rate)}")
    print(f"Completion:      {format_rate(metrics.completion_rate)}")
    print(f"Required tools:  {format_rate(metrics.required_tools_rate)}")
    print(f"Allowed tools:   {format_rate(metrics.allowed_tools_rate)}")
    print(f"Forbidden calls: {format_rate(metrics.forbidden_tool_violation_rate)}")

    print("\nRELIABILITY")
    print(f"Tool errors:     {format_rate(metrics.tool_error_rate)}")
    print(f"Provider errors: {format_rate(metrics.provider_error_rate)}")
    print(f"Provider calls:  {metrics.total_provider_calls}")
    print(f"Retries:         {metrics.total_retries}")
    print(f"Retry rate:      {format_rate(metrics.retry_rate)}")

    print("\nPERFORMANCE")
    print(f"Avg latency:     {metrics.avg_latency_ms:.0f} ms")
    print(f"Throttle wait:   {metrics.total_throttle_wait_ms:.0f} ms")
    print(f"Input tokens:    {metrics.total_input_tokens}")
    print(f"Output tokens:   {metrics.total_output_tokens}")

    print("\nDIAGNOSTICS")
    print(f"Exact sequence:  {format_rate(metrics.exact_sequence_rate)}")

    print("\nBY CATEGORY")
    print(f"{'Category':<24}{'Success':>10}{'Forbidden':>14}")
    for category in aggregate_by_category(run.results):
        print(
            f"{category.category:<24}"
            f"{format_rate(category.task_success_rate):>10}"
            f"{format_rate(category.forbidden_tool_violation_rate):>14}"
        )


asyncio.run(main())
