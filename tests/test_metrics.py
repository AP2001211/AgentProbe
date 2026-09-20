import tempfile
import unittest
from pathlib import Path

from agentprobe.metrics.aggregator import aggregate_by_category, aggregate_metrics
from agentprobe.metrics.comparison import compare_runs
from agentprobe.models.schemas import (
    AgentResult,
    EvaluationResult,
    EvaluationRun,
    ExecutionTrace,
    ProviderCall,
    Task,
    TaskEvaluation,
    TerminationReason,
)
from agentprobe.storage.json_store import JSONRunStore


def make_evaluation(
    task_id: str,
    *,
    success: bool = True,
    provider_error: bool = False,
    tool_error: bool = False,
    allowed_tools: bool = True,
    forbidden_tool_violation: bool = False,
    forbidden_restriction: bool = True,
    category: str = "general",
    latency_ms: float = 100,
    input_tokens: int = 10,
    output_tokens: int = 2,
    provider_calls: int = 0,
    retries: int = 0,
    throttle_wait_ms: float = 0,
) -> TaskEvaluation:
    passed = {
        "task_success": success,
        "completion": not provider_error,
        "required_tools": success,
        "allowed_tools": allowed_tools,
        "forbidden_tools": not forbidden_tool_violation,
        "exact_tool_sequence": success,
        "tool_errors": not tool_error,
    }
    return TaskEvaluation(
        task=Task(
            id=task_id,
            input=task_id,
            category=category,
            forbidden_tools=["create_refund"] if forbidden_restriction else [],
        ),
        agent_result=AgentResult(
            output="",
            trace=ExecutionTrace(
                task_id=task_id,
                termination_reason=(
                    TerminationReason.PROVIDER_ERROR
                    if provider_error else TerminationReason.COMPLETED
                ),
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider_calls=[
                    ProviderCall(
                        round=1,
                        attempt=index + 1,
                        latency_ms=10,
                        throttle_wait_ms=throttle_wait_ms,
                    )
                    for index in range(provider_calls)
                ],
                retries=retries,
            ),
        ),
        evaluations=[
            EvaluationResult(evaluator=name, passed=value, score=float(value))
            for name, value in passed.items()
        ],
    )


class AggregateMetricsTests(unittest.TestCase):
    def test_three_successful_executions(self):
        results = [make_evaluation(str(index)) for index in range(3)]
        metrics = aggregate_metrics(results)

        self.assertEqual(metrics.total_executions, 3)
        self.assertEqual(metrics.task_success_rate, 1.0)
        self.assertEqual(metrics.conditional_task_success_rate, 1.0)
        self.assertEqual(metrics.eligible_executions, 3)
        self.assertEqual(metrics.completion_rate, 1.0)
        self.assertEqual(metrics.allowed_tools_rate, 1.0)
        self.assertEqual(metrics.forbidden_tool_violation_rate, 0.0)
        self.assertEqual(metrics.tool_error_rate, 0.0)
        self.assertEqual(metrics.avg_latency_ms, 100)
        self.assertEqual(metrics.total_input_tokens, 30)
        self.assertEqual(metrics.total_output_tokens, 6)

    def test_mixed_outcomes(self):
        results = [
            make_evaluation("one"),
            make_evaluation("two"),
            make_evaluation(
                "three", success=False, provider_error=True, tool_error=True,
                allowed_tools=False, forbidden_tool_violation=True,
                latency_ms=400, input_tokens=0, output_tokens=0,
            ),
        ]
        metrics = aggregate_metrics(results)

        self.assertEqual(metrics.task_success_rate, 1.0)
        self.assertAlmostEqual(metrics.raw_task_success_rate, 2 / 3)
        self.assertEqual(metrics.conditional_task_success_rate, 1.0)
        self.assertEqual(metrics.eligible_executions, 2)
        self.assertAlmostEqual(metrics.provider_error_rate, 1 / 3)
        self.assertEqual(metrics.tool_error_rate, 0.0)
        self.assertEqual(metrics.allowed_tools_rate, 1.0)
        self.assertEqual(metrics.forbidden_tool_violation_rate, 0.0)
        self.assertEqual(metrics.avg_latency_ms, 200)
        self.assertEqual(metrics.total_input_tokens, 20)
        self.assertEqual(metrics.total_output_tokens, 4)

    def test_retry_metrics_count_attempts_and_retries(self):
        results = [
            make_evaluation("one", provider_calls=2, retries=1,
                            throttle_wait_ms=100),
            make_evaluation("two", provider_calls=1, throttle_wait_ms=250),
        ]
        metrics = aggregate_metrics(results)

        self.assertEqual(metrics.total_provider_calls, 3)
        self.assertEqual(metrics.total_retries, 1)
        self.assertAlmostEqual(metrics.retry_rate, 1 / 3)
        self.assertEqual(metrics.total_throttle_wait_ms, 450)

    def test_conditional_success_is_na_without_eligible_applicable_tasks(self):
        result = make_evaluation("one", success=True, provider_error=True)
        metrics = aggregate_metrics([result])

        self.assertIsNone(metrics.task_success_rate)
        self.assertEqual(metrics.raw_task_success_rate, 1.0)
        self.assertEqual(metrics.eligible_executions, 0)
        self.assertIsNone(metrics.conditional_task_success_rate)

    def test_not_applicable_does_not_inflate_rates(self):
        applicable = make_evaluation("one")
        not_applicable = make_evaluation("two", success=False, allowed_tools=False)
        for evaluation in not_applicable.evaluations:
            if evaluation.evaluator in ("task_success", "allowed_tools"):
                evaluation.passed = None
                evaluation.score = None

        metrics = aggregate_metrics([applicable, not_applicable])

        self.assertEqual(metrics.task_success_rate, 1.0)
        self.assertEqual(metrics.allowed_tools_rate, 1.0)

        failed = make_evaluation("three", success=False, allowed_tools=False)
        metrics = aggregate_metrics([applicable, not_applicable, failed])

        self.assertEqual(metrics.task_success_rate, 0.5)
        self.assertEqual(metrics.allowed_tools_rate, 0.5)

        all_na = make_evaluation("four")
        for evaluation in all_na.evaluations:
            evaluation.passed = None
            evaluation.score = None
        metrics = aggregate_metrics([all_na])

        self.assertIsNone(metrics.task_success_rate)
        self.assertIsNone(metrics.allowed_tools_rate)
        self.assertIsNone(metrics.tool_error_rate)

    def test_forbidden_rate_uses_only_restricted_tasks(self):
        unrestricted = make_evaluation(
            "one", forbidden_restriction=False, forbidden_tool_violation=False
        )
        restricted_clean = make_evaluation("two")
        restricted_violation = make_evaluation(
            "three", forbidden_tool_violation=True
        )

        metrics = aggregate_metrics(
            [unrestricted, restricted_clean, restricted_violation]
        )

        self.assertEqual(metrics.forbidden_tool_violation_rate, 0.5)

    def test_category_metrics_preserve_not_applicable(self):
        normal = make_evaluation("normal", category="normal")
        unsafe = make_evaluation("unsafe", category="ambiguous_action",
                                 forbidden_tool_violation=True)
        for evaluation in unsafe.evaluations:
            if evaluation.evaluator == "task_success":
                evaluation.passed = None
                evaluation.score = None

        by_category = {
            metric.category: metric
            for metric in aggregate_by_category([normal, unsafe])
        }

        self.assertEqual(by_category["normal"].task_success_rate, 1.0)
        self.assertIsNone(by_category["ambiguous_action"].task_success_rate)
        self.assertEqual(
            by_category["ambiguous_action"].forbidden_tool_violation_rate, 1.0
        )

    def test_empty_run_fails(self):
        with self.assertRaisesRegex(ValueError, "empty evaluation run"):
            aggregate_metrics([])

    def test_run_has_unique_id_and_utc_creation_time(self):
        result = make_evaluation("one")
        first = EvaluationRun(
            agent_name="fake", dataset_name="sample", repetitions=1, results=[result]
        )
        second = EvaluationRun(
            agent_name="fake", dataset_name="sample", repetitions=1, results=[result]
        )

        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(first.created_at.utcoffset().total_seconds(), 0)
        self.assertEqual(first.results[0].task.id, "one")

    def test_json_store_round_trip_and_missing_run(self):
        run = EvaluationRun(
            agent_name="fake",
            dataset_name="sample",
            repetitions=2,
            results=[make_evaluation("one"), make_evaluation("one")],
        )
        run.results[0].repetition = 1
        run.results[1].repetition = 2

        with tempfile.TemporaryDirectory() as directory:
            store = JSONRunStore(directory)
            path = store.save(run)
            loaded = store.load(run.run_id)

            self.assertEqual(path, Path(directory) / f"{run.run_id}.json")
            self.assertEqual(loaded, run)
            self.assertNotEqual(loaded.results[0].execution_id,
                                loaded.results[1].execution_id)
            self.assertEqual([item.repetition for item in loaded.results], [1, 2])
            with self.assertRaisesRegex(FileNotFoundError, "Evaluation run not found"):
                store.load("unknown")

    def test_comparison_reports_positive_and_negative_deltas(self):
        baseline = EvaluationRun(
            agent_name="fake-v1",
            dataset_name="sample",
            repetitions=1,
            results=[
                make_evaluation("one", latency_ms=100),
                make_evaluation("two", success=False, provider_error=True,
                                tool_error=True, allowed_tools=False,
                                forbidden_tool_violation=True, latency_ms=200),
            ],
        )
        candidate = EvaluationRun(
            agent_name="fake-v2",
            dataset_name="sample",
            repetitions=1,
            results=[
                make_evaluation("one", latency_ms=250),
                make_evaluation("two", latency_ms=350),
            ],
        )

        comparison = compare_runs(baseline, candidate)

        self.assertEqual(comparison.baseline_run_id, baseline.run_id)
        self.assertEqual(comparison.candidate_run_id, candidate.run_id)
        self.assertEqual(comparison.task_success_rate.delta, 0.0)
        self.assertEqual(comparison.completion_rate.delta, 0.5)
        self.assertEqual(comparison.tool_error_rate.delta, 0.0)
        self.assertEqual(comparison.allowed_tools_rate.delta, 0.0)
        self.assertEqual(comparison.forbidden_tool_violation_rate.delta, 0.0)
        self.assertEqual(comparison.provider_error_rate.delta, -0.5)
        self.assertEqual(comparison.avg_latency_ms.delta, 150)


if __name__ == "__main__":
    unittest.main()
