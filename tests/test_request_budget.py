import unittest
from types import SimpleNamespace

from agentprobe.agents.gemini_agent import GeminiAgent
from agentprobe.evaluators.pipeline import EvaluatorPipeline
from agentprobe.evaluators.task_success import CompletionEvaluator
from agentprobe.metrics.aggregator import aggregate_metrics
from agentprobe.metrics.failure_report import classify_failures
from agentprobe.models.schemas import (
    AgentResult, EvaluationResult, ExecutionStatus, ExecutionTrace,
    ProviderBudget, RetryConfig, Task, TaskEvaluation, TerminationReason,
)
from agentprobe.runner.dataset_runner import DatasetRunner
from agentprobe.runner.request_budget import (
    BenchmarkPreflightError, RequestBudget, RequestBudgetExceeded,
    estimate_benchmark, require_preflight,
)


class BudgetTests(unittest.TestCase):
    def test_minimum_preflight_blocks_25_with_budget_20(self):
        estimate = estimate_benchmark(25, ProviderBudget(requests_per_minute=5, requests_per_day=20))
        self.assertEqual(estimate.minimum_provider_requests, 25)
        self.assertEqual(estimate.estimated_provider_requests, 25)
        self.assertTrue(estimate.likely_to_exceed_budget)
        with self.assertRaises(BenchmarkPreflightError):
            require_preflight(estimate)

    def test_preflight_uses_remaining_configured_budget(self):
        budget = ProviderBudget(requests_per_day=20)
        self.assertFalse(estimate_benchmark(3, budget, used_requests=17).likely_to_exceed_budget)
        self.assertTrue(estimate_benchmark(4, budget, used_requests=17).likely_to_exceed_budget)
        self.assertFalse(estimate_benchmark(100, ProviderBudget()).likely_to_exceed_budget)

    def test_request_budget_consumption(self):
        budget = RequestBudget(2)
        self.assertEqual(budget.remaining, 2)
        budget.consume()
        budget.consume()
        self.assertEqual((budget.used, budget.remaining), (2, 0))
        with self.assertRaises(RequestBudgetExceeded):
            budget.consume()
        self.assertEqual(budget.used, 2)

    def test_behavioral_failure_report_is_separate(self):
        item = TaskEvaluation(
            task=Task(id="bad", input="bad", category="normal"),
            agent_result=AgentResult(
                output="", trace=ExecutionTrace(
                    task_id="bad", termination_reason=TerminationReason.COMPLETED,
                ),
            ),
            evaluations=[EvaluationResult(
                evaluator="required_tools", passed=False, score=0,
                details={"expected": ["get_order"], "actual": []},
            )],
        )
        report = classify_failures([item])
        self.assertEqual(len(report["behavioral_failures"]), 1)
        self.assertEqual(report["behavioral_failures"][0]["task_id"], "bad")
        self.assertEqual(report["behavioral_failures"][0]["expected"], ["get_order"])
        self.assertEqual(report["provider_failures"], [])


class FakeAgent:
    def __init__(self, failures):
        self.failures = failures
        self.seen = []

    async def run(self, task):
        self.seen.append(task.id)
        reason, daily = self.failures.get(task.id, (TerminationReason.COMPLETED, False))
        return AgentResult(
            output="done" if reason == TerminationReason.COMPLETED else "",
            error="quota" if reason == TerminationReason.PROVIDER_ERROR else None,
            trace=ExecutionTrace(
                task_id=task.id, termination_reason=reason,
                daily_quota_exhausted=daily,
            ),
        )


class RunnerBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_daily_quota_failure_skips_remaining_without_evaluation(self):
        agent = FakeAgent({"two": (TerminationReason.PROVIDER_ERROR, True)})
        runner = DatasetRunner(agent, EvaluatorPipeline([CompletionEvaluator()]))
        results = await runner.run([Task(id=x, input=x) for x in ("one", "two", "three")], repetitions=2)
        self.assertEqual(agent.seen, ["one", "one", "two"])
        self.assertEqual([r.status for r in results], [
            ExecutionStatus.ELIGIBLE, ExecutionStatus.ELIGIBLE,
            ExecutionStatus.PROVIDER_FAILED, ExecutionStatus.SKIPPED,
            ExecutionStatus.SKIPPED, ExecutionStatus.SKIPPED,
        ])
        self.assertEqual([r.evaluations for r in results[2:]], [[], [], [], []])
        self.assertTrue(all(r.skip_reason == "provider_daily_quota_exhausted" for r in results[3:]))
        metrics = aggregate_metrics(results)
        self.assertEqual((metrics.eligible_executions, metrics.provider_failed_executions, metrics.skipped_executions), (2, 1, 3))
        self.assertEqual(metrics.task_success_rate, None)
        self.assertEqual(metrics.provider_error_rate, 1 / 3)
        self.assertEqual(metrics.completion_rate, 2 / 3)
        report = classify_failures(results)
        self.assertEqual([len(report[k]) for k in report], [0, 1, 3])

    async def test_local_budget_exhaustion_skips_current_and_remaining(self):
        agent = FakeAgent({"two": (TerminationReason.REQUEST_BUDGET_EXCEEDED, False)})
        results = await DatasetRunner(agent, EvaluatorPipeline([CompletionEvaluator()])).run(
            [Task(id=x, input=x) for x in ("one", "two", "three")]
        )
        self.assertEqual(agent.seen, ["one", "two"])
        self.assertEqual([r.status for r in results], [
            ExecutionStatus.ELIGIBLE, ExecutionStatus.SKIPPED, ExecutionStatus.SKIPPED
        ])

    async def test_gemini_retry_consumes_each_attempt_and_daily_error_is_terminal(self):
        class Chat:
            calls = 0
            def send_message(self, message):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("503 UNAVAILABLE")
                raise RuntimeError("429 RESOURCE_EXHAUSTED quotaId=GenerateRequestsPerDayPerProjectPerModel-FreeTier")

        chat = Chat()
        agent = object.__new__(GeminiAgent)
        agent.model = "fake"
        agent.retry_config = RetryConfig(max_retries=3, initial_backoff_seconds=0)
        agent.rate_limiter = None
        agent.request_budget = RequestBudget(2)
        agent.tools = []
        agent.client = SimpleNamespace(chats=SimpleNamespace(create=lambda **kwargs: chat))
        result = await agent.run(Task(id="one", input="one"))
        self.assertEqual((chat.calls, agent.request_budget.used, result.trace.retries), (2, 2, 1))
        self.assertTrue(result.trace.daily_quota_exhausted)
        self.assertEqual(result.trace.termination_reason, TerminationReason.PROVIDER_ERROR)


if __name__ == "__main__":
    unittest.main()
