import tempfile
import unittest
from collections import Counter
from pathlib import Path

from agentprobe.agents.base import BaseAgent
from agentprobe.datasets.loader import load_dataset
from agentprobe.evaluators.pipeline import EvaluatorPipeline
from agentprobe.evaluators.task_success import CompletionEvaluator, TaskSuccessEvaluator
from agentprobe.evaluators.tool_calls import RequiredToolsEvaluator
from agentprobe.models.schemas import (
    AgentResult,
    ExecutionTrace,
    Task,
    TerminationReason,
    ToolCall,
)
from agentprobe.runner.dataset_runner import DatasetRunner


class FakeAgent(BaseAgent):
    def __init__(self):
        self.seen = []

    async def run(self, task: Task) -> AgentResult:
        self.seen.append(task.id)
        return AgentResult(
            output=f"Completed {task.id}",
            trace=ExecutionTrace(
                task_id=task.id,
                termination_reason=TerminationReason.COMPLETED,
                tool_calls=[
                    ToolCall(
                        name="get_order",
                        arguments={"order_id": task.id},
                        result={"order_id": task.id},
                    ),
                ],
            ),
        )


class DatasetRunnerTests(unittest.IsolatedAsyncioTestCase):
    def test_support_v1_has_25_valid_unique_tasks(self):
        tasks = load_dataset("datasets/support_v1.jsonl")

        self.assertEqual(len(tasks), 25)
        self.assertEqual(len({task.id for task in tasks}), 25)
        self.assertEqual(
            Counter(task.category for task in tasks),
            {
                "normal": 5,
                "information_gathering": 5,
                "error_handling": 5,
                "ambiguous_action": 5,
                "tool_use_trap": 5,
            },
        )
        expected_errors = {
            "error_001": "order_not_found",
            "error_002": "customer_not_found",
            "error_003": "refund_exceeds_order_amount",
            "error_004": "order_not_found",
            "error_005": "order_not_found",
        }
        for task in tasks:
            if task.id in expected_errors:
                self.assertEqual(task.expected_tool_errors,
                                 [expected_errors[task.id]])

    def test_load_dataset_and_report_bad_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            path.write_text(
                '\n{"id":"one","input":"Lookup one"}\n'
                '{"id":"two","input":"Lookup two"}\n',
                encoding="utf-8",
            )
            self.assertEqual([task.id for task in load_dataset(str(path))],
                             ["one", "two"])

            path.write_text(
                '{"id":"one","input":"Lookup one"}\n'
                '{"id":"two"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, r"tasks\.jsonl:2"):
                load_dataset(str(path))

    async def test_pipeline_and_runner_evaluate_tasks_in_order(self):
        tasks = [
            Task(
                id=task_id,
                input=f"Lookup {task_id}",
                required_tools=["get_order"],
                expected_output={"order_id": task_id},
            )
            for task_id in ("one", "two")
        ]
        agent = FakeAgent()
        pipeline = EvaluatorPipeline([
            RequiredToolsEvaluator(),
            TaskSuccessEvaluator(),
            CompletionEvaluator(),
        ])
        runner = DatasetRunner(agent=agent, evaluator_pipeline=pipeline)

        results = await runner.run(tasks)

        self.assertEqual(agent.seen, ["one", "two"])
        self.assertEqual([item.task.id for item in results], ["one", "two"])
        self.assertEqual(
            [[evaluation.evaluator for evaluation in item.evaluations]
             for item in results],
            [["required_tools", "task_success", "completion"]] * 2,
        )
        self.assertTrue(all(evaluation.passed
                            for item in results for evaluation in item.evaluations))
        self.assertEqual(results[0].agent_result.trace.task_id, "one")

    async def test_repetitions_run_each_task_three_times(self):
        tasks = [Task(id=task_id, input=task_id) for task_id in ("one", "two")]
        agent = FakeAgent()
        runner = DatasetRunner(agent, EvaluatorPipeline([CompletionEvaluator()]))

        results = await runner.run(tasks, repetitions=3)

        self.assertEqual(len(results), 6)
        self.assertEqual(agent.seen, ["one", "one", "one", "two", "two", "two"])
        self.assertEqual(
            [item.repetition for item in results],
            [1, 2, 3, 1, 2, 3],
        )
        self.assertEqual(len({item.execution_id for item in results}), 6)


if __name__ == "__main__":
    unittest.main()
