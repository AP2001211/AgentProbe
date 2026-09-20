import unittest

from agentprobe.evaluators.task_success import CompletionEvaluator, TaskSuccessEvaluator
from agentprobe.evaluators.tool_calls import (
    ExactToolSequenceEvaluator,
    RequiredToolsEvaluator,
    ToolErrorEvaluator,
)
from agentprobe.models.schemas import (
    AgentResult,
    ExecutionTrace,
    Task,
    TerminationReason,
    ToolCall,
)


class TaskSuccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_refund_success_is_distinct_from_provider_failure(self):
        task = Task(
            id="refund_001",
            input="Refund damaged order O456",
            required_tools=["get_order", "create_refund"],
            expected_output={"status": "refunded"},
        )
        result = AgentResult(
            output="",
            error="503 UNAVAILABLE",
            trace=ExecutionTrace(
                task_id=task.id,
                termination_reason=TerminationReason.PROVIDER_ERROR,
                tool_calls=[
                    ToolCall(
                        name="get_order",
                        arguments={"order_id": "O456"},
                        result={"amount": 49.99},
                    ),
                    ToolCall(
                        name="create_refund",
                        arguments={"order_id": "O456", "amount": 49.99},
                        result={"status": "refunded", "order_id": "O456"},
                    ),
                ],
            ),
        )

        evaluations = [
            await evaluator.evaluate(task, result)
            for evaluator in (
                RequiredToolsEvaluator(),
                ExactToolSequenceEvaluator(),
                ToolErrorEvaluator(),
                TaskSuccessEvaluator(),
                CompletionEvaluator(),
            )
        ]

        self.assertEqual([evaluation.passed for evaluation in evaluations],
                         [True, True, True, True, False])
        self.assertEqual(evaluations[3].score, 1.0)
        self.assertEqual(evaluations[3].details["matches"][0]["tool"],
                         "create_refund")
        self.assertEqual(evaluations[3].details["termination_reason"],
                         "provider_error")
        self.assertEqual(evaluations[4].details["error"], "503 UNAVAILABLE")

    async def test_unmatched_result_fails_task_success(self):
        task = Task(id="refund_001", input="Refund O456", expected_output={"status": "refunded"})
        result = AgentResult(
            output="Refund completed.",
            trace=ExecutionTrace(
                task_id=task.id,
                termination_reason=TerminationReason.COMPLETED,
                tool_calls=[
                    ToolCall(
                        name="create_refund",
                        arguments={"order_id": "O456", "amount": 500},
                        result={"error": "refund_exceeds_order_amount"},
                    ),
                ],
            ),
        )

        success = await TaskSuccessEvaluator().evaluate(task, result)
        completion = await CompletionEvaluator().evaluate(task, result)
        tool_errors = await ToolErrorEvaluator().evaluate(task, result)

        self.assertFalse(success.passed)
        self.assertTrue(completion.passed)
        self.assertFalse(tool_errors.passed)
        self.assertEqual(tool_errors.details["unexpected_errors"][0]["type"],
                         "tool_result_error")

    async def test_no_expected_output_is_not_applicable(self):
        task = Task(id="test", input="Hello")
        result = AgentResult(output="", trace=ExecutionTrace(task_id=task.id))

        evaluation = await TaskSuccessEvaluator().evaluate(task, result)

        self.assertIsNone(evaluation.passed)
        self.assertIsNone(evaluation.score)
        self.assertEqual(evaluation.details["reason"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
