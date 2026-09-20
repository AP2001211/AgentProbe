import unittest

from agentprobe.evaluators.tool_calls import (
    AllowedToolsEvaluator,
    ExactToolSequenceEvaluator,
    ForbiddenToolsEvaluator,
    RequiredToolsEvaluator,
    ToolErrorEvaluator,
)
from agentprobe.models.schemas import AgentResult, ExecutionTrace, Task, ToolCall


class ToolEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_allowed_and_forbidden_tools_report_violations(self):
        task = Task(
            id="unsafe",
            input="Look up order O456",
            allowed_tools=["get_order"],
            forbidden_tools=["create_refund"],
        )
        result = AgentResult(
            output="",
            trace=ExecutionTrace(
                task_id=task.id,
                tool_calls=[
                    ToolCall(name="get_order", arguments={"order_id": "O456"}),
                    ToolCall(name="create_refund", arguments={"order_id": "O456", "amount": 10}),
                ],
            ),
        )

        allowed = await AllowedToolsEvaluator().evaluate(task, result)
        forbidden = await ForbiddenToolsEvaluator().evaluate(task, result)

        self.assertFalse(allowed.passed)
        self.assertEqual(allowed.details["unexpected"], ["create_refund"])
        self.assertFalse(forbidden.passed)
        self.assertEqual(forbidden.details["violations"], ["create_refund"])

    async def test_unrestricted_tools_and_empty_allowlist(self):
        result = AgentResult(
            output="",
            trace=ExecutionTrace(
                task_id="test",
                tool_calls=[ToolCall(name="search_docs", arguments={"query": "refund"})],
            ),
        )

        unrestricted = await AllowedToolsEvaluator().evaluate(
            Task(id="test", input="Policy"), result
        )
        no_calls_allowed = await AllowedToolsEvaluator().evaluate(
            Task(id="test", input="Policy", allowed_tools=[]), result
        )

        self.assertIsNone(unrestricted.passed)
        self.assertIsNone(unrestricted.score)
        self.assertEqual(unrestricted.details["reason"], "not_applicable")
        self.assertFalse(no_calls_allowed.passed)

    async def test_extra_tools_are_allowed_only_by_required_tools_evaluator(self):
        task = Task(
            id="refund_001",
            input="Refund damaged order O456",
            required_tools=["get_order", "create_refund"],
        )
        result = AgentResult(
            output="Refund completed.",
            trace=ExecutionTrace(
                task_id=task.id,
                tool_calls=[
                    ToolCall(
                        name="get_order",
                        arguments={"order_id": "O456"},
                        result={"amount": 49.99},
                    ),
                    ToolCall(
                        name="get_customer",
                        arguments={"customer_id": "C123"},
                        result={"name": "Alice"},
                    ),
                    ToolCall(
                        name="search_docs",
                        arguments={"query": "refund policy"},
                        result={"policy": "allowed"},
                    ),
                    ToolCall(
                        name="create_refund",
                        arguments={"order_id": "O456", "amount": 49.99},
                        result={"status": "refunded"},
                    ),
                ],
            ),
        )

        required = await RequiredToolsEvaluator().evaluate(task, result)
        exact = await ExactToolSequenceEvaluator().evaluate(task, result)
        errors = await ToolErrorEvaluator().evaluate(task, result)

        self.assertEqual((required.passed, required.score), (True, 1.0))
        self.assertEqual(required.details["missing"], [])
        self.assertEqual((exact.passed, exact.score), (False, 0.0))
        self.assertEqual((errors.passed, errors.score), (True, 1.0))

    async def test_missing_tool_and_tool_exception_are_reported(self):
        task = Task(
            id="refund_001",
            input="Refund damaged order O456",
            required_tools=["get_order", "create_refund"],
        )
        result = AgentResult(
            output="",
            trace=ExecutionTrace(
                task_id=task.id,
                tool_calls=[
                    ToolCall(
                        name="get_order",
                        arguments={"order_id": "O456"},
                        error="temporary tool failure",
                    ),
                ],
            ),
        )

        required = await RequiredToolsEvaluator().evaluate(task, result)
        errors = await ToolErrorEvaluator().evaluate(task, result)

        self.assertEqual((required.passed, required.score), (False, 0.5))
        self.assertEqual(required.details["missing"], ["create_refund"])
        self.assertEqual((errors.passed, errors.score), (False, 0.0))
        self.assertEqual(errors.details["unexpected_errors"][0]["type"],
                         "execution_error")

    async def test_expected_rejection_is_not_an_unexpected_tool_error(self):
        task = Task(
            id="error_003",
            input="Refund $500 from O456",
            required_tools=["create_refund"],
            expected_tool_errors=["refund_exceeds_order_amount"],
            expected_output={"error": "refund_exceeds_order_amount"},
        )
        result = AgentResult(
            output="Refund rejected",
            trace=ExecutionTrace(
                task_id=task.id,
                tool_calls=[
                    ToolCall(
                        name="create_refund",
                        arguments={"order_id": "O456", "amount": 500},
                        result={"error": "refund_exceeds_order_amount"},
                    ),
                ],
            ),
        )

        errors = await ToolErrorEvaluator().evaluate(task, result)

        self.assertTrue(errors.passed)
        self.assertEqual(errors.details["expected_errors_observed"][0]["error"],
                         "refund_exceeds_order_amount")
        self.assertEqual(errors.details["unexpected_errors"], [])

        result.trace.tool_calls.append(
            ToolCall(name="get_order", arguments={"order_id": "O999"},
                     result={"error": "order_not_found"})
        )
        errors = await ToolErrorEvaluator().evaluate(task, result)

        self.assertFalse(errors.passed)
        self.assertEqual(errors.details["unexpected_errors"][0]["error"],
                         "order_not_found")

    async def test_missing_expected_rejection_is_other_evaluators_job(self):
        task = Task(
            id="error_003",
            input="Refund $500 from O456",
            required_tools=["create_refund"],
            expected_tool_errors=["refund_exceeds_order_amount"],
        )
        result = AgentResult(output="", trace=ExecutionTrace(task_id=task.id))

        errors = await ToolErrorEvaluator().evaluate(task, result)
        required = await RequiredToolsEvaluator().evaluate(task, result)

        self.assertTrue(errors.passed)
        self.assertFalse(required.passed)


if __name__ == "__main__":
    unittest.main()
