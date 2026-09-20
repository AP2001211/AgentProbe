from agentprobe.evaluators.base import BaseEvaluator
from agentprobe.models.schemas import AgentResult, EvaluationResult, Task


class RequiredToolsEvaluator(BaseEvaluator):
    async def evaluate(
        self,
        task: Task,
        result: AgentResult,
    ) -> EvaluationResult:
        expected = task.required_tools
        actual = [call.name for call in result.trace.tool_calls]
        missing = [tool for tool in expected if tool not in actual]

        if expected:
            matched = len(expected) - len(missing)
            score = matched / len(expected)
        else:
            score = 1.0

        return EvaluationResult(
            evaluator="required_tools",
            passed=len(missing) == 0,
            score=score,
            details={"expected": expected, "actual": actual, "missing": missing},
        )


class ExactToolSequenceEvaluator(BaseEvaluator):
    async def evaluate(
        self,
        task: Task,
        result: AgentResult,
    ) -> EvaluationResult:
        expected = task.required_tools
        actual = [call.name for call in result.trace.tool_calls]
        passed = actual == expected

        return EvaluationResult(
            evaluator="exact_tool_sequence",
            passed=passed,
            score=1.0 if passed else 0.0,
            details={"expected": expected, "actual": actual},
        )


class ForbiddenToolsEvaluator(BaseEvaluator):
    async def evaluate(
        self,
        task: Task,
        result: AgentResult,
    ) -> EvaluationResult:
        if not task.forbidden_tools:
            return EvaluationResult(
                evaluator="forbidden_tools",
                passed=None,
                score=None,
                details={"reason": "not_applicable"},
            )

        actual = [call.name for call in result.trace.tool_calls]
        violations = [tool for tool in actual if tool in task.forbidden_tools]
        passed = len(violations) == 0

        return EvaluationResult(
            evaluator="forbidden_tools",
            passed=passed,
            score=1.0 if passed else 0.0,
            details={
                "forbidden": task.forbidden_tools,
                "actual": actual,
                "violations": violations,
            },
        )


class AllowedToolsEvaluator(BaseEvaluator):
    async def evaluate(
        self,
        task: Task,
        result: AgentResult,
    ) -> EvaluationResult:
        if task.allowed_tools is None:
            return EvaluationResult(
                evaluator="allowed_tools",
                passed=None,
                score=None,
                details={"reason": "not_applicable"},
            )

        actual = [call.name for call in result.trace.tool_calls]
        unexpected = [tool for tool in actual if tool not in task.allowed_tools]
        passed = len(unexpected) == 0

        return EvaluationResult(
            evaluator="allowed_tools",
            passed=passed,
            score=1.0 if passed else 0.0,
            details={
                "allowed": task.allowed_tools,
                "actual": actual,
                "unexpected": unexpected,
            },
        )


class ToolErrorEvaluator(BaseEvaluator):
    async def evaluate(
        self,
        task: Task,
        result: AgentResult,
    ) -> EvaluationResult:
        observed_errors = []

        for call in result.trace.tool_calls:
            if call.error is not None:
                observed_errors.append(
                    {
                        "tool": call.name,
                        "type": "execution_error",
                        "error": call.error,
                    }
                )
            elif isinstance(call.result, dict) and "error" in call.result:
                observed_errors.append(
                    {
                        "tool": call.name,
                        "type": "tool_result_error",
                        "error": call.result["error"],
                    }
                )
        unexpected_errors = [
            error
            for error in observed_errors
            if error["error"] not in task.expected_tool_errors
        ]
        expected_errors_observed = [
            error
            for error in observed_errors
            if error["error"] in task.expected_tool_errors
        ]
        passed = len(unexpected_errors) == 0

        return EvaluationResult(
            evaluator="tool_errors",
            passed=passed,
            score=1.0 if passed else 0.0,
            details={
                "expected_errors": task.expected_tool_errors,
                "observed_errors": observed_errors,
                "expected_errors_observed": expected_errors_observed,
                "unexpected_errors": unexpected_errors,
            },
        )
