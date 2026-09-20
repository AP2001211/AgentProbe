from typing import Any

from agentprobe.evaluators.base import BaseEvaluator
from agentprobe.models.schemas import (
    AgentResult,
    EvaluationResult,
    Task,
    TerminationReason,
)


class TaskSuccessEvaluator(BaseEvaluator):
    async def evaluate(
        self,
        task: Task,
        result: AgentResult,
    ) -> EvaluationResult:
        expected = task.expected_output

        if expected is None:
            return EvaluationResult(
                evaluator="task_success",
                passed=None,
                score=None,
                details={"reason": "not_applicable"},
            )

        matches = []

        for call in result.trace.tool_calls:
            if isinstance(call.result, dict):
                if self._matches_expected(call.result, expected):
                    matches.append({"tool": call.name, "result": call.result})

        passed = len(matches) > 0

        return EvaluationResult(
            evaluator="task_success",
            passed=passed,
            score=1.0 if passed else 0.0,
            details={
                "expected": expected,
                "matches": matches,
                "termination_reason": (
                    result.trace.termination_reason.value
                    if result.trace.termination_reason
                    else None
                ),
            },
        )

    def _matches_expected(
        self,
        actual: dict[str, Any],
        expected: dict[str, Any],
    ) -> bool:
        return all(
            key in actual and actual[key] == value
            for key, value in expected.items()
        )


class CompletionEvaluator(BaseEvaluator):
    async def evaluate(
        self,
        task: Task,
        result: AgentResult,
    ) -> EvaluationResult:
        completed = result.trace.termination_reason == TerminationReason.COMPLETED

        return EvaluationResult(
            evaluator="completion",
            passed=completed,
            score=1.0 if completed else 0.0,
            details={
                "termination_reason": (
                    result.trace.termination_reason.value
                    if result.trace.termination_reason
                    else None
                ),
                "error": result.error,
            },
        )
