from typing import Any

from agentprobe.models.schemas import ExecutionStatus, TaskEvaluation
from agentprobe.runner.dataset_runner import execution_status


def classify_failures(results: list[TaskEvaluation]) -> dict[str, list[dict[str, Any]]]:
    report: dict[str, list[dict[str, Any]]] = {
        "behavioral_failures": [],
        "provider_failures": [],
        "skipped_executions": [],
    }
    for item in results:
        status = execution_status(item)
        trace = item.agent_result.trace
        base = {
            "task_id": item.task.id,
            "category": item.task.category,
            "execution_id": item.execution_id,
            "repetition": item.repetition,
            "tool_calls": [call.model_dump(mode="json") for call in trace.tool_calls],
            "termination_reason": trace.termination_reason.value if trace.termination_reason else None,
        }
        if status == ExecutionStatus.SKIPPED:
            report["skipped_executions"].append({**base, "reason": item.skip_reason})
        elif status == ExecutionStatus.PROVIDER_FAILED:
            report["provider_failures"].append({
                **base,
                "error": item.agent_result.error,
                "provider_calls": [call.model_dump(mode="json") for call in trace.provider_calls],
            })
        else:
            for evaluation in item.evaluations:
                if evaluation.passed is False:
                    report["behavioral_failures"].append({
                        **base,
                        "evaluator": evaluation.evaluator,
                        "expected": evaluation.details.get("expected", evaluation.details.get("allowed", evaluation.details.get("forbidden"))),
                        "actual": evaluation.details.get("actual"),
                        "details": evaluation.details,
                    })
    return report
