from agentprobe.agents.base import BaseAgent
from agentprobe.evaluators.pipeline import EvaluatorPipeline
from agentprobe.models.schemas import (
    AgentResult, ExecutionStatus, ExecutionTrace, Task, TaskEvaluation,
    TerminationReason,
)


def execution_status(item: TaskEvaluation) -> ExecutionStatus:
    """Also classify runs saved before explicit statuses were introduced."""
    if item.status == ExecutionStatus.SKIPPED:
        return ExecutionStatus.SKIPPED
    reason = item.agent_result.trace.termination_reason
    if reason == TerminationReason.REQUEST_BUDGET_EXCEEDED:
        return ExecutionStatus.SKIPPED
    if reason == TerminationReason.PROVIDER_ERROR:
        return ExecutionStatus.PROVIDER_FAILED
    return ExecutionStatus.ELIGIBLE


class DatasetRunner:
    def __init__(
        self,
        agent: BaseAgent,
        evaluator_pipeline: EvaluatorPipeline,
    ):
        self.agent = agent
        self.evaluator_pipeline = evaluator_pipeline

    async def run_task(
        self,
        task: Task,
        repetition: int = 1,
    ) -> TaskEvaluation:
        result = await self.agent.run(task)
        if result.trace.termination_reason == TerminationReason.REQUEST_BUDGET_EXCEEDED:
            status = ExecutionStatus.SKIPPED
            evaluations = []
            skip_reason = "configured_request_budget_exhausted"
        elif result.trace.termination_reason == TerminationReason.PROVIDER_ERROR:
            status = ExecutionStatus.PROVIDER_FAILED
            evaluations = []
            skip_reason = None
        else:
            status = ExecutionStatus.ELIGIBLE
            evaluations = await self.evaluator_pipeline.evaluate(task, result)
            skip_reason = None

        return TaskEvaluation(
            repetition=repetition,
            status=status,
            skip_reason=skip_reason,
            task=task,
            agent_result=result,
            evaluations=evaluations,
        )

    async def run(
        self,
        tasks: list[Task],
        repetitions: int = 1,
    ) -> list[TaskEvaluation]:
        results = []

        stop_reason = None
        for task in tasks:
            for repetition in range(1, repetitions + 1):
                if stop_reason is not None:
                    results.append(TaskEvaluation(
                        repetition=repetition,
                        status=ExecutionStatus.SKIPPED,
                        skip_reason=stop_reason,
                        task=task,
                        agent_result=AgentResult(
                            output="",
                            trace=ExecutionTrace(task_id=task.id),
                        ),
                        evaluations=[],
                    ))
                    continue
                evaluation = await self.run_task(task, repetition=repetition)
                results.append(evaluation)
                if evaluation.agent_result.trace.daily_quota_exhausted:
                    stop_reason = "provider_daily_quota_exhausted"
                elif evaluation.status == ExecutionStatus.SKIPPED:
                    stop_reason = evaluation.skip_reason

        return results
