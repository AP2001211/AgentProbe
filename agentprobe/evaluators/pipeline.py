from agentprobe.evaluators.base import BaseEvaluator
from agentprobe.models.schemas import AgentResult, EvaluationResult, Task


class EvaluatorPipeline:
    def __init__(self, evaluators: list[BaseEvaluator]):
        self.evaluators = evaluators

    async def evaluate(
        self,
        task: Task,
        result: AgentResult,
    ) -> list[EvaluationResult]:
        evaluations = []

        for evaluator in self.evaluators:
            evaluation = await evaluator.evaluate(task, result)
            evaluations.append(evaluation)

        return evaluations
