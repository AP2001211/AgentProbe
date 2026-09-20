from abc import ABC, abstractmethod

from agentprobe.models.schemas import AgentResult, EvaluationResult, Task


class BaseEvaluator(ABC):
    @abstractmethod
    async def evaluate(
        self,
        task: Task,
        result: AgentResult,
    ) -> EvaluationResult:
        pass
