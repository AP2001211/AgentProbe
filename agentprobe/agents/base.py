from abc import ABC, abstractmethod

from agentprobe.models.schemas import AgentResult, Task


class BaseAgent(ABC):
    @abstractmethod
    async def run(self, task: Task) -> AgentResult:
        """Execute one task and return its result and execution trace."""
        pass
