import asyncio

from agentprobe.agents.gemini_agent import GeminiAgent
from agentprobe.models.schemas import Task


async def main():
    task = Task(
        id="refund_001",
        input=(
            "Customer C123 received damaged order O456. "
            "Please refund the full order."
        ),
        required_tools=["get_order", "create_refund"],
    )

    agent = GeminiAgent()
    result = await agent.run(task)

    print("\n--- RESULT ---")
    print(result)


asyncio.run(main())
