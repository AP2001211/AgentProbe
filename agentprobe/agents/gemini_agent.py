import asyncio
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from agentprobe.agents.base import BaseAgent
from agentprobe.models.schemas import (
    AgentResult,
    ExecutionTrace,
    ProviderCall,
    RetryConfig,
    Task,
    TerminationReason,
    ToolCall,
)
from agentprobe.runner.rate_limiter import RateLimiter
from agentprobe.runner.request_budget import RequestBudget, RequestBudgetExceeded
from agentprobe.tools.mock_tools import (
    create_refund,
    execute_tool,
    get_customer,
    get_order,
    search_docs,
)


load_dotenv()


class GeminiAgent(BaseAgent):
    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        retry_config: RetryConfig | None = None,
        rate_limiter: RateLimiter | None = None,
        request_budget: RequestBudget | None = None,
    ):
        self.model = model
        self.retry_config = retry_config or RetryConfig()
        self.rate_limiter = rate_limiter
        self.request_budget = request_budget
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.tools = [get_order, get_customer, create_refund, search_docs]

    def _is_retryable(self, error: Exception) -> bool:
        message = str(error).lower()
        if self._is_daily_quota_error(error):
            return False
        return (
            "429" in message
            or "503" in message
            or "resource_exhausted" in message
            or "unavailable" in message
        )

    @staticmethod
    def _is_daily_quota_error(error: Exception) -> bool:
        message = str(error).lower()
        return "429" in message and any(marker in message for marker in (
            "perday", "per_day", "per day", "daily quota", "daily limit"
        ))

    async def _send_message(
        self,
        chat,
        message,
        trace: ExecutionTrace,
        round_number: int,
    ):
        config = self.retry_config

        for attempt in range(config.max_retries + 1):
            request_budget = getattr(self, "request_budget", None)
            if request_budget is not None and request_budget.remaining == 0:
                raise RequestBudgetExceeded("Configured request budget exhausted")
            throttle_wait_ms = 0.0
            if self.rate_limiter is not None:
                throttle_wait_ms = await self.rate_limiter.acquire()
            if request_budget is not None:
                request_budget.consume()

            start = time.perf_counter()

            try:
                response = chat.send_message(message)
                latency_ms = (time.perf_counter() - start) * 1000
                usage = response.usage_metadata

                provider_call = ProviderCall(
                    round=round_number,
                    attempt=attempt + 1,
                    latency_ms=latency_ms,
                    throttle_wait_ms=throttle_wait_ms,
                    input_tokens=(usage.prompt_token_count or 0) if usage else 0,
                    output_tokens=(usage.candidates_token_count or 0) if usage else 0,
                )
                trace.provider_calls.append(provider_call)
                trace.input_tokens += provider_call.input_tokens
                trace.output_tokens += provider_call.output_tokens

                return response

            except Exception as e:
                if self._is_daily_quota_error(e):
                    trace.daily_quota_exhausted = True
                latency_ms = (time.perf_counter() - start) * 1000
                trace.provider_calls.append(
                    ProviderCall(
                        round=round_number,
                        attempt=attempt + 1,
                        latency_ms=latency_ms,
                        throttle_wait_ms=throttle_wait_ms,
                        error=str(e),
                    )
                )

                if not self._is_retryable(e) or attempt >= config.max_retries:
                    raise

                trace.retries += 1
                delay = min(
                    config.initial_backoff_seconds * (2 ** attempt),
                    config.max_backoff_seconds,
                )
                await asyncio.sleep(delay)

    async def run(self, task: Task) -> AgentResult:
        start_time = time.perf_counter()
        trace = ExecutionTrace(task_id=task.id)

        try:
            chat = self.client.chats.create(
                model=self.model,
                config=types.GenerateContentConfig(
                    tools=self.tools,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )

            round_number = 1
            response = await self._send_message(chat, task.input, trace, round_number)
            max_steps = 10

            for _ in range(max_steps):
                function_calls = response.function_calls

                if not function_calls:
                    trace.termination_reason = TerminationReason.COMPLETED
                    trace.latency_ms = (time.perf_counter() - start_time) * 1000
                    return AgentResult(output=response.text or "", trace=trace)

                tool_responses = []

                for function_call in function_calls:
                    name = function_call.name
                    arguments = dict(function_call.args)

                    try:
                        result = execute_tool(name, arguments)
                        trace.tool_calls.append(
                            ToolCall(name=name, arguments=arguments, result=result)
                        )
                        tool_responses.append(
                            types.Part.from_function_response(
                                name=name,
                                response={"result": result},
                            )
                        )
                    except Exception as e:
                        trace.tool_calls.append(
                            ToolCall(name=name, arguments=arguments, error=str(e))
                        )
                        tool_responses.append(
                            types.Part.from_function_response(
                                name=name,
                                response={"error": str(e)},
                            )
                        )

                round_number += 1
                response = await self._send_message(
                    chat, tool_responses, trace, round_number
                )

            trace.termination_reason = TerminationReason.MAX_STEPS
            raise RuntimeError(f"Agent exceeded maximum steps ({max_steps})")

        except Exception as e:
            trace.latency_ms = (time.perf_counter() - start_time) * 1000
            if trace.termination_reason is None:
                if isinstance(e, RequestBudgetExceeded):
                    trace.termination_reason = TerminationReason.REQUEST_BUDGET_EXCEEDED
                elif trace.provider_calls and trace.provider_calls[-1].error:
                    trace.termination_reason = TerminationReason.PROVIDER_ERROR
                else:
                    trace.termination_reason = TerminationReason.AGENT_ERROR
            return AgentResult(output="", trace=trace, error=str(e))
