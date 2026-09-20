import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agentprobe.agents.gemini_agent import GeminiAgent
from agentprobe.models.schemas import RetryConfig, Task, TerminationReason


class FakeChat:
    def __init__(self, responses):
        self.responses = iter(responses)

    def send_message(self, message):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def make_response(function_calls=None, text="", input_tokens=5, output_tokens=2):
    return SimpleNamespace(
        function_calls=function_calls or [],
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=input_tokens,
            candidates_token_count=output_tokens,
        ),
    )


def make_agent(chat, retry_config=None, rate_limiter=None):
    agent = object.__new__(GeminiAgent)
    agent.model = "fake"
    agent.retry_config = retry_config or RetryConfig(max_retries=0)
    agent.rate_limiter = rate_limiter
    agent.tools = []
    agent.client = SimpleNamespace(chats=SimpleNamespace(create=lambda **kwargs: chat))
    return agent


class GeminiTracingTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_tool_round_acquires_limiter(self):
        class CountingLimiter:
            def __init__(self):
                self.calls = 0

            async def acquire(self):
                self.calls += 1
                return 0.0

        get_order_call = SimpleNamespace(name="get_order", args={"order_id": "O456"})
        limiter = CountingLimiter()
        agent = make_agent(
            FakeChat([
                make_response(function_calls=[get_order_call]),
                make_response(text="Order found"),
            ]),
            rate_limiter=limiter,
        )

        result = await agent.run(Task(id="test", input="Look up O456"))

        self.assertEqual(limiter.calls, 2)
        self.assertEqual([call.round for call in result.trace.provider_calls], [1, 2])
        self.assertEqual(result.trace.termination_reason, TerminationReason.COMPLETED)

    async def test_retry_reacquires_limiter_and_records_waits(self):
        class FakeLimiter:
            def __init__(self):
                self.calls = 0

            async def acquire(self):
                self.calls += 1
                return self.calls * 1000.0

        limiter = FakeLimiter()
        agent = make_agent(
            FakeChat([RuntimeError("503 UNAVAILABLE"), make_response(text="Done")]),
            RetryConfig(max_retries=1, initial_backoff_seconds=0),
            rate_limiter=limiter,
        )

        result = await agent.run(Task(id="test", input="Hello"))

        self.assertEqual(limiter.calls, 2)
        self.assertEqual(result.trace.retries, 1)
        self.assertEqual(
            [call.throttle_wait_ms for call in result.trace.provider_calls],
            [1000.0, 2000.0],
        )
        self.assertEqual(result.trace.termination_reason, TerminationReason.COMPLETED)

    async def test_backoff_doubles_and_respects_cap(self):
        chat = FakeChat([
            RuntimeError("503 UNAVAILABLE"),
            RuntimeError("503 UNAVAILABLE"),
            RuntimeError("503 UNAVAILABLE"),
            make_response(text="Done"),
        ])
        agent = make_agent(chat, RetryConfig(
            max_retries=3, initial_backoff_seconds=2, max_backoff_seconds=3,
        ))

        with patch("agentprobe.agents.gemini_agent.asyncio.sleep",
                   new_callable=AsyncMock) as sleep:
            result = await agent.run(Task(id="test", input="Hello"))

        self.assertIsNone(result.error)
        self.assertEqual([call.args[0] for call in sleep.call_args_list],
                         [2, 3, 3])
        self.assertEqual(result.trace.retries, 3)

    async def test_daily_quota_error_is_not_retried(self):
        agent = make_agent(
            FakeChat([RuntimeError("429 RESOURCE_EXHAUSTED quotaId=RequestsPerDay")]),
            RetryConfig(max_retries=3, initial_backoff_seconds=0),
        )

        result = await agent.run(Task(id="test", input="Hello"))

        self.assertEqual(result.trace.retries, 0)
        self.assertEqual(len(result.trace.provider_calls), 1)

    async def test_retryable_503_records_each_attempt_then_succeeds(self):
        chat = FakeChat([RuntimeError("503 UNAVAILABLE"), make_response(text="Done")])
        agent = make_agent(chat, RetryConfig(max_retries=2,
                                             initial_backoff_seconds=0,
                                             max_backoff_seconds=0))

        result = await agent.run(Task(id="test", input="Hello"))

        self.assertIsNone(result.error)
        self.assertEqual(result.trace.termination_reason, TerminationReason.COMPLETED)
        self.assertEqual(result.trace.retries, 1)
        self.assertEqual([(call.round, call.attempt)
                          for call in result.trace.provider_calls], [(1, 1), (1, 2)])
        self.assertIn("503", result.trace.provider_calls[0].error)
        self.assertIsNone(result.trace.provider_calls[1].error)
        self.assertEqual((result.trace.input_tokens, result.trace.output_tokens),
                         (5, 2))

    async def test_retryable_429_stops_at_limit(self):
        chat = FakeChat([RuntimeError("429 RESOURCE_EXHAUSTED") for _ in range(3)])
        agent = make_agent(chat, RetryConfig(max_retries=2,
                                             initial_backoff_seconds=0,
                                             max_backoff_seconds=0))

        result = await agent.run(Task(id="test", input="Hello"))

        self.assertEqual(result.trace.termination_reason, TerminationReason.PROVIDER_ERROR)
        self.assertEqual(result.trace.retries, 2)
        self.assertEqual([call.attempt for call in result.trace.provider_calls],
                         [1, 2, 3])
        self.assertEqual([call.round for call in result.trace.provider_calls],
                         [1, 1, 1])

    async def test_nonretryable_error_has_one_attempt(self):
        agent = make_agent(
            FakeChat([RuntimeError("400 INVALID_ARGUMENT")]),
            RetryConfig(max_retries=3, initial_backoff_seconds=0),
        )

        result = await agent.run(Task(id="test", input="Hello"))

        self.assertEqual(result.trace.retries, 0)
        self.assertEqual(len(result.trace.provider_calls), 1)
        self.assertEqual(result.trace.termination_reason, TerminationReason.PROVIDER_ERROR)

    async def test_completed_run_records_provider_usage(self):
        chat = FakeChat([make_response(text="Done", input_tokens=7, output_tokens=3)])
        result = await make_agent(chat).run(Task(id="test", input="Hello"))

        self.assertIsNone(result.error)
        self.assertEqual(result.output, "Done")
        self.assertEqual(result.trace.termination_reason, TerminationReason.COMPLETED)
        self.assertEqual(len(result.trace.provider_calls), 1)
        self.assertEqual(result.trace.provider_calls[0].round, 1)
        self.assertEqual(result.trace.provider_calls[0].throttle_wait_ms, 0)
        self.assertEqual((result.trace.input_tokens, result.trace.output_tokens), (7, 3))

    async def test_provider_error_preserves_completed_mock_refund(self):
        refund_call = SimpleNamespace(
            name="create_refund",
            args={"order_id": "O456", "amount": 49.99},
        )
        chat = FakeChat([make_response(function_calls=[refund_call]), RuntimeError("503")])
        result = await make_agent(chat).run(Task(id="refund_001", input="Refund O456"))

        self.assertEqual(result.trace.termination_reason, TerminationReason.PROVIDER_ERROR)
        self.assertEqual([call.round for call in result.trace.provider_calls], [1, 2])
        self.assertEqual(result.trace.provider_calls[-1].error, "503")
        self.assertEqual(result.trace.tool_calls[0].result["status"], "refunded")
        self.assertEqual(result.trace.input_tokens, 5)
        self.assertEqual(result.trace.retries, 0)

    async def test_max_steps_is_distinct_from_provider_error(self):
        get_order_call = SimpleNamespace(name="get_order", args={"order_id": "O456"})
        chat = FakeChat([make_response(function_calls=[get_order_call]) for _ in range(11)])
        result = await make_agent(chat).run(Task(id="test", input="Look up O456"))

        self.assertEqual(result.trace.termination_reason, TerminationReason.MAX_STEPS)
        self.assertEqual(len(result.trace.provider_calls), 11)
        self.assertEqual(len(result.trace.tool_calls), 10)
        self.assertIn("maximum steps", result.error)


if __name__ == "__main__":
    unittest.main()
