import asyncio
import unittest

from pydantic import ValidationError

from agentprobe.models.schemas import EvaluationConfig
from agentprobe.runner.rate_limiter import RateLimiter


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        await asyncio.sleep(0)
        self.now += seconds


class RateLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_request_is_immediate_and_next_waits(self):
        fake = FakeTime()
        limiter = RateLimiter(5, clock=fake.monotonic, sleep=fake.sleep)

        first_wait = await limiter.acquire()
        second_wait = await limiter.acquire()

        self.assertEqual(first_wait, 0)
        self.assertEqual(second_wait, 12000)
        self.assertEqual(fake.sleeps, [12.0])

    async def test_request_far_enough_apart_does_not_wait(self):
        fake = FakeTime()
        limiter = RateLimiter(5, clock=fake.monotonic, sleep=fake.sleep)
        await limiter.acquire()
        fake.now = 13.0

        wait = await limiter.acquire()

        self.assertEqual(wait, 0)
        self.assertEqual(fake.sleeps, [])

    async def test_concurrent_callers_are_serialized(self):
        fake = FakeTime()
        limiter = RateLimiter(5, clock=fake.monotonic, sleep=fake.sleep)

        waits = await asyncio.gather(
            limiter.acquire(), limiter.acquire(), limiter.acquire()
        )

        self.assertEqual(waits, [0, 12000, 24000])
        self.assertEqual(fake.sleeps, [12.0, 12.0])
        self.assertEqual(fake.now, 24.0)

    async def test_invalid_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            RateLimiter(0)

    async def test_benchmark_configuration_validates_limits(self):
        config = EvaluationConfig(repetitions=1, requests_per_minute=5)
        self.assertEqual(config.requests_per_minute, 5)
        self.assertIsNone(EvaluationConfig().requests_per_minute)
        with self.assertRaises(ValidationError):
            EvaluationConfig(requests_per_minute=0)


if __name__ == "__main__":
    unittest.main()
