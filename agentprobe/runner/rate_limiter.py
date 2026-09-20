import asyncio
import time
from collections.abc import Awaitable, Callable


class RateLimiter:
    def __init__(
        self,
        requests_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than 0")

        self.min_interval = 60.0 / requests_per_minute
        self._last_request_at: float | None = None
        self._lock = asyncio.Lock()
        self._clock = clock
        self._sleep = sleep

    async def acquire(self) -> float:
        start = self._clock()

        async with self._lock:
            now = self._clock()

            if self._last_request_at is not None:
                elapsed = now - self._last_request_at
                wait = self.min_interval - elapsed

                if wait > 0:
                    await self._sleep(wait)

            self._last_request_at = self._clock()

        return (self._clock() - start) * 1000
