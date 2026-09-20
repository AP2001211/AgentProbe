from agentprobe.models.schemas import BenchmarkEstimate, ProviderBudget


class RequestBudgetExceeded(Exception):
    """The configured client-side request allowance is exhausted."""


class BenchmarkPreflightError(ValueError):
    """The benchmark cannot fit even its minimum requests in the budget."""


class RequestBudget:
    def __init__(self, max_requests: int | None):
        if max_requests is not None and max_requests <= 0:
            raise ValueError("max_requests must be greater than 0")
        self.max_requests = max_requests
        self.used = 0

    @property
    def remaining(self) -> int | None:
        if self.max_requests is None:
            return None
        return max(self.max_requests - self.used, 0)

    def consume(self) -> None:
        if self.remaining == 0:
            raise RequestBudgetExceeded("Configured request budget exhausted")
        self.used += 1


def estimate_benchmark(
    executions: int,
    budget: ProviderBudget,
    used_requests: int = 0,
) -> BenchmarkEstimate:
    if executions < 0 or used_requests < 0:
        raise ValueError("executions and used_requests must be nonnegative")
    minimum = executions
    remaining = (
        None if budget.requests_per_day is None
        else max(budget.requests_per_day - used_requests, 0)
    )
    return BenchmarkEstimate(
        executions=executions,
        minimum_provider_requests=minimum,
        estimated_provider_requests=minimum,
        configured_daily_budget=budget.requests_per_day,
        likely_to_exceed_budget=remaining is not None and minimum > remaining,
    )


def require_preflight(estimate: BenchmarkEstimate) -> None:
    if estimate.likely_to_exceed_budget:
        raise BenchmarkPreflightError(
            f"Minimum {estimate.minimum_provider_requests} provider requests "
            f"exceeds the available configured daily budget "
            f"({estimate.configured_daily_budget})"
        )
