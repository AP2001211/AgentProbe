from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Task(BaseModel):
    id: str
    input: str

    category: str = "general"

    required_tools: list[str] = Field(default_factory=list)
    allowed_tools: list[str] | None = None
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_tool_errors: list[str] = Field(default_factory=list)

    expected_output: dict[str, Any] | None = None


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: Any | None = None
    error: str | None = None


class TerminationReason(str, Enum):
    COMPLETED = "completed"
    PROVIDER_ERROR = "provider_error"
    MAX_STEPS = "max_steps"
    AGENT_ERROR = "agent_error"
    REQUEST_BUDGET_EXCEEDED = "request_budget_exceeded"


class ExecutionStatus(str, Enum):
    ELIGIBLE = "eligible"
    PROVIDER_FAILED = "provider_failed"
    SKIPPED = "skipped"


class ProviderCall(BaseModel):
    round: int
    attempt: int = 1
    latency_ms: float
    throttle_wait_ms: float = 0.0

    input_tokens: int = 0
    output_tokens: int = 0

    error: str | None = None


class ExecutionTrace(BaseModel):
    task_id: str

    tool_calls: list[ToolCall] = Field(default_factory=list)
    provider_calls: list[ProviderCall] = Field(default_factory=list)

    latency_ms: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0

    termination_reason: TerminationReason | None = None
    daily_quota_exhausted: bool = False


class AgentResult(BaseModel):
    output: str
    trace: ExecutionTrace
    error: str | None = None


class EvaluationResult(BaseModel):
    evaluator: str
    passed: bool | None
    score: float | None
    details: dict[str, Any] = Field(default_factory=dict)


class TaskEvaluation(BaseModel):
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    repetition: int = 1
    status: ExecutionStatus = ExecutionStatus.ELIGIBLE
    skip_reason: str | None = None

    task: Task
    agent_result: AgentResult
    evaluations: list[EvaluationResult]


class EvaluationRun(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    agent_name: str
    dataset_name: str
    repetitions: int

    results: list[TaskEvaluation]


class AggregateMetrics(BaseModel):
    total_executions: int

    task_success_rate: float | None
    conditional_task_success_rate: float | None
    eligible_executions: int
    provider_failed_executions: int = 0
    skipped_executions: int = 0
    raw_task_success_rate: float | None = None
    completion_rate: float | None
    required_tools_rate: float | None
    allowed_tools_rate: float | None
    forbidden_tool_violation_rate: float | None
    exact_sequence_rate: float | None
    tool_error_rate: float | None
    provider_error_rate: float | None
    total_provider_calls: int
    total_retries: int
    retry_rate: float | None
    total_throttle_wait_ms: float

    avg_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int


class CategoryMetrics(BaseModel):
    category: str
    executions: int

    task_success_rate: float | None
    completion_rate: float | None
    required_tools_rate: float | None
    allowed_tools_rate: float | None
    forbidden_tool_violation_rate: float | None


class RetryConfig(BaseModel):
    max_retries: int = Field(default=3, ge=0)
    initial_backoff_seconds: float = Field(default=2.0, ge=0)
    max_backoff_seconds: float = Field(default=30.0, ge=0)


class EvaluationConfig(BaseModel):
    repetitions: int = Field(default=1, ge=1)
    requests_per_minute: int | None = Field(default=None, ge=1)
    max_retries: int = Field(default=3, ge=0)
    initial_backoff_seconds: float = Field(default=2.0, ge=0)
    max_backoff_seconds: float = Field(default=30.0, ge=0)


class ProviderBudget(BaseModel):
    requests_per_minute: int | None = Field(default=None, ge=1)
    requests_per_day: int | None = Field(default=None, ge=1)


class BenchmarkEstimate(BaseModel):
    executions: int = Field(ge=0)
    minimum_provider_requests: int = Field(ge=0)
    estimated_provider_requests: int = Field(ge=0)
    configured_daily_budget: int | None = None
    likely_to_exceed_budget: bool
