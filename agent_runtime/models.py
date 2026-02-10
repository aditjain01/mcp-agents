"""
Result models returned by the agentic loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .entities import Run


@dataclass
class ToolCallRecord:
    """Record of a single tool call within a step."""

    tool_name: str
    args: dict[str, Any]
    result: str
    is_error: bool = False


@dataclass
class StepResult:
    """What happened in one iteration of the agentic loop."""

    model_response: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    stop: bool = False  # True if the model didn't request any tool calls


@dataclass
class RunResult:
    """Final result of a complete run."""

    output: str  # The model's final text response
    run_id: str
    thread_id: str
    iterations: int
    stop_reason: str | None
    steps: list[StepResult] = field(default_factory=list)

    @classmethod
    def from_run(cls, run: Run, output: str, steps: list[StepResult]) -> RunResult:
        return cls(
            output=output,
            run_id=run.id,
            thread_id=run.thread_id,
            iterations=run.iterations,
            stop_reason=run.stop_reason,
            steps=steps,
        )
