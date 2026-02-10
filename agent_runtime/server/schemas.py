"""
API request/response schemas for the runtime server.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..core.config import MCPServerConfig
from ..core.entities import Agent, Run, RunEvent, Thread
from ..core.models import RunResult
from ..core.types import (
    JSONObject,
    JSONValue,
    ModelConfigV0,
    ModelProvider,
    RunEventPayloadV0,
)


class CreateAgentRequest(BaseModel):
    """Payload for creating an agent."""

    name: str
    system_prompt: str = ""
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    max_iterations: int = 10

    # Convenience fields for v0
    model: str = "gpt-4o"
    provider: ModelProvider = "openai"
    temperature: float = 0.0
    api_key: str | None = None
    model_params: dict[str, JSONValue] = Field(default_factory=dict)

    # Preferred explicit field
    model_config_payload: ModelConfigV0 | None = None


class CreateThreadRequest(BaseModel):
    """Payload for creating a thread."""

    metadata: JSONObject = Field(default_factory=dict)


class ExecuteRunRequest(BaseModel):
    """Payload for executing one runtime run."""

    thread_id: str
    agent_id: str
    user_message: str


class AgentResponse(BaseModel):
    """Serialized Agent entity for API responses."""

    id: str
    name: str
    model: ModelConfigV0
    system_prompt: str
    mcp_servers: list[MCPServerConfig]
    max_iterations: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_entity(cls, agent: Agent) -> "AgentResponse":
        return cls.model_validate(agent.to_dict())


class ThreadResponse(BaseModel):
    """Serialized Thread entity for API responses."""

    id: str
    metadata: JSONObject = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_entity(cls, thread: Thread) -> "ThreadResponse":
        return cls.model_validate(thread.to_dict())


class RunResponse(BaseModel):
    """Serialized Run entity for API responses."""

    id: str
    thread_id: str
    agent_id: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    iterations: int
    stop_reason: str | None = None
    error: str | None = None
    mcp_servers_used: list[str] | None = None
    context: JSONObject = Field(default_factory=dict)

    @classmethod
    def from_entity(cls, run: Run) -> "RunResponse":
        return cls.model_validate(run.to_dict())


class RunEventResponse(BaseModel):
    """Serialized RunEvent entity for API responses."""

    id: str
    run_id: str
    seq: int
    type: str
    payload: RunEventPayloadV0
    created_at: datetime

    @classmethod
    def from_entity(cls, event: RunEvent) -> "RunEventResponse":
        return cls.model_validate(event.to_dict())


class RunResultResponse(BaseModel):
    """Serialized run result returned by the `run` endpoint."""

    output: str
    run_id: str
    thread_id: str
    iterations: int
    stop_reason: str | None
    steps: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_domain(cls, result: RunResult) -> "RunResultResponse":
        return cls(
            output=result.output,
            run_id=result.run_id,
            thread_id=result.thread_id,
            iterations=result.iterations,
            stop_reason=result.stop_reason,
            steps=[asdict(step) for step in result.steps],
        )
