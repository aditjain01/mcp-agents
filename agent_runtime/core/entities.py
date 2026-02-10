"""
Core entities: Agent, Thread, Run.

These are the three pillars of the data model:
- Agent  = persistent configuration (created once, reused)
- Thread = message history (a conversation, can be resumed)
- Run    = one execution of the agentic loop (execution record)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .config import MCPServerConfig
from .types import JSONObject, ModelConfig, ModelConfigV0, RunStatus, RunStopReason


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Agent -- persistent configuration
# ---------------------------------------------------------------------------


class Agent(BaseModel):
    """
    Persistent agent configuration. Created once, stored, reused across
    many Threads and Runs. Similar to an OpenAI Assistant.
    """

    id: str = Field(default_factory=_new_id)
    name: str
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfigV0(model="gpt-4o"),
        description=(
            "Versioned model configuration blob. Includes provider/model/config "
            "so it can be persisted as one typed JSON field."
        ),
    )
    system_prompt: str = ""
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    max_iterations: int = 10
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @property
    def provider(self) -> str:
        """Convenience accessor for provider within model config."""
        return self.model.provider

    @property
    def model_name(self) -> str:
        """Convenience accessor for model identifier within model config."""
        return self.model.model

    @property
    def temperature(self) -> float:
        """Convenience accessor for temperature from model config."""
        value = self.model.config.get("temperature", 0.0)
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    @property
    def api_key(self) -> str | None:
        """Convenience accessor for API key from model config."""
        value = self.model.config.get("api_key")
        if isinstance(value, str):
            return value
        return None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        # Backward compatibility for older payloads where model/provider/temperature
        # were top-level Agent fields.
        payload = dict(data)
        if isinstance(payload.get("model"), str):
            model_name = payload.pop("model")
            provider = payload.pop("provider", "openai")
            temperature = payload.pop("temperature", 0.0)
            api_key = payload.pop("api_key", None)

            model_payload: dict[str, Any] = {
                "version": "v0",
                "provider": provider,
                "model": model_name,
                "config": {
                    "temperature": temperature,
                },
            }
            if api_key is not None:
                model_payload["config"]["api_key"] = api_key
            payload["model"] = model_payload

        return cls.model_validate(payload)


# ---------------------------------------------------------------------------
# Thread -- message history
# ---------------------------------------------------------------------------


class Thread(BaseModel):
    """
    A conversation (ordered list of messages). Independent of any Agent.
    Can be used with different Agents by creating Runs that specify both
    thread_id and agent_id.

    Messages are stored as LangChain BaseMessage objects but serialized
    to/from dicts for persistence.
    """

    id: str = Field(default_factory=_new_id)
    metadata: JSONObject = Field(
        default_factory=dict,
        description="Open-ended context (e.g. user info, session tags)",
    )
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    # Messages are stored outside the pydantic model for flexibility
    # (BaseMessage isn't a plain pydantic model). We keep a private list
    # and expose helpers.
    _messages: list[BaseMessage] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        messages = data.pop("messages", [])
        super().__init__(**data)
        object.__setattr__(self, "_messages", list(messages))

    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["messages"] = messages_to_dict(self._messages)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Thread:
        raw_messages = data.pop("messages", [])
        thread = cls(**data)
        if raw_messages:
            restored = messages_from_dict(raw_messages)
            object.__setattr__(thread, "_messages", list(restored))
        return thread


# ---------------------------------------------------------------------------
# Run -- one execution of the agentic loop
# ---------------------------------------------------------------------------


class Run(BaseModel):
    """
    Records what happened during a single invocation of the agentic loop.
    One Thread can have many Runs (that's a conversation).
    """

    id: str = Field(default_factory=_new_id)
    thread_id: str
    agent_id: str
    status: RunStatus = "queued"
    started_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None
    iterations: int = 0
    stop_reason: RunStopReason | None = None
    error: str | None = None
    mcp_servers_used: list[str] | None = Field(
        default=None,
        description="Which MCP servers were active. v0: all from Agent.",
    )
    context: JSONObject = Field(
        default_factory=dict,
        description="Run-specific params or overrides (extensible for future).",
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        return cls.model_validate(data)
