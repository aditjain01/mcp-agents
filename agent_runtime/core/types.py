"""
Shared type definitions for the runtime.

These are "first-principles" types that we expect to reuse as the runtime grows:
- JSON value aliases for structured payloads stored in DB JSON columns
- versioned model configuration schema for agent model/provider settings
- common literals used across entities and persistence layers
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, TypeAlias

from pydantic import BaseModel, Field, JsonValue

# ---------------------------------------------------------------------------
# Generic JSON typing primitives
# ---------------------------------------------------------------------------

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JsonValue
JSONObject: TypeAlias = dict[str, JsonValue]


# ---------------------------------------------------------------------------
# Common runtime literals
# ---------------------------------------------------------------------------

MCPTransport: TypeAlias = Literal["stdio", "sse"]
ModelProvider: TypeAlias = Literal["openai", "anthropic"]
RunStatus: TypeAlias = Literal["queued", "running", "completed", "failed", "cancelled"]
RunStopReason: TypeAlias = Literal["end_turn", "max_iterations", "error", "cancelled"]
RunEventType: TypeAlias = Literal[
    "run.queued",
    "run.started",
    "run.cancel_requested",
    "run.cancelled",
    "run.failed",
    "run.completed",
    "step.started",
    "step.completed",
    "model.completed",
    "tool.started",
    "tool.completed",
]


# ---------------------------------------------------------------------------
# Versioned model configuration
# ---------------------------------------------------------------------------


class VersionedConfig(BaseModel):
    """Base class for payloads that are persisted as versioned JSON blobs."""

    version: str


class ModelConfigV0(VersionedConfig):
    """
    v0 schema for model configuration.

    We store model/provider/config in one versioned JSON blob so we can evolve
    the shape later without breaking existing rows.
    """

    version: Literal["v0"] = "v0"
    provider: ModelProvider = "openai"
    model: str = Field(description="Provider model identifier, e.g. 'gpt-4o'")
    config: JSONObject = Field(
        default_factory=dict,
        description="Provider-specific model kwargs (temperature, api_key, etc.)",
    )


ModelConfig: TypeAlias = ModelConfigV0


class RunEventPayloadV0(VersionedConfig):
    """
    v0 schema for run-event payloads persisted as JSON.

    The payload shape can evolve over time while remaining backward-compatible
    by branching on `version`.
    """

    version: Literal["v0"] = "v0"
    data: JSONObject = Field(default_factory=dict)


RunEventPayload: TypeAlias = RunEventPayloadV0


def parse_model_config(value: ModelConfig | Mapping[str, Any]) -> ModelConfig:
    """Parse a dict-like payload into a concrete versioned ModelConfig object."""
    if isinstance(value, ModelConfigV0):
        return value

    version = value.get("version", "v0")
    if version == "v0":
        return ModelConfigV0.model_validate(value)

    raise ValueError(f"Unsupported model config version: {version}")


def parse_run_event_payload(
    value: RunEventPayload | Mapping[str, Any],
) -> RunEventPayload:
    """Parse a dict-like payload into a concrete versioned RunEventPayload."""
    if isinstance(value, RunEventPayloadV0):
        return value

    version = value.get("version", "v0")
    if version == "v0":
        return RunEventPayloadV0.model_validate(value)

    raise ValueError(f"Unsupported run event payload version: {version}")


def build_model_config_v0(
    *,
    model: str,
    provider: ModelProvider = "openai",
    temperature: float = 0.0,
    api_key: str | None = None,
    extra_config: Mapping[str, JSONValue] | None = None,
) -> ModelConfigV0:
    """Create a v0 model config from convenience runtime parameters."""
    config: JSONObject = {"temperature": temperature}
    if api_key is not None:
        config["api_key"] = api_key
    if extra_config:
        config.update(dict(extra_config))

    return ModelConfigV0(
        provider=provider,
        model=model,
        config=config,
    )
