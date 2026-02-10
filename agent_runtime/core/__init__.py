"""
Core SDK modules.

Contains runtime-domain types, entities, and orchestration logic.
"""

from .config import MCPServerConfig
from .entities import Agent, Run, RunEvent, Thread
from .models import RunResult, StepResult, ToolCallRecord
from .runtime import AgentRuntime
from .types import (
    JSONPrimitive,
    JSONObject,
    JSONValue,
    MCPTransport,
    ModelConfig,
    ModelConfigV0,
    ModelProvider,
    RunEventPayload,
    RunEventPayloadV0,
    RunEventType,
    RunStatus,
    RunStopReason,
    VersionedConfig,
    build_model_config_v0,
    parse_model_config,
    parse_run_event_payload,
)

__all__ = [
    "AgentRuntime",
    "MCPServerConfig",
    "Agent",
    "Thread",
    "Run",
    "RunEvent",
    "RunResult",
    "StepResult",
    "ToolCallRecord",
    "JSONPrimitive",
    "JSONValue",
    "JSONObject",
    "MCPTransport",
    "ModelProvider",
    "RunStatus",
    "RunStopReason",
    "RunEventType",
    "VersionedConfig",
    "ModelConfigV0",
    "ModelConfig",
    "RunEventPayloadV0",
    "RunEventPayload",
    "parse_model_config",
    "parse_run_event_payload",
    "build_model_config_v0",
]
