"""
Core SDK modules.

Contains runtime-domain types, entities, and orchestration logic.
"""

from .config import MCPServerConfig
from .entities import Agent, Run, Thread
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
    RunStatus,
    RunStopReason,
    VersionedConfig,
    build_model_config_v0,
    parse_model_config,
)

__all__ = [
    "AgentRuntime",
    "MCPServerConfig",
    "Agent",
    "Thread",
    "Run",
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
    "VersionedConfig",
    "ModelConfigV0",
    "ModelConfig",
    "parse_model_config",
    "build_model_config_v0",
]
