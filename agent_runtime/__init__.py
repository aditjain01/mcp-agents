"""
Agent Runtime SDK - Run agentic loops with MCP tool integration.

Public API exports:
- AgentRuntime: Main orchestrator
- Agent, Thread, Run: Core entities
- MCPServerConfig: MCP server configuration
- Store, InMemoryStore: Persistence layer
- RunResult, StepResult: Result objects
"""

from .config import MCPServerConfig
from .entities import Agent, Run, Thread
from .models import RunResult, StepResult, ToolCallRecord
from .runtime import AgentRuntime
from .sqlalchemy_store import SQLAlchemyStore
from .store import InMemoryStore, Store
from .types import ModelConfig, ModelConfigV0

__all__ = [
    # Main runtime
    "AgentRuntime",
    # Core entities
    "Agent",
    "Thread",
    "Run",
    # Configuration
    "MCPServerConfig",
    # Persistence
    "Store",
    "InMemoryStore",
    "SQLAlchemyStore",
    # Results
    "RunResult",
    "StepResult",
    "ToolCallRecord",
    # Versioned typed config blobs
    "ModelConfig",
    "ModelConfigV0",
]

__version__ = "0.1.0"
