"""
Agent Runtime SDK - Run agentic loops with MCP tool integration.

Public API exports:
- AgentRuntime: Main orchestrator
- Agent, Thread, Run: Core entities
- MCPServerConfig: MCP server configuration
- Store, InMemoryStore: Persistence layer
- RunResult, StepResult: Result objects
"""

from .core import (
    MCPServerConfig,
    Agent,
    Run,
    Thread,
    RunResult,
    StepResult,
    ToolCallRecord,
    AgentRuntime,
    ModelConfig,
    ModelConfigV0,
)
from .server import app, create_app
from .stores import InMemoryStore, SQLAlchemyStore, Store

__all__ = [
    # Main runtime
    "AgentRuntime",
    # FastAPI server
    "app",
    "create_app",
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
