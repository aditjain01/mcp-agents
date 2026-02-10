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
from .store import InMemoryStore, Store

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
    # Results
    "RunResult",
    "StepResult",
    "ToolCallRecord",
]

__version__ = "0.1.0"
