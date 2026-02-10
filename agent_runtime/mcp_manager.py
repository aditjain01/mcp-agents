"""
Backward-compatible re-export for MCP manager.
"""

from .mcp.manager import MCPManager, run_async

__all__ = ["MCPManager", "run_async"]

