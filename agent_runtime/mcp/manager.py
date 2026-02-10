"""
MCP server connection lifecycle management.

Uses the official `mcp` Python client SDK to connect to MCP servers
via stdio or SSE transports, discover tools, and route tool calls.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

from ..core.config import MCPServerConfig


class MCPManager:
    """
    Manages connections to multiple MCP servers.

    Responsibilities:
    - Connect to MCP servers (stdio, SSE)
    - Discover and aggregate tools from all servers
    - Route tool calls to the correct server
    - Clean up connections
    """

    def __init__(self, server_configs: list[MCPServerConfig]) -> None:
        self.server_configs = [s for s in server_configs if s.enabled]
        self._sessions: dict[str, ClientSession] = {}
        self._exit_stack: AsyncExitStack | None = None
        self._tools_cache: dict[str, list[dict[str, Any]]] = {}  # server_name -> tools

    async def connect_all(self) -> None:
        """Connect to all configured MCP servers."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for config in self.server_configs:
            try:
                session = await self._connect_server(config)
                self._sessions[config.name] = session
                # Discover tools from this server
                tools_result = await session.list_tools()
                self._tools_cache[config.name] = [
                    tool.model_dump() for tool in tools_result.tools
                ]
            except Exception as exc:
                print(f"Warning: Failed to connect to MCP server {config.name}: {exc}")

    async def _connect_server(self, config: MCPServerConfig) -> ClientSession:
        """Connect to a single MCP server based on transport type."""
        if config.transport == "stdio":
            params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env or None,
            )
            read, write = await self._exit_stack.enter_async_context(stdio_client(params))
            session = ClientSession(read, write)
            await self._exit_stack.enter_async_context(session)
            await session.initialize()
            return session

        if config.transport == "sse":
            read, write = await self._exit_stack.enter_async_context(
                sse_client(config.url)
            )
            session = ClientSession(read, write)
            await self._exit_stack.enter_async_context(session)
            await session.initialize()
            return session

        raise ValueError(f"Unsupported transport type: {config.transport}")

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Get all tools from all connected servers.

        Returns a list of tool definitions with an added 'mcp_server' field
        to track which server provides each tool.
        """
        all_tools = []
        for server_name, tools in self._tools_cache.items():
            for tool in tools:
                tool_with_server = tool.copy()
                tool_with_server["mcp_server"] = server_name
                all_tools.append(tool_with_server)
        return all_tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Call a tool by name, routing to the correct MCP server.

        Returns the tool result content.
        """
        # Find which server provides this tool
        server_name = None
        for srv_name, tools in self._tools_cache.items():
            if any(t["name"] == tool_name for t in tools):
                server_name = srv_name
                break

        if not server_name:
            raise ValueError(f"Tool '{tool_name}' not found in any connected MCP server")

        session = self._sessions.get(server_name)
        if not session:
            raise RuntimeError(f"MCP server '{server_name}' not connected")

        # Call the tool
        result = await session.call_tool(tool_name, arguments)

        # Extract content from the result
        if result.content:
            # MCP returns a list of content items
            return [item.model_dump() if hasattr(item, "model_dump") else item for item in result.content]
        return None

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        if self._exit_stack:
            await self._exit_stack.__aexit__(None, None, None)
            self._exit_stack = None
        self._sessions.clear()
        self._tools_cache.clear()


def run_async(coro: Any) -> Any:
    """Run an async coroutine in a sync context."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        import nest_asyncio

        nest_asyncio.apply()
        return asyncio.run(coro)
