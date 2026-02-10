"""
Configuration models for MCP server connections.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server connection."""

    name: str = Field(description="Human-readable server identifier")
    transport: str = Field(
        description="Transport type: 'stdio' or 'sse'"
    )  # Literal["stdio", "sse"] kept as str for flexibility

    # stdio transport fields
    command: str | None = Field(
        default=None, description="Command to run for stdio transport (e.g. 'uvx')"
    )
    args: list[str] = Field(
        default_factory=list,
        description="Arguments for stdio command (e.g. ['mcp-server-weather'])",
    )

    # SSE transport fields
    url: str | None = Field(
        default=None, description="URL for SSE transport"
    )

    # Common fields
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables passed to the server process",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this server is enabled. v0: always True, exists for future per-run overrides.",
    )
