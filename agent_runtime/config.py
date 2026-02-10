"""
Configuration models for MCP server connections.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, Field
from pydantic import model_validator

from .types import MCPTransport


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server connection."""

    name: str = Field(description="Human-readable server identifier")
    transport: MCPTransport = Field(
        description="Transport type: 'stdio' or 'sse'"
    )

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

    @model_validator(mode="after")
    def validate_transport_fields(self) -> Self:
        """Validate required fields for the selected transport."""
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio transport requires 'command'")
        if self.transport == "sse" and not self.url:
            raise ValueError("sse transport requires 'url'")
        return self
