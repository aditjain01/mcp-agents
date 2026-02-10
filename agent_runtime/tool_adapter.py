"""
Adapter to convert MCP tools to LangChain-compatible tool format.

MCP tools have a specific schema format. LangChain's ChatModel.bind_tools()
expects tools in a specific format. This adapter bridges the two.
"""

from __future__ import annotations

from typing import Any, Callable


class ToolAdapter:
    """
    Converts MCP tool definitions to LangChain-compatible tool schemas.
    
    Also creates callables that route execution back through the MCP manager.
    """

    def __init__(self, mcp_manager: Any) -> None:
        """
        Args:
            mcp_manager: MCPManager instance for routing tool calls
        """
        self.mcp_manager = mcp_manager

    def convert_tools(self, mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert MCP tool definitions to LangChain tool schema format.
        
        LangChain expects tools in OpenAI function calling format:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "...",
                "parameters": { ... JSON schema ... }
            }
        }
        
        MCP tools already have 'name', 'description', and 'inputSchema'.
        """
        langchain_tools = []
        
        for tool in mcp_tools:
            langchain_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
                }
            }
            langchain_tools.append(langchain_tool)
        
        return langchain_tools

    def create_tool_executor(self) -> Callable[[str, dict[str, Any]], Any]:
        """
        Create a callable that executes tools via the MCP manager.
        
        This is used by the runtime to execute tool calls returned by the model.
        """
        async def execute_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
            return await self.mcp_manager.call_tool(tool_name, arguments)
        
        return execute_tool
