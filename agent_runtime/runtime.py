"""
AgentRuntime - the main orchestrator for the agentic loop.

Connects agents, threads, and MCP servers to execute agentic workflows.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Mapping

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from .config import MCPServerConfig
from .entities import Agent, Run, Thread
from .mcp_manager import MCPManager
from .models import RunResult, StepResult, ToolCallRecord
from .store import Store
from .types import JSONValue, ModelConfig, ModelProvider, build_model_config_v0, parse_model_config


class AgentRuntime:
    """
    The main entry point for executing agentic workflows.
    
    Provides:
    - Agent and Thread CRUD operations
    - Run execution (the agentic loop)
    """

    def __init__(self, store: Store) -> None:
        self.store = store

    # ---- Agent operations ----

    def create_agent(
        self,
        name: str,
        model: str = "gpt-4o",
        system_prompt: str = "",
        mcp_servers: list[MCPServerConfig] | None = None,
        provider: ModelProvider = "openai",
        max_iterations: int = 10,
        temperature: float = 0.0,
        api_key: str | None = None,
        model_params: Mapping[str, JSONValue] | None = None,
        model_config: ModelConfig | Mapping[str, Any] | None = None,
    ) -> Agent:
        """
        Create and persist an Agent.

        For v0, callers may either:
        - pass `model_config` directly (preferred, explicit)
        - use convenience params (`model`, `provider`, `temperature`, `api_key`)
          which are composed into a v0 ModelConfig blob.
        """
        resolved_model_config = (
            parse_model_config(model_config)
            if model_config is not None
            else build_model_config_v0(
                model=model,
                provider=provider,
                temperature=temperature,
                api_key=api_key,
                extra_config=model_params,
            )
        )

        agent = Agent(
            name=name,
            model=resolved_model_config,
            system_prompt=system_prompt,
            mcp_servers=mcp_servers or [],
            max_iterations=max_iterations,
        )
        self.store.save_agent(agent)
        return agent

    def get_agent(self, agent_id: str) -> Agent | None:
        """Retrieve an Agent by ID."""
        return self.store.get_agent(agent_id)

    def list_agents(self) -> list[Agent]:
        """List all Agents."""
        return self.store.list_agents()

    # ---- Thread operations ----

    def create_thread(self, metadata: dict[str, Any] | None = None) -> Thread:
        """Create and persist a Thread."""
        thread = Thread(metadata=metadata or {})
        self.store.save_thread(thread)
        return thread

    def get_thread(self, thread_id: str) -> Thread | None:
        """Retrieve a Thread by ID."""
        return self.store.get_thread(thread_id)

    def list_threads(self) -> list[Thread]:
        """List all Threads."""
        return self.store.list_threads()

    # ---- Run execution ----

    def run(
        self,
        thread_id: str,
        agent_id: str,
        user_message: str,
    ) -> RunResult:
        """
        Execute the agentic loop.
        
        Args:
            thread_id: Which conversation thread to use
            agent_id: Which agent configuration to use
            user_message: The user's input message
        
        Returns:
            RunResult with the final output and execution metadata
        """
        return asyncio.run(self._run_async(thread_id, agent_id, user_message))

    async def _run_async(
        self,
        thread_id: str,
        agent_id: str,
        user_message: str,
    ) -> RunResult:
        """Async implementation of the agentic loop."""
        # Load thread and agent
        thread = self.store.get_thread(thread_id)
        if not thread:
            raise ValueError(f"Thread {thread_id} not found")

        agent = self.store.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        # Create a Run record
        run = Run(
            thread_id=thread.id,
            agent_id=agent.id,
            status="running",
            mcp_servers_used=[s.name for s in agent.mcp_servers if s.enabled],
        )
        self.store.save_run(run)

        try:
            # Append user message to thread
            thread.messages.append(HumanMessage(content=user_message))

            # Connect to MCP servers and discover tools
            mcp_manager = MCPManager(agent.mcp_servers)
            await mcp_manager.connect_all()

            from .tool_adapter import ToolAdapter
            tool_adapter = ToolAdapter(mcp_manager)
            mcp_tools = mcp_manager.get_tools()
            langchain_tools = tool_adapter.convert_tools(mcp_tools)

            # Create the model
            model = self._create_model(agent)
            if langchain_tools:
                model = model.bind_tools(langchain_tools)

            # Execute the agentic loop
            steps: list[StepResult] = []
            final_output = ""

            for i in range(agent.max_iterations):
                # Prepare messages with system prompt
                messages = []
                if agent.system_prompt:
                    messages.append(SystemMessage(content=agent.system_prompt))
                messages.extend(thread.messages)

                # Call the model
                response: AIMessage = await model.ainvoke(messages)
                thread.messages.append(response)

                # Check if model wants to use tools
                if not response.tool_calls:
                    # Model is done - extract final output
                    final_output = response.content if isinstance(response.content, str) else str(response.content)
                    run.stop_reason = "end_turn"
                    steps.append(StepResult(
                        model_response=final_output,
                        tool_calls=[],
                        stop=True,
                    ))
                    break

                # Execute tool calls
                tool_records = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call.get("args", {})
                    tool_call_id = tool_call.get("id", "")

                    try:
                        result = await mcp_manager.call_tool(tool_name, tool_args)
                        result_str = str(result)
                        
                        thread.messages.append(
                            ToolMessage(
                                content=result_str,
                                tool_call_id=tool_call_id,
                            )
                        )
                        
                        tool_records.append(ToolCallRecord(
                            tool_name=tool_name,
                            args=tool_args,
                            result=result_str,
                            is_error=False,
                        ))
                    except Exception as e:
                        error_msg = f"Error calling tool {tool_name}: {str(e)}"
                        thread.messages.append(
                            ToolMessage(
                                content=error_msg,
                                tool_call_id=tool_call_id,
                            )
                        )
                        tool_records.append(ToolCallRecord(
                            tool_name=tool_name,
                            args=tool_args,
                            result=error_msg,
                            is_error=True,
                        ))

                steps.append(StepResult(
                    model_response=response.content if isinstance(response.content, str) else "",
                    tool_calls=tool_records,
                    stop=False,
                ))
            else:
                # Hit max iterations
                run.stop_reason = "max_iterations"
                final_output = "Maximum iterations reached"

            # Finalize the run
            run.status = "completed"
            run.iterations = len(steps)
            run.completed_at = datetime.now(timezone.utc)

            # Save updated thread and run
            self.store.save_thread(thread)
            self.store.save_run(run)

            # Cleanup MCP connections
            await mcp_manager.disconnect_all()

            return RunResult.from_run(run, final_output, steps)

        except Exception as e:
            # Handle errors
            run.status = "failed"
            run.error = str(e)
            run.stop_reason = "error"
            run.completed_at = datetime.now(timezone.utc)
            self.store.save_run(run)
            raise

    def _create_model(self, agent: Agent) -> ChatOpenAI:
        """Create a LangChain model instance from agent config."""
        model_config = agent.model
        if model_config.version != "v0":
            raise NotImplementedError(
                f"Unsupported model config version '{model_config.version}'"
            )

        if model_config.provider != "openai":
            raise NotImplementedError(
                f"Provider {model_config.provider} not yet implemented. "
                "Only 'openai' is supported in v0."
            )

        raw_kwargs = dict(model_config.config)

        raw_temperature = raw_kwargs.pop("temperature", 0.0)
        if not isinstance(raw_temperature, (int, float)):
            raise ValueError("Model config 'temperature' must be numeric")
        temperature = float(raw_temperature)

        raw_api_key = raw_kwargs.pop("api_key", None)
        if raw_api_key is not None and not isinstance(raw_api_key, str):
            raise ValueError("Model config 'api_key' must be a string when provided")

        api_key = raw_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY env var "
                "or include api_key in agent.model.config."
            )

        return ChatOpenAI(
            model=model_config.model,
            temperature=temperature,
            api_key=api_key,
            **raw_kwargs,
        )
