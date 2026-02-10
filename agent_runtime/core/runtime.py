"""
AgentRuntime - the main orchestrator for the agentic loop.

Connects agents, threads, and MCP servers to execute agentic workflows.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from .config import MCPServerConfig
from .entities import Agent, Run, Thread
from .models import RunResult, StepResult, ToolCallRecord
from .types import JSONValue, ModelConfig, ModelProvider, build_model_config_v0, parse_model_config
from ..mcp.manager import MCPManager
from ..mcp.tool_adapter import ToolAdapter
from ..stores.base import Store


RunEventEmitter = Callable[[str, dict[str, JSONValue]], Awaitable[None] | None]


class RunCancelledError(Exception):
    """Raised when a running loop should stop due to cancellation."""


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

    def create_run(
        self,
        thread_id: str,
        agent_id: str,
        *,
        context: Mapping[str, JSONValue] | None = None,
    ) -> Run:
        """Create a queued Run record that can be executed later."""
        thread = self.store.get_thread(thread_id)
        if not thread:
            raise ValueError(f"Thread {thread_id} not found")

        agent = self.store.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        run = Run(
            thread_id=thread.id,
            agent_id=agent.id,
            status="queued",
            mcp_servers_used=[s.name for s in agent.mcp_servers if s.enabled],
            context=dict(context or {}),
        )
        self.store.save_run(run)
        return run

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
        run = self.create_run(
            thread_id=thread_id,
            agent_id=agent_id,
            context={"user_message": user_message},
        )
        return asyncio.run(self.execute_run(run.id, user_message))

    async def arun(
        self,
        thread_id: str,
        agent_id: str,
        user_message: str,
    ) -> RunResult:
        """Async API for creating and executing a run immediately."""
        run = self.create_run(
            thread_id=thread_id,
            agent_id=agent_id,
            context={"user_message": user_message},
        )
        return await self.execute_run(run.id, user_message)

    async def execute_run(
        self,
        run_id: str,
        user_message: str,
        *,
        event_emitter: RunEventEmitter | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> RunResult:
        """
        Execute an already-created run record.

        This method emits step-level events and checkpoints run/thread state after
        each step, making execution resilient for streaming and replay.
        """
        run = self.store.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")
        if run.status == "cancelled":
            return RunResult.from_run(run, "Run cancelled", [])

        thread = self.store.get_thread(run.thread_id)
        if not thread:
            raise ValueError(f"Thread {run.thread_id} not found")

        agent = self.store.get_agent(run.agent_id)
        if not agent:
            raise ValueError(f"Agent {run.agent_id} not found")

        run.status = "running"
        run.error = None
        run.stop_reason = None
        run.completed_at = None
        run.started_at = self._now()
        run.iterations = 0
        self.store.save_run(run)
        await self._emit_event(
            event_emitter,
            "run.started",
            {
                "run_id": run.id,
                "thread_id": run.thread_id,
                "agent_id": run.agent_id,
            },
        )

        mcp_manager: MCPManager | None = None
        steps: list[StepResult] = []

        try:
            await self._raise_if_cancelled(cancel_event)

            # Append user message to thread
            thread.messages.append(HumanMessage(content=user_message))
            thread.updated_at = self._now()
            self.store.save_thread(thread)

            # Connect to MCP servers and discover tools
            mcp_manager = MCPManager(agent.mcp_servers)
            await mcp_manager.connect_all()
            tool_adapter = ToolAdapter(mcp_manager)
            mcp_tools = mcp_manager.get_tools()
            langchain_tools = tool_adapter.convert_tools(mcp_tools)

            # Create the model
            model = self._create_model(agent)
            if langchain_tools:
                model = model.bind_tools(langchain_tools)

            # Execute the agentic loop
            final_output = ""

            for step_index in range(agent.max_iterations):
                await self._raise_if_cancelled(cancel_event)
                await self._emit_event(
                    event_emitter,
                    "step.started",
                    {
                        "run_id": run.id,
                        "step_index": step_index,
                    },
                )

                # Prepare messages with system prompt
                messages = []
                if agent.system_prompt:
                    messages.append(SystemMessage(content=agent.system_prompt))
                messages.extend(thread.messages)

                # Call the model
                response: AIMessage = await model.ainvoke(messages)
                thread.messages.append(response)
                await self._emit_event(
                    event_emitter,
                    "model.completed",
                    {
                        "run_id": run.id,
                        "step_index": step_index,
                        "content": self._as_json_value(response.content),
                        "tool_calls_count": len(response.tool_calls),
                    },
                )

                # Check if model wants to use tools
                if not response.tool_calls:
                    # Model is done - extract final output
                    final_output = response.content if isinstance(response.content, str) else str(response.content)
                    run.stop_reason = "end_turn"
                    steps.append(
                        StepResult(
                            model_response=final_output,
                            tool_calls=[],
                            stop=True,
                        )
                    )
                    run.iterations = len(steps)
                    thread.updated_at = self._now()
                    self.store.save_thread(thread)
                    self.store.save_run(run)
                    await self._emit_event(
                        event_emitter,
                        "step.completed",
                        {
                            "run_id": run.id,
                            "step_index": step_index,
                            "stop": True,
                            "tool_calls_count": 0,
                        },
                    )
                    break

                # Execute tool calls
                tool_records = []
                for tool_call_index, tool_call in enumerate(response.tool_calls):
                    await self._raise_if_cancelled(cancel_event)
                    tool_name = tool_call["name"]
                    raw_tool_args = tool_call.get("args", {})
                    if not isinstance(raw_tool_args, dict):
                        tool_args = {}
                    else:
                        tool_args = raw_tool_args
                    tool_call_id = tool_call.get("id", "")
                    await self._emit_event(
                        event_emitter,
                        "tool.started",
                        {
                            "run_id": run.id,
                            "step_index": step_index,
                            "tool_call_index": tool_call_index,
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "args": self._as_json_value(tool_args),
                        },
                    )

                    try:
                        result = await mcp_manager.call_tool(tool_name, tool_args)
                        result_str = str(result)

                        thread.messages.append(
                            ToolMessage(
                                content=result_str,
                                tool_call_id=tool_call_id,
                            )
                        )

                        tool_records.append(
                            ToolCallRecord(
                                tool_name=tool_name,
                                args=tool_args,
                                result=result_str,
                                is_error=False,
                            )
                        )
                        await self._emit_event(
                            event_emitter,
                            "tool.completed",
                            {
                                "run_id": run.id,
                                "step_index": step_index,
                                "tool_call_index": tool_call_index,
                                "tool_name": tool_name,
                                "tool_call_id": tool_call_id,
                                "is_error": False,
                                "result": self._as_json_value(result),
                            },
                        )
                    except Exception as exc:
                        error_msg = f"Error calling tool {tool_name}: {str(exc)}"
                        thread.messages.append(
                            ToolMessage(
                                content=error_msg,
                                tool_call_id=tool_call_id,
                            )
                        )
                        tool_records.append(
                            ToolCallRecord(
                                tool_name=tool_name,
                                args=tool_args,
                                result=error_msg,
                                is_error=True,
                            )
                        )
                        await self._emit_event(
                            event_emitter,
                            "tool.completed",
                            {
                                "run_id": run.id,
                                "step_index": step_index,
                                "tool_call_index": tool_call_index,
                                "tool_name": tool_name,
                                "tool_call_id": tool_call_id,
                                "is_error": True,
                                "result": error_msg,
                            },
                        )

                steps.append(
                    StepResult(
                        model_response=response.content if isinstance(response.content, str) else "",
                        tool_calls=tool_records,
                        stop=False,
                    )
                )
                run.iterations = len(steps)
                thread.updated_at = self._now()
                self.store.save_thread(thread)
                self.store.save_run(run)
                await self._emit_event(
                    event_emitter,
                    "step.completed",
                    {
                        "run_id": run.id,
                        "step_index": step_index,
                        "stop": False,
                        "tool_calls_count": len(tool_records),
                    },
                )
            else:
                # Hit max iterations
                run.stop_reason = "max_iterations"
                final_output = "Maximum iterations reached"

            # Finalize the run
            run.status = "completed"
            run.iterations = len(steps)
            run.completed_at = self._now()

            # Save updated thread and run
            self.store.save_thread(thread)
            self.store.save_run(run)
            await self._emit_event(
                event_emitter,
                "run.completed",
                {
                    "run_id": run.id,
                    "iterations": run.iterations,
                    "stop_reason": run.stop_reason,
                    "output": final_output,
                },
            )

            return RunResult.from_run(run, final_output, steps)

        except RunCancelledError:
            run.status = "cancelled"
            run.stop_reason = "cancelled"
            run.completed_at = self._now()
            self.store.save_thread(thread)
            self.store.save_run(run)
            await self._emit_event(
                event_emitter,
                "run.cancelled",
                {
                    "run_id": run.id,
                    "iterations": run.iterations,
                },
            )
            return RunResult.from_run(run, "Run cancelled", steps)
        except Exception as exc:
            # Handle errors
            run.status = "failed"
            run.error = str(exc)
            run.stop_reason = "error"
            run.completed_at = self._now()
            self.store.save_run(run)
            await self._emit_event(
                event_emitter,
                "run.failed",
                {
                    "run_id": run.id,
                    "error": str(exc),
                },
            )
            raise
        finally:
            if mcp_manager is not None:
                await mcp_manager.disconnect_all()

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

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    async def _emit_event(
        self,
        emitter: RunEventEmitter | None,
        event_type: str,
        payload: dict[str, JSONValue],
    ) -> None:
        if emitter is None:
            return
        maybe_awaitable = emitter(event_type, payload)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable

    async def _raise_if_cancelled(self, cancel_event: asyncio.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RunCancelledError("Run was cancelled")

    def _as_json_value(self, value: Any) -> JSONValue:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [self._as_json_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._as_json_value(item) for key, item in value.items()}
        return str(value)
