"""
FastAPI application for serving the runtime via HTTP APIs.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.concurrency import run_in_threadpool

from ..core.runtime import AgentRuntime
from ..stores.base import Store
from ..stores.memory import InMemoryStore
from ..stores.sqlalchemy import SQLAlchemyStore
from .dependencies import get_runtime
from .schemas import (
    AgentResponse,
    CreateAgentRequest,
    CreateThreadRequest,
    ExecuteRunRequest,
    RunResponse,
    RunResultResponse,
    ThreadResponse,
)


def _create_store_from_env() -> Store:
    """
    Build a store implementation from environment variables.

    Supported:
    - AGENT_RUNTIME_STORE=in_memory (default)
    - AGENT_RUNTIME_STORE=sqlalchemy
    - AGENT_RUNTIME_DATABASE_URL=<sqlalchemy-url> (used by sqlalchemy store)
    """
    kind = os.getenv("AGENT_RUNTIME_STORE", "in_memory").strip().lower()
    if kind in {"sqlalchemy", "sqlite", "postgres", "postgresql"}:
        database_url = os.getenv(
            "AGENT_RUNTIME_DATABASE_URL",
            "sqlite+pysqlite:///./agent_runtime.db",
        )
        return SQLAlchemyStore(database_url=database_url)
    return InMemoryStore()


def create_app(*, store: Store | None = None) -> FastAPI:
    """Create a configured FastAPI app with runtime lifecycle management."""

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        resolved_store = store or _create_store_from_env()
        app_instance.state.store = resolved_store
        app_instance.state.runtime = AgentRuntime(store=resolved_store)
        try:
            yield
        finally:
            close = getattr(resolved_store, "close", None)
            if callable(close):
                close()

    app_instance = FastAPI(
        title="Agent Runtime API",
        version="0.1.0",
        description=(
            "Opinionated HTTP API for AgentRuntime. "
            "Create agents/threads, execute runs, and inspect execution records."
        ),
        lifespan=lifespan,
    )

    @app_instance.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # ---- Agent routes ----

    @app_instance.post("/agents", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
    async def create_agent(
        payload: CreateAgentRequest,
        runtime: AgentRuntime = Depends(get_runtime),
    ) -> AgentResponse:
        agent = runtime.create_agent(
            name=payload.name,
            system_prompt=payload.system_prompt,
            mcp_servers=payload.mcp_servers,
            max_iterations=payload.max_iterations,
            model=payload.model,
            provider=payload.provider,
            temperature=payload.temperature,
            api_key=payload.api_key,
            model_params=payload.model_params,
            model_config=payload.model_config_payload,
        )
        return AgentResponse.from_entity(agent)

    @app_instance.get("/agents", response_model=list[AgentResponse])
    async def list_agents(runtime: AgentRuntime = Depends(get_runtime)) -> list[AgentResponse]:
        return [AgentResponse.from_entity(agent) for agent in runtime.list_agents()]

    @app_instance.get("/agents/{agent_id}", response_model=AgentResponse)
    async def get_agent(
        agent_id: str,
        runtime: AgentRuntime = Depends(get_runtime),
    ) -> AgentResponse:
        agent = runtime.get_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
        return AgentResponse.from_entity(agent)

    @app_instance.delete("/agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_agent(
        agent_id: str,
        runtime: AgentRuntime = Depends(get_runtime),
    ) -> Response:
        runtime.store.delete_agent(agent_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # ---- Thread routes ----

    @app_instance.post("/threads", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
    async def create_thread(
        payload: CreateThreadRequest,
        runtime: AgentRuntime = Depends(get_runtime),
    ) -> ThreadResponse:
        thread = runtime.create_thread(metadata=dict(payload.metadata))
        return ThreadResponse.from_entity(thread)

    @app_instance.get("/threads", response_model=list[ThreadResponse])
    async def list_threads(runtime: AgentRuntime = Depends(get_runtime)) -> list[ThreadResponse]:
        return [ThreadResponse.from_entity(thread) for thread in runtime.list_threads()]

    @app_instance.get("/threads/{thread_id}", response_model=ThreadResponse)
    async def get_thread(
        thread_id: str,
        runtime: AgentRuntime = Depends(get_runtime),
    ) -> ThreadResponse:
        thread = runtime.get_thread(thread_id)
        if thread is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
        return ThreadResponse.from_entity(thread)

    @app_instance.delete("/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_thread(
        thread_id: str,
        runtime: AgentRuntime = Depends(get_runtime),
    ) -> Response:
        runtime.store.delete_thread(thread_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # ---- Run routes ----

    @app_instance.post("/runs", response_model=RunResultResponse, status_code=status.HTTP_201_CREATED)
    async def execute_run(
        payload: ExecuteRunRequest,
        runtime: AgentRuntime = Depends(get_runtime),
    ) -> RunResultResponse:
        try:
            result = await run_in_threadpool(
                runtime.run,
                payload.thread_id,
                payload.agent_id,
                payload.user_message,
            )
        except ValueError as exc:
            message = str(exc)
            if "not found" in message.lower():
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message) from exc
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Run execution failed: {exc}",
            ) from exc

        return RunResultResponse.from_domain(result)

    @app_instance.get("/runs/{run_id}", response_model=RunResponse)
    async def get_run(
        run_id: str,
        runtime: AgentRuntime = Depends(get_runtime),
    ) -> RunResponse:
        run = runtime.store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
        return RunResponse.from_entity(run)

    @app_instance.get("/threads/{thread_id}/runs", response_model=list[RunResponse])
    async def list_thread_runs(
        thread_id: str,
        runtime: AgentRuntime = Depends(get_runtime),
    ) -> list[RunResponse]:
        runs = runtime.store.list_runs(thread_id)
        return [RunResponse.from_entity(run) for run in runs]

    return app_instance


app = create_app()
