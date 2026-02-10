from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from types import MethodType
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent_runtime import AgentRuntime, InMemoryStore, create_app
from agent_runtime.core.models import RunResult, StepResult


async def _emit(
    event_emitter: Any,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    if event_emitter is None:
        return
    maybe = event_emitter(event_type, payload)
    if asyncio.iscoroutine(maybe):
        await maybe


def install_stubbed_execute_run(runtime: AgentRuntime) -> None:
    """
    Patch runtime.execute_run with a deterministic async stub for tests.

    Special behavior:
    - user_message == "wait-for-cancel" waits briefly for cancel_event.
    """

    async def fake_execute_run(
        self: AgentRuntime,
        run_id: str,
        user_message: str,
        *,
        event_emitter=None,
        cancel_event=None,
    ):
        run = self.store.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        run.status = "running"
        run.error = None
        run.stop_reason = None
        run.completed_at = None
        self.store.save_run(run)

        await _emit(event_emitter, "run.started", {"run_id": run.id})
        await _emit(event_emitter, "step.started", {"run_id": run.id, "step_index": 0})

        if user_message == "wait-for-cancel":
            for _ in range(120):
                if cancel_event is not None and cancel_event.is_set():
                    run.status = "cancelled"
                    run.stop_reason = "cancelled"
                    run.completed_at = datetime.now(timezone.utc)
                    self.store.save_run(run)
                    await _emit(
                        event_emitter,
                        "run.cancelled",
                        {"run_id": run.id, "iterations": run.iterations},
                    )
                    return RunResult.from_run(run, "Run cancelled", [])
                await asyncio.sleep(0.005)

        run.status = "completed"
        run.iterations = 1
        run.stop_reason = "end_turn"
        run.completed_at = datetime.now(timezone.utc)
        self.store.save_run(run)

        await _emit(
            event_emitter,
            "step.completed",
            {
                "run_id": run.id,
                "step_index": 0,
                "stop": True,
                "tool_calls_count": 0,
            },
        )
        await _emit(
            event_emitter,
            "run.completed",
            {
                "run_id": run.id,
                "iterations": run.iterations,
                "stop_reason": run.stop_reason,
                "output": f"echo:{user_message}",
            },
        )
        return RunResult(
            output=f"echo:{user_message}",
            run_id=run.id,
            thread_id=run.thread_id,
            iterations=1,
            stop_reason="end_turn",
            steps=[StepResult(model_response=f"echo:{user_message}", stop=True)],
        )

    runtime.execute_run = MethodType(fake_execute_run, runtime)


def wait_for_run_status(
    runtime: AgentRuntime,
    run_id: str,
    *,
    expected: set[str],
    timeout_seconds: float = 3.0,
) -> str:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        run = runtime.store.get_run(run_id)
        if run is not None and run.status in expected:
            return run.status
        time.sleep(0.01)
    run = runtime.store.get_run(run_id)
    return run.status if run else "missing"


@pytest.fixture
def runtime_with_memory_store() -> AgentRuntime:
    runtime = AgentRuntime(store=InMemoryStore())
    install_stubbed_execute_run(runtime)
    return runtime


@pytest.fixture
def client() -> TestClient:
    app = create_app(store=InMemoryStore())
    with TestClient(app) as test_client:
        runtime = test_client.app.state.runtime
        install_stubbed_execute_run(runtime)
        yield test_client
