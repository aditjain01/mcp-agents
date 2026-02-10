from __future__ import annotations

import asyncio
import time

import pytest

from agent_runtime.server.run_manager import RunManager


def _wait_for_status(runtime, run_id: str, expected: set[str], timeout: float = 2.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = runtime.store.get_run(run_id)
        if run is not None and run.status in expected:
            return run.status
        time.sleep(0.01)
    run = runtime.store.get_run(run_id)
    return run.status if run else "missing"


def test_run_manager_submit_run_completes_and_records_events(runtime_with_memory_store) -> None:
    runtime = runtime_with_memory_store
    agent = runtime.create_agent(name="agent", mcp_servers=[])
    thread = runtime.create_thread()
    manager = RunManager(runtime=runtime)

    async def scenario() -> None:
        await manager.start()
        run = await manager.submit_run(
            thread_id=thread.id,
            agent_id=agent.id,
            user_message="ping",
        )
        status = _wait_for_status(runtime, run.id, {"completed", "failed", "cancelled"})
        assert status == "completed"
        events = manager.list_events(run.id)
        types = [event.type for event in events]
        assert types[0] == "run.queued"
        assert "run.started" in types
        assert "run.completed" in types
        assert [event.seq for event in events] == sorted(event.seq for event in events)
        await manager.stop()

    asyncio.run(scenario())


def test_run_manager_cancel_queued_run(runtime_with_memory_store) -> None:
    runtime = runtime_with_memory_store
    agent = runtime.create_agent(name="agent", mcp_servers=[])
    thread = runtime.create_thread()
    manager = RunManager(runtime=runtime)

    async def scenario() -> None:
        run = await manager.submit_run(
            thread_id=thread.id,
            agent_id=agent.id,
            user_message="ping",
        )
        cancelled = await manager.cancel_run(run.id)
        assert cancelled.status == "cancelled"
        events = manager.list_events(run.id)
        types = [event.type for event in events]
        assert "run.cancel_requested" in types
        assert "run.cancelled" in types

    asyncio.run(scenario())


def test_run_manager_retry_uses_previous_context_user_message(runtime_with_memory_store) -> None:
    runtime = runtime_with_memory_store
    agent = runtime.create_agent(name="agent", mcp_servers=[])
    thread = runtime.create_thread()
    source_run = runtime.create_run(
        thread.id,
        agent.id,
        context={"user_message": "retry-me", "source": "test"},
    )
    source_run.status = "completed"
    runtime.store.save_run(source_run)

    manager = RunManager(runtime=runtime)

    async def scenario() -> None:
        retried = await manager.retry_run(source_run.id)
        assert retried.id != source_run.id
        assert retried.thread_id == source_run.thread_id
        assert retried.agent_id == source_run.agent_id
        assert retried.context["user_message"] == "retry-me"
        assert retried.context["retry_of_run_id"] == source_run.id

    asyncio.run(scenario())


def test_run_manager_retry_requires_source_user_message(runtime_with_memory_store) -> None:
    runtime = runtime_with_memory_store
    agent = runtime.create_agent(name="agent", mcp_servers=[])
    thread = runtime.create_thread()
    source_run = runtime.create_run(thread.id, agent.id, context={})
    runtime.store.save_run(source_run)
    manager = RunManager(runtime=runtime)

    async def scenario() -> None:
        with pytest.raises(ValueError):
            await manager.retry_run(source_run.id)

    asyncio.run(scenario())
