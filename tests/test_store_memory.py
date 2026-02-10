from __future__ import annotations

from agent_runtime.core.entities import Agent, Run, RunEvent, Thread
from agent_runtime.core.types import ModelConfigV0, RunEventPayloadV0
from agent_runtime.stores.memory import InMemoryStore


def test_memory_store_crud_for_core_entities() -> None:
    store = InMemoryStore()

    agent = Agent(name="agent", model=ModelConfigV0(model="gpt-4o"))
    thread = Thread(metadata={"test": True})
    run = Run(thread_id=thread.id, agent_id=agent.id, status="queued")

    store.save_agent(agent)
    store.save_thread(thread)
    store.save_run(run)

    assert store.get_agent(agent.id) is not None
    assert store.get_thread(thread.id) is not None
    assert store.get_run(run.id) is not None
    assert len(store.list_agents()) == 1
    assert len(store.list_threads()) == 1
    assert len(store.list_runs(thread.id)) == 1

    store.delete_thread(thread.id)
    assert store.get_thread(thread.id) is None


def test_memory_store_run_events_filter_and_latest_seq() -> None:
    store = InMemoryStore()
    run_id = "run-1"

    store.save_run_event(
        RunEvent(
            run_id=run_id,
            seq=1,
            type="run.queued",
            payload=RunEventPayloadV0(data={"run_id": run_id}),
        )
    )
    store.save_run_event(
        RunEvent(
            run_id=run_id,
            seq=2,
            type="run.started",
            payload=RunEventPayloadV0(data={"run_id": run_id}),
        )
    )

    assert store.get_latest_run_event_seq(run_id) == 2

    all_events = store.list_run_events(run_id)
    assert [event.seq for event in all_events] == [1, 2]

    filtered = store.list_run_events(run_id, after_seq=1)
    assert [event.seq for event in filtered] == [2]
