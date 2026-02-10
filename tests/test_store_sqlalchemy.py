from __future__ import annotations

from agent_runtime.core.entities import Agent, Run, RunEvent, Thread
from agent_runtime.core.types import ModelConfigV0, RunEventPayloadV0
from agent_runtime.stores.sqlalchemy import SQLAlchemyStore


def test_sqlalchemy_store_crud_roundtrip() -> None:
    store = SQLAlchemyStore(database_url="sqlite+pysqlite:///:memory:")
    agent = Agent(name="sql-agent", model=ModelConfigV0(model="gpt-4o-mini"))
    thread = Thread(metadata={"origin": "sql"})
    run = Run(thread_id=thread.id, agent_id=agent.id, status="queued")

    store.save_agent(agent)
    store.save_thread(thread)
    store.save_run(run)

    loaded_agent = store.get_agent(agent.id)
    loaded_thread = store.get_thread(thread.id)
    loaded_run = store.get_run(run.id)

    assert loaded_agent is not None
    assert loaded_agent.model.model == "gpt-4o-mini"
    assert loaded_thread is not None
    assert loaded_thread.metadata["origin"] == "sql"
    assert loaded_run is not None
    assert loaded_run.status == "queued"


def test_sqlalchemy_store_run_events_querying() -> None:
    store = SQLAlchemyStore(database_url="sqlite+pysqlite:///:memory:")
    run_id = "run-sql"

    first = RunEvent(
        run_id=run_id,
        seq=1,
        type="run.queued",
        payload=RunEventPayloadV0(data={"run_id": run_id}),
    )
    second = RunEvent(
        run_id=run_id,
        seq=2,
        type="run.started",
        payload=RunEventPayloadV0(data={"run_id": run_id}),
    )
    store.save_run_event(first)
    store.save_run_event(second)

    assert store.get_latest_run_event_seq(run_id) == 2
    assert [event.seq for event in store.list_run_events(run_id)] == [1, 2]
    assert [event.seq for event in store.list_run_events(run_id, after_seq=1)] == [2]
