from __future__ import annotations

import time

from langchain_core.messages import AIMessage, HumanMessage


def _create_agent_and_thread(client):
    agent_resp = client.post(
        "/agents",
        json={"name": "api-agent", "model": "gpt-4o-mini", "mcp_servers": []},
    )
    assert agent_resp.status_code == 201, agent_resp.text
    thread_resp = client.post("/threads", json={"metadata": {"source": "api-test"}})
    assert thread_resp.status_code == 201, thread_resp.text
    return agent_resp.json(), thread_resp.json()


def _wait_for_run(client, run_id: str, expected: set[str], timeout: float = 3.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        run_resp = client.get(f"/runs/{run_id}")
        assert run_resp.status_code == 200, run_resp.text
        run_data = run_resp.json()
        if run_data["status"] in expected:
            return run_data
        time.sleep(0.01)
    run_resp = client.get(f"/runs/{run_id}")
    return run_resp.json()


def test_server_health_and_basic_crud(client) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    agent, thread = _create_agent_and_thread(client)

    list_agents = client.get("/agents")
    assert list_agents.status_code == 200
    assert any(item["id"] == agent["id"] for item in list_agents.json())

    list_threads = client.get("/threads")
    assert list_threads.status_code == 200
    assert any(item["id"] == thread["id"] for item in list_threads.json())


def test_server_run_submit_poll_events_stream_cancel_retry_and_sync_execute(client) -> None:
    agent, thread = _create_agent_and_thread(client)

    submit = client.post(
        "/runs",
        json={
            "thread_id": thread["id"],
            "agent_id": agent["id"],
            "user_message": "ping",
        },
    )
    assert submit.status_code == 202, submit.text
    run = submit.json()

    completed_run = _wait_for_run(client, run["id"], {"completed", "failed", "cancelled"})
    assert completed_run["status"] == "completed"

    events = client.get(f"/runs/{run['id']}/events")
    assert events.status_code == 200, events.text
    event_types = [event["type"] for event in events.json()]
    assert "run.queued" in event_types
    assert "run.started" in event_types
    assert "run.completed" in event_types

    after_seq = events.json()[-1]["seq"] - 1
    tail = client.get(f"/runs/{run['id']}/events", params={"after_seq": after_seq})
    assert tail.status_code == 200
    assert all(item["seq"] > after_seq for item in tail.json())

    stream_resp = client.get(f"/runs/{run['id']}/stream")
    assert stream_resp.status_code == 200
    assert "event: run.completed" in stream_resp.text

    retry = client.post(f"/runs/{run['id']}/retry")
    assert retry.status_code == 202, retry.text
    retry_run_id = retry.json()["id"]
    assert retry_run_id != run["id"]
    retried_run = _wait_for_run(client, retry_run_id, {"completed", "failed", "cancelled"})
    assert retried_run["status"] == "completed"

    cancel_submit = client.post(
        "/runs",
        json={
            "thread_id": thread["id"],
            "agent_id": agent["id"],
            "user_message": "wait-for-cancel",
        },
    )
    assert cancel_submit.status_code == 202, cancel_submit.text
    cancel_run_id = cancel_submit.json()["id"]

    cancel_resp = client.post(f"/runs/{cancel_run_id}/cancel")
    assert cancel_resp.status_code == 200, cancel_resp.text
    cancelled_run = _wait_for_run(client, cancel_run_id, {"cancelled", "completed", "failed"})
    assert cancelled_run["status"] == "cancelled"

    execute_sync = client.post(
        "/runs/execute",
        json={
            "thread_id": thread["id"],
            "agent_id": agent["id"],
            "user_message": "sync-call",
        },
    )
    assert execute_sync.status_code == 201, execute_sync.text
    assert execute_sync.json()["output"] == "echo:sync-call"


def test_server_thread_fork_endpoint(client) -> None:
    agent, thread = _create_agent_and_thread(client)
    runtime = client.app.state.runtime
    source = runtime.get_thread(thread["id"])
    assert source is not None
    source.messages.append(HumanMessage(content="m1"))
    source.messages.append(AIMessage(content="m2"))
    source.messages.append(HumanMessage(content="m3"))
    runtime.store.save_thread(source)

    fork_resp = client.post(
        f"/threads/{thread['id']}/fork",
        json={
            "up_to_message_index": 1,
            "metadata": {"branch": "alt"},
            "include_source_metadata": True,
        },
    )
    assert fork_resp.status_code == 201, fork_resp.text
    forked = fork_resp.json()
    assert forked["id"] != thread["id"]
    assert len(forked["messages"]) == 2
    assert forked["metadata"]["branch"] == "alt"
    assert forked["metadata"]["forked_from_thread_id"] == thread["id"]
    assert forked["metadata"]["forked_from_message_index"] == 1

    bad_fork = client.post(
        f"/threads/{thread['id']}/fork",
        json={"up_to_message_index": 999},
    )
    assert bad_fork.status_code == 400

    # Ensure existing agent still accessible (sanity check with fixture-created app state).
    assert client.get(f"/agents/{agent['id']}").status_code == 200


def test_server_thread_rerun_endpoint(client) -> None:
    agent, thread = _create_agent_and_thread(client)
    runtime = client.app.state.runtime
    source = runtime.get_thread(thread["id"])
    assert source is not None
    source.messages.append(HumanMessage(content="history-1"))
    source.messages.append(AIMessage(content="history-2"))
    runtime.store.save_thread(source)

    rerun_resp = client.post(
        f"/threads/{thread['id']}/rerun",
        json={
            "agent_id": agent["id"],
            "user_message": "new-branch",
            "up_to_message_index": 1,
            "metadata": {"reason": "experiment"},
            "include_source_metadata": True,
        },
    )
    assert rerun_resp.status_code == 202, rerun_resp.text
    rerun_payload = rerun_resp.json()

    forked_thread = rerun_payload["thread"]
    rerun = rerun_payload["run"]
    assert forked_thread["id"] != thread["id"]
    assert rerun["thread_id"] == forked_thread["id"]
    assert forked_thread["metadata"]["reason"] == "experiment"
    assert forked_thread["metadata"]["forked_from_thread_id"] == thread["id"]

    run_data = _wait_for_run(client, rerun["id"], {"completed", "failed", "cancelled"})
    assert run_data["status"] == "completed"
    assert run_data["context"]["rerun_from_thread_id"] == thread["id"]
