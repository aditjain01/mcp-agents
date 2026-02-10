from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agent_runtime.core.entities import Agent, RunEvent, Thread


def test_agent_from_legacy_payload_migrates_model_config() -> None:
    payload = {
        "id": "agent-1",
        "name": "legacy",
        "model": "gpt-4o",
        "provider": "openai",
        "temperature": 0.4,
        "api_key": "legacy-key",
        "system_prompt": "",
        "mcp_servers": [],
        "max_iterations": 10,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    agent = Agent.from_dict(payload)
    assert agent.model.version == "v0"
    assert agent.model.model == "gpt-4o"
    assert agent.model.config["temperature"] == 0.4
    assert agent.model.config["api_key"] == "legacy-key"


def test_thread_roundtrip_preserves_message_types_and_order() -> None:
    thread = Thread(metadata={"source": "test"})
    thread.messages.append(HumanMessage(content="hello"))
    thread.messages.append(AIMessage(content="world"))

    data = thread.to_dict()
    restored = Thread.from_dict(data)

    assert restored.metadata["source"] == "test"
    assert len(restored.messages) == 2
    assert restored.messages[0].__class__.__name__ == "HumanMessage"
    assert restored.messages[1].__class__.__name__ == "AIMessage"
    assert restored.messages[0].content == "hello"
    assert restored.messages[1].content == "world"


def test_run_event_from_dict_wraps_unversioned_payload() -> None:
    event = RunEvent.from_dict(
        {
            "id": "evt-1",
            "run_id": "run-1",
            "seq": 1,
            "type": "run.started",
            "payload": {"run_id": "run-1"},
            "created_at": "2026-01-01T00:00:00Z",
        }
    )
    assert event.payload.version == "v0"
    assert event.payload.data["run_id"] == "run-1"
