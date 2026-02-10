from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent_runtime.core.runtime import AgentRuntime
from agent_runtime.stores.memory import InMemoryStore


def _build_runtime_with_thread() -> tuple[AgentRuntime, str]:
    runtime = AgentRuntime(store=InMemoryStore())
    thread = runtime.create_thread(metadata={"session": "alpha"})
    thread.messages.append(HumanMessage(content="first"))
    thread.messages.append(AIMessage(content="second"))
    thread.messages.append(HumanMessage(content="third"))
    runtime.store.save_thread(thread)
    return runtime, thread.id


def test_fork_thread_copies_all_messages_and_metadata() -> None:
    runtime, thread_id = _build_runtime_with_thread()

    forked = runtime.fork_thread(thread_id)
    source = runtime.get_thread(thread_id)
    assert source is not None

    assert forked.id != source.id
    assert len(forked.messages) == len(source.messages)
    assert forked.metadata["session"] == "alpha"
    assert forked.metadata["forked_from_thread_id"] == source.id
    assert forked.metadata["forked_from_message_count"] == 3
    # Ensure message objects are deep-copied
    assert forked.messages[0] is not source.messages[0]


def test_fork_thread_with_up_to_message_index() -> None:
    runtime, thread_id = _build_runtime_with_thread()

    forked = runtime.fork_thread(
        thread_id,
        up_to_message_index=1,
        metadata_override={"fork_reason": "rerun"},
    )

    assert len(forked.messages) == 2
    assert forked.messages[0].content == "first"
    assert forked.messages[1].content == "second"
    assert forked.metadata["fork_reason"] == "rerun"
    assert forked.metadata["forked_from_message_index"] == 1


def test_fork_thread_not_found_raises() -> None:
    runtime = AgentRuntime(store=InMemoryStore())
    with pytest.raises(ValueError):
        runtime.fork_thread("missing")


def test_fork_thread_invalid_index_raises() -> None:
    runtime, thread_id = _build_runtime_with_thread()
    with pytest.raises(ValueError):
        runtime.fork_thread(thread_id, up_to_message_index=100)
    with pytest.raises(ValueError):
        runtime.fork_thread(thread_id, up_to_message_index=-1)
