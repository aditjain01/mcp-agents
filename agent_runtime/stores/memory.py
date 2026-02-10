"""
In-memory store implementation.
"""

from __future__ import annotations

from ..core.entities import Agent, Run, RunEvent, Thread


class InMemoryStore:
    """
    Dict-based in-memory store. Three dicts keyed by id.
    Drop-in replacement for a DB store later.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._threads: dict[str, Thread] = {}
        self._runs: dict[str, Run] = {}
        self._run_events: dict[str, list[RunEvent]] = {}

    # -- Agent --

    def save_agent(self, agent: Agent) -> None:
        self._agents[agent.id] = agent

    def get_agent(self, agent_id: str) -> Agent | None:
        return self._agents.get(agent_id)

    def list_agents(self) -> list[Agent]:
        return list(self._agents.values())

    def delete_agent(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    # -- Thread --

    def save_thread(self, thread: Thread) -> None:
        self._threads[thread.id] = thread

    def get_thread(self, thread_id: str) -> Thread | None:
        return self._threads.get(thread_id)

    def list_threads(self) -> list[Thread]:
        return list(self._threads.values())

    def delete_thread(self, thread_id: str) -> None:
        self._threads.pop(thread_id, None)

    # -- Run --

    def save_run(self, run: Run) -> None:
        self._runs[run.id] = run

    def get_run(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def list_runs(self, thread_id: str) -> list[Run]:
        return [r for r in self._runs.values() if r.thread_id == thread_id]

    # -- Run events --

    def save_run_event(self, event: RunEvent) -> None:
        events = self._run_events.setdefault(event.run_id, [])
        events.append(event)
        events.sort(key=lambda item: item.seq)

    def list_run_events(
        self,
        run_id: str,
        *,
        after_seq: int = 0,
        limit: int = 1000,
    ) -> list[RunEvent]:
        events = self._run_events.get(run_id, [])
        if limit <= 0:
            return []
        return [event for event in events if event.seq > after_seq][:limit]

    def get_latest_run_event_seq(self, run_id: str) -> int:
        events = self._run_events.get(run_id, [])
        if not events:
            return 0
        return max(event.seq for event in events)
