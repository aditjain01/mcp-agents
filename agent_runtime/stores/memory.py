"""
In-memory store implementation.
"""

from __future__ import annotations

from ..core.entities import Agent, Run, Thread


class InMemoryStore:
    """
    Dict-based in-memory store. Three dicts keyed by id.
    Drop-in replacement for a DB store later.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._threads: dict[str, Thread] = {}
        self._runs: dict[str, Run] = {}

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
