"""
Persistence abstraction.

Defines a Store protocol and an InMemoryStore implementation.
The protocol is the contract that any future DB store (Prisma, SQLAlchemy, etc.)
must satisfy -- swap implementations without changing runtime code.
"""

from __future__ import annotations

from typing import Protocol

from .entities import Agent, Thread, Run


# ---------------------------------------------------------------------------
# Store Protocol
# ---------------------------------------------------------------------------


class Store(Protocol):
    """
    Persistence interface for all three core entities.
    Any implementation (in-memory, Postgres, SQLite) must satisfy this.
    """

    # -- Agent CRUD --
    def save_agent(self, agent: Agent) -> None: ...
    def get_agent(self, agent_id: str) -> Agent | None: ...
    def list_agents(self) -> list[Agent]: ...
    def delete_agent(self, agent_id: str) -> None: ...

    # -- Thread CRUD --
    def save_thread(self, thread: Thread) -> None: ...
    def get_thread(self, thread_id: str) -> Thread | None: ...
    def list_threads(self) -> list[Thread]: ...
    def delete_thread(self, thread_id: str) -> None: ...

    # -- Run CRUD --
    def save_run(self, run: Run) -> None: ...
    def get_run(self, run_id: str) -> Run | None: ...
    def list_runs(self, thread_id: str) -> list[Run]: ...


# ---------------------------------------------------------------------------
# InMemoryStore -- v0 implementation
# ---------------------------------------------------------------------------


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
