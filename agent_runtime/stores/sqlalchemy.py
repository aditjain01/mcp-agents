"""
SQLAlchemy-backed persistence store.

This store implements the same Store protocol as InMemoryStore and persists
Agent/Thread/Run entities to a relational database.

Schema creation strategy:
- We intentionally rely on `Base.metadata.create_all(...)` only.
- No migration tooling is introduced in v0.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from sqlalchemy import (
    JSON,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from ..core.entities import Agent, Run, RunEvent, Thread

JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    """Base declarative model for SQLAlchemy tables."""


class AgentRecord(Base):
    """SQL row model for persisted Agent entities."""

    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_payload: Mapped[dict[str, Any]] = mapped_column("model", JSON_TYPE, nullable=False)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    mcp_servers: Mapped[list[dict[str, Any]]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    max_iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ThreadRecord(Base):
    """SQL row model for persisted Thread entities."""

    __tablename__ = "threads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    metadata_payload: Mapped[dict[str, Any]] = mapped_column("metadata", JSON_TYPE, nullable=False, default=dict)
    messages: Mapped[list[dict[str, Any]]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RunRecord(Base):
    """SQL row model for persisted Run entities."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    agent_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stop_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    mcp_servers_used: Mapped[list[str] | None] = mapped_column(JSON_TYPE, nullable=True)
    context: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False, default=dict)


class RunEventRecord(Base):
    """SQL row model for persisted RunEvent entities."""

    __tablename__ = "run_events"
    __table_args__ = (UniqueConstraint("run_id", "seq", name="uq_run_events_run_id_seq"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column("type", String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def _ensure_aware_utc(value: datetime | None) -> datetime | None:
    """Normalize DB datetime values to timezone-aware UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class SQLAlchemyStore:
    """
    SQLAlchemy implementation of the Store protocol.

    The store is session-based and uses straightforward upsert-on-save semantics.
    """

    def __init__(
        self,
        database_url: str = "sqlite+pysqlite:///:memory:",
        *,
        engine: Engine | None = None,
        session_factory: sessionmaker[Session] | None = None,
        create_schema: bool = True,
    ) -> None:
        self.engine = engine or create_engine(database_url, future=True)
        self._session_factory = session_factory or sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=Session,
        )
        if create_schema:
            Base.metadata.create_all(self.engine)

    @contextmanager
    def _write_session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def _read_session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
        finally:
            session.close()

    # -- Agent --

    def save_agent(self, agent: Agent) -> None:
        payload = {
            "id": agent.id,
            "name": agent.name,
            "model_payload": agent.model.model_dump(mode="json"),
            "system_prompt": agent.system_prompt,
            "mcp_servers": [server.model_dump(mode="json") for server in agent.mcp_servers],
            "max_iterations": agent.max_iterations,
            "created_at": agent.created_at,
            "updated_at": agent.updated_at,
        }
        with self._write_session() as session:
            record = session.get(AgentRecord, agent.id)
            if record is None:
                session.add(AgentRecord(**payload))
                return
            for key, value in payload.items():
                setattr(record, key, value)

    def get_agent(self, agent_id: str) -> Agent | None:
        with self._read_session() as session:
            record = session.get(AgentRecord, agent_id)
            if record is None:
                return None
            return Agent.from_dict(
                {
                    "id": record.id,
                    "name": record.name,
                    "model": record.model_payload,
                    "system_prompt": record.system_prompt,
                    "mcp_servers": record.mcp_servers,
                    "max_iterations": record.max_iterations,
                    "created_at": _ensure_aware_utc(record.created_at),
                    "updated_at": _ensure_aware_utc(record.updated_at),
                }
            )

    def list_agents(self) -> list[Agent]:
        with self._read_session() as session:
            rows = session.scalars(select(AgentRecord).order_by(AgentRecord.created_at.asc())).all()
            return [
                Agent.from_dict(
                    {
                        "id": record.id,
                        "name": record.name,
                        "model": record.model_payload,
                        "system_prompt": record.system_prompt,
                        "mcp_servers": record.mcp_servers,
                        "max_iterations": record.max_iterations,
                        "created_at": _ensure_aware_utc(record.created_at),
                        "updated_at": _ensure_aware_utc(record.updated_at),
                    }
                )
                for record in rows
            ]

    def delete_agent(self, agent_id: str) -> None:
        with self._write_session() as session:
            record = session.get(AgentRecord, agent_id)
            if record is not None:
                session.delete(record)

    # -- Thread --

    def save_thread(self, thread: Thread) -> None:
        payload = thread.to_dict()
        row_payload = {
            "id": thread.id,
            "metadata_payload": payload.get("metadata", {}),
            "messages": payload.get("messages", []),
            "created_at": thread.created_at,
            "updated_at": thread.updated_at,
        }
        with self._write_session() as session:
            record = session.get(ThreadRecord, thread.id)
            if record is None:
                session.add(ThreadRecord(**row_payload))
                return
            for key, value in row_payload.items():
                setattr(record, key, value)

    def get_thread(self, thread_id: str) -> Thread | None:
        with self._read_session() as session:
            record = session.get(ThreadRecord, thread_id)
            if record is None:
                return None
            return Thread.from_dict(
                {
                    "id": record.id,
                    "metadata": record.metadata_payload,
                    "messages": record.messages,
                    "created_at": _ensure_aware_utc(record.created_at),
                    "updated_at": _ensure_aware_utc(record.updated_at),
                }
            )

    def list_threads(self) -> list[Thread]:
        with self._read_session() as session:
            rows = session.scalars(select(ThreadRecord).order_by(ThreadRecord.created_at.asc())).all()
            return [
                Thread.from_dict(
                    {
                        "id": record.id,
                        "metadata": record.metadata_payload,
                        "messages": record.messages,
                        "created_at": _ensure_aware_utc(record.created_at),
                        "updated_at": _ensure_aware_utc(record.updated_at),
                    }
                )
                for record in rows
            ]

    def delete_thread(self, thread_id: str) -> None:
        with self._write_session() as session:
            record = session.get(ThreadRecord, thread_id)
            if record is not None:
                session.delete(record)

    # -- Run --

    def save_run(self, run: Run) -> None:
        payload = {
            "id": run.id,
            "thread_id": run.thread_id,
            "agent_id": run.agent_id,
            "status": run.status,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "iterations": run.iterations,
            "stop_reason": run.stop_reason,
            "error": run.error,
            "mcp_servers_used": run.mcp_servers_used,
            "context": run.context,
        }
        with self._write_session() as session:
            record = session.get(RunRecord, run.id)
            if record is None:
                session.add(RunRecord(**payload))
                return
            for key, value in payload.items():
                setattr(record, key, value)

    def get_run(self, run_id: str) -> Run | None:
        with self._read_session() as session:
            record = session.get(RunRecord, run_id)
            if record is None:
                return None
            return Run.from_dict(
                {
                    "id": record.id,
                    "thread_id": record.thread_id,
                    "agent_id": record.agent_id,
                    "status": record.status,
                    "started_at": _ensure_aware_utc(record.started_at),
                    "completed_at": _ensure_aware_utc(record.completed_at),
                    "iterations": record.iterations,
                    "stop_reason": record.stop_reason,
                    "error": record.error,
                    "mcp_servers_used": record.mcp_servers_used,
                    "context": record.context,
                }
            )

    def list_runs(self, thread_id: str) -> list[Run]:
        with self._read_session() as session:
            rows = session.scalars(
                select(RunRecord)
                .where(RunRecord.thread_id == thread_id)
                .order_by(RunRecord.started_at.asc())
            ).all()
            return [
                Run.from_dict(
                    {
                        "id": record.id,
                        "thread_id": record.thread_id,
                        "agent_id": record.agent_id,
                        "status": record.status,
                        "started_at": _ensure_aware_utc(record.started_at),
                        "completed_at": _ensure_aware_utc(record.completed_at),
                        "iterations": record.iterations,
                        "stop_reason": record.stop_reason,
                        "error": record.error,
                        "mcp_servers_used": record.mcp_servers_used,
                        "context": record.context,
                    }
                )
                for record in rows
            ]

    def close(self) -> None:
        """Dispose the underlying engine and release pooled connections."""
        self.engine.dispose()

    # -- Run events --

    def save_run_event(self, event: RunEvent) -> None:
        payload = {
            "id": event.id,
            "run_id": event.run_id,
            "seq": event.seq,
            "event_type": event.type,
            "payload": event.payload.model_dump(mode="json"),
            "created_at": event.created_at,
        }
        with self._write_session() as session:
            record = session.get(RunEventRecord, event.id)
            if record is None:
                session.add(RunEventRecord(**payload))
                return
            for key, value in payload.items():
                setattr(record, key, value)

    def list_run_events(
        self,
        run_id: str,
        *,
        after_seq: int = 0,
        limit: int = 1000,
    ) -> list[RunEvent]:
        if limit <= 0:
            return []

        with self._read_session() as session:
            rows = session.scalars(
                select(RunEventRecord)
                .where(RunEventRecord.run_id == run_id)
                .where(RunEventRecord.seq > after_seq)
                .order_by(RunEventRecord.seq.asc())
                .limit(limit)
            ).all()
            return [
                RunEvent.from_dict(
                    {
                        "id": row.id,
                        "run_id": row.run_id,
                        "seq": row.seq,
                        "type": row.event_type,
                        "payload": row.payload,
                        "created_at": _ensure_aware_utc(row.created_at),
                    }
                )
                for row in rows
            ]

    def get_latest_run_event_seq(self, run_id: str) -> int:
        with self._read_session() as session:
            value = session.scalars(
                select(RunEventRecord.seq)
                .where(RunEventRecord.run_id == run_id)
                .order_by(RunEventRecord.seq.desc())
                .limit(1)
            ).first()
            return int(value or 0)
