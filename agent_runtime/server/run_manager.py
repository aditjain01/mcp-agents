"""
In-process run queue/worker manager.

This keeps the first streaming/cancellation iteration simple:
- no external queue dependency
- one async worker task
- persisted run-event log for replay
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator

from ..core.entities import Run, RunEvent
from ..core.runtime import AgentRuntime
from ..core.types import JSONValue, RunEventPayloadV0

TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}


@dataclass
class RunRequest:
    """Queued run request payload for the worker loop."""

    run_id: str
    user_message: str


class RunManager:
    """
    Coordinates run submission, background execution, event persistence, and
    event fanout for streaming subscribers.
    """

    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime
        self._queue: asyncio.Queue[RunRequest | None] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._seq_by_run: dict[str, int] = {}
        self._seq_lock = asyncio.Lock()
        self._subscribers: dict[str, set[asyncio.Queue[RunEvent]]] = {}

    async def start(self) -> None:
        """Start the background run worker task."""
        if self._worker_task and not self._worker_task.done():
            return
        self._worker_task = asyncio.create_task(
            self._worker_loop(),
            name="agent-runtime-run-worker",
        )

    async def stop(self) -> None:
        """Stop the worker and request cancellation of active runs."""
        if self._worker_task is None:
            return
        for cancel_event in self._cancel_events.values():
            cancel_event.set()
        await self._queue.put(None)
        await self._worker_task
        self._worker_task = None

    async def submit_run(
        self,
        *,
        thread_id: str,
        agent_id: str,
        user_message: str,
        context: dict[str, JSONValue] | None = None,
    ) -> Run:
        """Create a queued run and enqueue it for background execution."""
        run_context = dict(context or {})
        run_context.setdefault("user_message", user_message)

        run = self.runtime.create_run(
            thread_id=thread_id,
            agent_id=agent_id,
            context=run_context,
        )
        await self._record_event(
            run_id=run.id,
            event_type="run.queued",
            payload={
                "run_id": run.id,
                "thread_id": run.thread_id,
                "agent_id": run.agent_id,
            },
        )
        await self._queue.put(RunRequest(run_id=run.id, user_message=user_message))
        return run

    async def cancel_run(self, run_id: str) -> Run:
        """Request cancellation for a queued/running run."""
        run = self.runtime.store.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        await self._record_event(
            run_id=run.id,
            event_type="run.cancel_requested",
            payload={"run_id": run.id},
        )

        if run.status in TERMINAL_RUN_STATUSES:
            return run

        if run.status == "queued":
            run.status = "cancelled"
            run.stop_reason = "cancelled"
            run.completed_at = self._now()
            self.runtime.store.save_run(run)
            await self._record_event(
                run_id=run.id,
                event_type="run.cancelled",
                payload={
                    "run_id": run.id,
                    "iterations": run.iterations,
                },
            )
            return run

        cancel_event = self._cancel_events.get(run.id)
        if cancel_event is not None:
            cancel_event.set()
        return run

    async def retry_run(self, source_run_id: str) -> Run:
        """
        Retry a previous run by creating a new queued run with same thread/agent.

        For v0, we reuse the original user message from run context.
        """
        source_run = self.runtime.store.get_run(source_run_id)
        if source_run is None:
            raise ValueError(f"Run {source_run_id} not found")

        raw_user_message = source_run.context.get("user_message")
        if not isinstance(raw_user_message, str) or not raw_user_message:
            raise ValueError(
                "Cannot retry run: source run does not have context.user_message"
            )

        retry_context = dict(source_run.context)
        retry_context["retry_of_run_id"] = source_run.id

        return await self.submit_run(
            thread_id=source_run.thread_id,
            agent_id=source_run.agent_id,
            user_message=raw_user_message,
            context=retry_context,
        )

    def list_events(
        self,
        run_id: str,
        *,
        after_seq: int = 0,
        limit: int = 1000,
    ) -> list[RunEvent]:
        """List persisted run events for replay."""
        return self.runtime.store.list_run_events(
            run_id,
            after_seq=after_seq,
            limit=limit,
        )

    async def stream_events(
        self,
        run_id: str,
        *,
        after_seq: int = 0,
    ) -> AsyncIterator[RunEvent]:
        """Stream historical + live events for a run."""
        run = self.runtime.store.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        queue: asyncio.Queue[RunEvent] = asyncio.Queue(maxsize=200)
        subscribers = self._subscribers.setdefault(run_id, set())
        subscribers.add(queue)

        last_seq = after_seq
        try:
            historical = self.runtime.store.list_run_events(
                run_id,
                after_seq=after_seq,
                limit=10000,
            )
            for event in historical:
                if event.seq > last_seq:
                    last_seq = event.seq
                    yield event

            while True:
                current_run = self.runtime.store.get_run(run_id)
                if (
                    current_run is not None
                    and current_run.status in TERMINAL_RUN_STATUSES
                    and queue.empty()
                ):
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    continue

                if event.seq <= last_seq:
                    continue
                last_seq = event.seq
                yield event
        finally:
            subscribers = self._subscribers.get(run_id)
            if subscribers is not None:
                subscribers.discard(queue)
                if not subscribers:
                    self._subscribers.pop(run_id, None)

    async def _worker_loop(self) -> None:
        """Background run execution loop."""
        while True:
            request = await self._queue.get()
            if request is None:
                self._queue.task_done()
                break

            cancel_event = asyncio.Event()
            self._cancel_events[request.run_id] = cancel_event
            try:
                run = self.runtime.store.get_run(request.run_id)
                if run is None:
                    continue
                if run.status == "cancelled":
                    continue

                await self.runtime.execute_run(
                    request.run_id,
                    request.user_message,
                    event_emitter=lambda event_type, payload: self._record_event(
                        run_id=request.run_id,
                        event_type=event_type,
                        payload=payload,
                    ),
                    cancel_event=cancel_event,
                )
            except Exception as exc:
                run = self.runtime.store.get_run(request.run_id)
                if run and run.status not in TERMINAL_RUN_STATUSES:
                    run.status = "failed"
                    run.error = str(exc)
                    run.stop_reason = "error"
                    run.completed_at = self._now()
                    self.runtime.store.save_run(run)
                    await self._record_event(
                        run_id=run.id,
                        event_type="run.failed",
                        payload={"run_id": run.id, "error": str(exc)},
                    )
            finally:
                self._cancel_events.pop(request.run_id, None)
                self._queue.task_done()

    async def _record_event(
        self,
        *,
        run_id: str,
        event_type: str,
        payload: dict[str, JSONValue],
    ) -> RunEvent:
        seq = await self._next_seq(run_id)
        event = RunEvent(
            run_id=run_id,
            seq=seq,
            type=event_type,  # validated by RunEventType literal
            payload=RunEventPayloadV0(data=self._as_json_dict(payload)),
        )
        self.runtime.store.save_run_event(event)
        await self._publish_event(event)
        return event

    async def _next_seq(self, run_id: str) -> int:
        async with self._seq_lock:
            current = self._seq_by_run.get(run_id)
            if current is None:
                current = self.runtime.store.get_latest_run_event_seq(run_id)
            current += 1
            self._seq_by_run[run_id] = current
            return current

    async def _publish_event(self, event: RunEvent) -> None:
        subscribers = self._subscribers.get(event.run_id)
        if not subscribers:
            return

        for queue in list(subscribers):
            while queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # If still full after dropping oldest, skip this subscriber event.
                continue

    def _as_json_dict(self, payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
        return {str(key): self._as_json_value(value) for key, value in payload.items()}

    def _as_json_value(self, value: JSONValue) -> JSONValue:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [self._as_json_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._as_json_value(item) for key, item in value.items()}
        return str(value)

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)
