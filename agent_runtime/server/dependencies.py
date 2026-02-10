"""
FastAPI dependencies for runtime and store access.
"""

from __future__ import annotations

from fastapi import Request

from ..core.runtime import AgentRuntime


def get_runtime(request: Request) -> AgentRuntime:
    """Retrieve the initialized runtime instance from app state."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("Runtime is not initialized")
    return runtime
