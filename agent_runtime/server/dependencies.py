"""
FastAPI dependencies for runtime and store access.
"""

from __future__ import annotations

from fastapi import Request

from ..core.runtime import AgentRuntime
from .run_manager import RunManager


def get_runtime(request: Request) -> AgentRuntime:
    """Retrieve the initialized runtime instance from app state."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("Runtime is not initialized")
    return runtime


def get_run_manager(request: Request) -> RunManager:
    """Retrieve the initialized run manager from app state."""
    run_manager = getattr(request.app.state, "run_manager", None)
    if run_manager is None:
        raise RuntimeError("Run manager is not initialized")
    return run_manager
