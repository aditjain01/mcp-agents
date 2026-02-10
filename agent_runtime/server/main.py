"""
Command-line launcher for the runtime API server.
"""

from __future__ import annotations

import os

import uvicorn


def run() -> None:
    """Run the API server using uvicorn."""
    host = os.getenv("AGENT_RUNTIME_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_RUNTIME_PORT", "8000"))
    reload = os.getenv("AGENT_RUNTIME_RELOAD", "false").strip().lower() == "true"

    uvicorn.run(
        "agent_runtime.server.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run()
