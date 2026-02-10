"""
Backward-compatible re-export for store interfaces.
"""

from .stores.base import Store
from .stores.memory import InMemoryStore

__all__ = ["Store", "InMemoryStore"]
