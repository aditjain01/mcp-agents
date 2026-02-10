"""
Backward-compatible re-export for SQLAlchemy-backed store.
"""

from .stores.sqlalchemy import SQLAlchemyStore

__all__ = ["SQLAlchemyStore"]
