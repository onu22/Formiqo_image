"""FastAPI dependency providers."""

from __future__ import annotations

from functools import lru_cache

from app.config import Settings


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
