"""FastAPI dependency providers."""

from __future__ import annotations

from app.config import Settings


def get_settings() -> Settings:
    """Return settings loaded from environment / project ``.env``."""
    return Settings()
