"""Application configuration (environment variables)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment or ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="FORMIQO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    jobs_dir: Path = Field(
        default=Path("./data/jobs"),
        description="Directory where per-job workspaces are created.",
    )
    max_upload_bytes: int = Field(
        default=50 * 1024 * 1024,
        ge=1024,
        description="Maximum accepted PDF upload size in bytes.",
    )
    api_title: str = "Formiqo PDF Grounding API"
    api_version: str = "1.0.0"
    cors_allow_origins: str = Field(
        default="",
        description="Comma-separated browser origins for CORS; empty disables the middleware.",
    )
