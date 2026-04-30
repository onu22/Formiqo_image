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
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key used for field grounding endpoint.",
    )
    openai_model: str = Field(
        default="gpt-5",
        description="OpenAI model name used for per-page field grounding.",
    )
    openai_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Timeout for each OpenAI request in seconds.",
    )
    grounding_openai_max_output_tokens: int = Field(
        default=9600,
        ge=256,
        description="Max output tokens for OpenAI grounding and JSON repair calls.",
    )
    grounding_provider: str = Field(
        default="openai",
        description="Default provider for field grounding runs (openai or anthropic).",
    )
    grounding_model: str = Field(
        default="gpt-5",
        description="Default model string for field grounding when request does not override it.",
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key used when provider=anthropic.",
    )
    anthropic_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Timeout for each Anthropic grounding request in seconds.",
    )
    grounding_anthropic_max_tokens: int = Field(
        default=4800,
        ge=256,
        description="Max output tokens for Anthropic grounding and JSON repair calls.",
    )
    combined_default_anthropic_model: str = Field(
        default="claude-opus-4-7",
        description="Anthropic model used by /convert-and-ground dual-default flow.",
    )
    combined_default_openai_model: str = Field(
        default="gpt-5.5",
        description="OpenAI model used by /convert-and-ground dual-default flow.",
    )
