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
        description='Default Anthropic model when POST /convert-and-ground JSON omits "model" for provider anthropic.',
    )
    combined_default_openai_model: str = Field(
        default="gpt-5.5",
        description='Default OpenAI model when POST /convert-and-ground JSON omits "model" for provider openai.',
    )
    grounding_qa_max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max stamp→QA→apply rounds for /refine-grounding.",
    )
    grounding_qa_max_bbox_delta_px: int = Field(
        default=30,
        ge=1,
        le=200,
        description="Max absolute bbox delta per axis per QA refinement iteration.",
    )
    grounding_qa_consensus_translation_enabled: bool = Field(
        default=True,
        description="When the LLM omits page_translation, merge similar per-field deltas into one translation.",
    )
    grounding_qa_consensus_min_fields: int = Field(
        default=3,
        ge=2,
        le=500,
        description="Minimum per-field corrections on a page to run consensus translation merge.",
    )
    grounding_qa_consensus_max_spread_px: int = Field(
        default=4,
        ge=0,
        le=50,
        description="Max spread (max-min) of delta components on an axis to treat as consensus.",
    )
