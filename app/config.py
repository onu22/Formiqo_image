"""Application configuration (environment variables)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings loaded from environment or ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="FORMIQO_",
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    jobs_dir: Path = Field(
        default=Path("./data/jobs"),
        description="Directory where per-job workspaces are created.",
    )
    user_uploads_dir: Path = Field(
        default=Path("./data/user-uploads"),
        description="Drop folder for PDFs processed by the pipeline router (CLI / process-once API).",
    )
    templates_dir: Path = Field(
        default=Path("./data/templates"),
        description=(
            "Directory for the E7 template-memory index and per-fingerprint corrected-field "
            "payloads. Small on-disk store; no database."
        ),
    )
    template_memory_enabled: bool = Field(
        default=True,
        description=(
            "When true, capture human-corrected editor saves as reusable templates and reuse "
            "them on re-upload when a page's detected-line fingerprint matches (E7)."
        ),
    )
    template_fingerprint_bins: int = Field(
        default=200,
        ge=20,
        le=2000,
        description=(
            "Quantization resolution for the normalized detected-line fingerprint. Higher = "
            "stricter matching (less tolerance to sub-bin layout drift)."
        ),
    )
    template_min_lines: int = Field(
        default=4,
        ge=1,
        le=100,
        description=(
            "Minimum detected lines on a page before it is eligible for template capture/reuse. "
            "Guards against trivial/near-empty pages false-positive matching."
        ),
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
        default=120.0,
        gt=0,
        description="Timeout for each OpenAI request in seconds.",
    )
    grounding_openai_max_output_tokens: int = Field(
        default=16000,
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
        default=120.0,
        gt=0,
        description="Timeout for each Anthropic grounding request in seconds.",
    )
    grounding_prompt_dir: Path = Field(
        default=Path("./prompts"),
        description="Directory containing grounding_system.md, grounding_developer.md, grounding_user.md.",
    )
    grounding_anthropic_max_tokens: int = Field(
        default=16000,
        ge=256,
        description="Max output tokens for Anthropic grounding and JSON repair calls.",
    )
    grounding_qa_max_iterations: int = Field(
        default=6,
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
    grounding_qa_enabled: bool = Field(
        default=False,
        description=(
            "When true, run the E5 vision QA refinement loop automatically as the final stage "
            "of the upload pipeline. Off by default to keep the ready-path fast and cheap."
        ),
    )
    grounding_qa_judge_provider: str = Field(
        default="",
        description=(
            "Provider for the E5 QA judge (openai or anthropic). Empty picks the provider that "
            "differs from the grounder to avoid correlated blind spots."
        ),
    )
    grounding_qa_judge_model: str = Field(
        default="",
        description="Model id for the QA judge. Empty falls back to a per-provider default.",
    )
    grounding_qa_judge_max_tokens: int = Field(
        default=4000,
        ge=256,
        le=16000,
        description="Max output tokens for a QA judge call (one call per page per iteration).",
    )
    grounding_qa_crop_zoom: float = Field(
        default=3.0,
        ge=1.0,
        le=8.0,
        description="Zoom factor for per-field crops sent to the judge (~3x around the stamped bbox).",
    )
    grounding_qa_crop_context_px: int = Field(
        default=24,
        ge=0,
        le=200,
        description="Extra pixel context added around a field bbox before cropping for the judge.",
    )
    grounding_qa_max_fields_per_call: int = Field(
        default=12,
        ge=1,
        le=60,
        description="Max per-field crops bundled into a single judge call (bounds judge token cost).",
    )
    grounding_qa_flag_low_confidence: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Judge confidence at or below which a field is marked qa_status=flagged for reviewer "
            "attention even when the loop stops adjusting it."
        ),
    )
    grounding_line_padding_px: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Padding around detected lines when building geometry index forbidden zones.",
    )
    grounding_stamp_inset_px: int = Field(
        default=2,
        ge=0,
        le=20,
        description="Inset applied when snapping field bboxes to cell/band interiors.",
    )
    grounding_slim_line_detection: bool = Field(
        default=True,
        description=(
            "When true, send slim line_detection_json to the LLM (line_id, orientation, bbox only). "
            "Full detected_lines.json remains on disk for validation."
        ),
    )
    grounding_max_concurrency: int = Field(
        default=4,
        ge=1,
        le=16,
        description=(
            "Max number of pages grounded in parallel (bounded per-page concurrency). "
            "Per-page errors stay isolated regardless of this value."
        ),
    )
    grounding_structured_outputs: bool = Field(
        default=True,
        description=(
            "Use provider-native structured outputs (OpenAI strict json_schema, Anthropic "
            "tool-use). Falls back to compact-JSON prompting when disabled or unsupported."
        ),
    )
    grounding_label_anchors: bool = Field(
        default=True,
        description=(
            "For digital (text-layer) PDFs, extract label positions with PyMuPDF and pass them "
            "as anchors so field bboxes can be computed deterministically."
        ),
    )
    grounding_grid_overlay_enabled: bool = Field(
        default=False,
        description=(
            "Overlay a labeled coordinate grid (ticks every grid_overlay_spacing_px) on the "
            "page image sent to the model, improving pixel-fallback coordinate accuracy."
        ),
    )
    grounding_grid_overlay_spacing_px: int = Field(
        default=100,
        ge=25,
        le=500,
        description="Spacing in pixels between labeled coordinate-grid ticks when the overlay is enabled.",
    )
