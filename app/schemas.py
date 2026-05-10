"""Pydantic models for API responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ConvertResponse(BaseModel):
    """Result of a successful PDF conversion."""

    job_id: str
    page_count: int
    dpi: float
    allow_rotated_pages: bool
    source_filename: str
    document_manifest: dict[str, Any] = Field(
        description="Parsed contents of output/document_manifest.json",
    )


class ErrorResponse(BaseModel):
    detail: str


class GroundFieldsPageResult(BaseModel):
    page_index: int
    image_path: str
    status: str
    resolution: str | None = None
    output_file: str | None = None
    error: str | None = None


class GroundFieldsResponse(BaseModel):
    job_id: str
    provider: str
    model: str
    run_id: str
    run_dir: str
    page_count: int
    succeeded_count: int
    failed_count: int
    output_dir: str
    manifest_path: str
    stamping_sample_path: str | None = None
    pages: list[GroundFieldsPageResult]


class ConvertAndGroundRequest(BaseModel):
    """JSON payload for the multipart ``request`` field on ``POST /convert-and-ground``."""

    provider: Literal["openai", "anthropic"] = "anthropic"
    model: str | None = Field(
        default=None,
        description="Vision model id; omit or null to use server default for the chosen provider.",
    )


class ConvertAndGroundResponse(BaseModel):
    job_id: str
    convert: ConvertResponse
    grounding: GroundFieldsResponse


class StampProviderRequest(BaseModel):
    """JSON body for ``POST .../stamp-images`` and ``POST .../stamp-pdf``."""

    provider: Literal["openai", "anthropic"] = Field(
        default="anthropic",
        description="Must match field_grounding/manifest.json provider.",
    )


StampImagesRequest = StampProviderRequest


class StampImagesStyle(BaseModel):
    font_size_px: int = Field(default=22, ge=1, le=300)
    font_color: str = Field(default="#111111", pattern=r"^#[0-9a-fA-F]{6}$")
    padding_px: int = Field(default=3, ge=0, le=200)
    draw_debug_boxes: bool = False
    debug_box_color: str = Field(default="#ff0000", pattern=r"^#[0-9a-fA-F]{6}$")


class StampImagesPageResult(BaseModel):
    page_index: int
    status: str
    source_image: str | None = None
    grounding_file: str | None = None
    output_image: str | None = None
    field_count: int | None = None
    stamped_count: int | None = None
    missing_value_count: int | None = None
    unsupported_field_count: int | None = None
    warnings: list[str] | None = None
    error: str | None = None


class StampImagesResponse(BaseModel):
    job_id: str
    provider: str
    model: str
    stamp_run_id: str
    run_dir: str
    manifest_path: str
    page_count: int
    succeeded_count: int
    failed_count: int
    pages: list[StampImagesPageResult]


class StampPdfStyle(BaseModel):
    font_size_pt: float = Field(default=11.0, gt=0, le=200)
    font_color: str = Field(default="#111111", pattern=r"^#[0-9a-fA-F]{6}$")
    padding_pt: float = Field(default=1.0, ge=0, le=50)
    draw_debug_boxes: bool = False
    debug_box_color: str = Field(default="#ff0000", pattern=r"^#[0-9a-fA-F]{6}$")


class StampingJson(BaseModel):
    """On-disk config at ``field_grounding/stamping.json`` (written during grounding)."""

    model_config = {"extra": "ignore"}

    values: dict[str, str] = Field(default_factory=dict)
    require_all_values: bool = False
    image_style: StampImagesStyle | None = None


class StampPdfPageResult(BaseModel):
    page_index: int
    status: str
    grounding_file: str | None = None
    page_manifest: str | None = None
    source_pdf: str | None = None
    output_pdf: str | None = None
    field_count: int | None = None
    stamped_count: int | None = None
    missing_value_count: int | None = None
    unsupported_field_count: int | None = None
    warnings: list[str] | None = None
    error: str | None = None


class StampPdfResponse(BaseModel):
    job_id: str
    provider: str
    model: str
    stamp_run_id: str
    run_dir: str
    manifest_path: str
    output_pdf: str
    page_count: int
    succeeded_count: int
    failed_count: int
    pages: list[StampPdfPageResult]


class RefineGroundingIterationPage(BaseModel):
    page_index: int
    qa_status: str
    corrections_requested: int = 0
    preview_image: str | None = None
    note: str | None = None


class RefineGroundingResponse(BaseModel):
    job_id: str
    provider: str
    model: str
    session_id: str
    promoted: bool
    stopped_reason: str
    iterations_run: int
    refined_dir: str
    qa_session_dir: str
    final_preview_dir: str
    canonical_grounding_updated: bool
    iterations: list[list[RefineGroundingIterationPage]]


class UserUploadProcessItem(BaseModel):
    source: str
    ok: bool
    job_id: str | None = None
    detected_pdf_type: str | None = None
    pipeline: str | None = None
    error: str | None = None
    detail: dict[str, Any] | None = None


class ProcessUserUploadsResponse(BaseModel):
    processed: list[UserUploadProcessItem]
