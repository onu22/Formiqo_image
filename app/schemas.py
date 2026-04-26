"""Pydantic models for API responses."""

from __future__ import annotations

from typing import Any

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
    pages: list[GroundFieldsPageResult]


class ConvertAndGroundResponse(BaseModel):
    job_id: str
    convert: ConvertResponse
    grounding: GroundFieldsResponse


class StampImagesStyle(BaseModel):
    font_size_px: int = Field(default=22, ge=1, le=300)
    font_color: str = Field(default="#111111", pattern=r"^#[0-9a-fA-F]{6}$")
    padding_px: int = Field(default=3, ge=0, le=200)
    draw_debug_boxes: bool = False
    debug_box_color: str = Field(default="#ff0000", pattern=r"^#[0-9a-fA-F]{6}$")


class StampImagesProviderRequest(BaseModel):
    values: dict[str, str] = Field(default_factory=dict)
    style: StampImagesStyle | None = None
    require_all_values: bool = False


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
