"""Pydantic models for API responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, conlist, field_validator


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
    links: dict[str, str] = Field(
        description="Hyperlinks for follow-up downloads (relative to API root).",
    )


class ErrorResponse(BaseModel):
    detail: str


class GroundFieldsPageResult(BaseModel):
    page_index: int
    image_path: str
    status: str
    output_file: str | None = None
    error: str | None = None


class GroundFieldsRequest(BaseModel):
    provider: str | None = None
    model: str | None = None


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


class PrepareStampRequest(BaseModel):
    provider: str | None = None
    model: str | None = None


class PrepareStampPageResult(BaseModel):
    page_index: int
    status: str
    grounding_file: str | None = None
    stamp_plan_path: str | None = None
    field_count: int | None = None
    error: str | None = None


class PrepareStampResponse(BaseModel):
    job_id: str
    provider: str
    model: str
    run_id: str
    run_dir: str
    manifest_path: str
    page_count: int
    succeeded_count: int
    failed_count: int
    pages: list[PrepareStampPageResult]


class StampOptionsModel(BaseModel):
    fontsize: float = Field(default=10.0, gt=0, le=200.0)
    fontname: str = Field(default="helv", min_length=1, max_length=64)
    color_rgb: conlist(float, min_length=3, max_length=3) = Field(default=[0.0, 0.0, 0.0])
    align: Literal["left", "center", "right"] = "left"
    autoshrink: bool = True
    min_fontsize: float = Field(default=6.0, gt=0, le=200.0)

    @field_validator("color_rgb")
    @classmethod
    def _validate_color(cls, value: list[float]) -> list[float]:
        if any(c < 0 or c > 1 for c in value):
            raise ValueError("color_rgb components must be in [0, 1]")
        return value


class StampPdfRequest(BaseModel):
    provider: str | None = None
    model: str | None = None
    values: dict[str, str] = Field(default_factory=dict)
    strict: bool = False
    options: StampOptionsModel | None = None


class StampPdfFieldResult(BaseModel):
    field_id: str
    status: str
    final_fontsize: float | None = None
    error: str | None = None


class StampPdfPageResult(BaseModel):
    page_index: int
    stamped: int
    fields: list[StampPdfFieldResult]


class StampPdfResponse(BaseModel):
    job_id: str
    provider: str
    model: str
    run_dir: str
    stamped_pdf_path: str
    result_path: str
    page_count: int
    stamped_field_count: int
    skipped_missing_values: list[str]
    unknown_field_ids: list[str]
    download_url: str
    pages: list[StampPdfPageResult]
