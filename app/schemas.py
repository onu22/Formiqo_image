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
