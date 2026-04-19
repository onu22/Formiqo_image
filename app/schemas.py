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
