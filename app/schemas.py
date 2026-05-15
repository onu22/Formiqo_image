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


class FormLineDetectorConfig(BaseModel):
    """Optional overrides for ``POST /jobs/{job_id}/detect-form-lines`` request body ``config``."""

    min_horizontal_length_px: int = Field(default=45, ge=1)
    min_vertical_length_px: int = Field(default=45, ge=1)
    max_horizontal_thickness_px: int = Field(default=12, ge=1)
    max_vertical_thickness_px: int = Field(default=12, ge=1)
    horizontal_kernel_divisor: int = Field(default=35, ge=1)
    vertical_kernel_divisor: int = Field(default=35, ge=1)

    def to_detector_dict(self) -> dict[str, int]:
        return {
            "min_horizontal_length_px": self.min_horizontal_length_px,
            "min_vertical_length_px": self.min_vertical_length_px,
            "max_horizontal_thickness_px": self.max_horizontal_thickness_px,
            "max_vertical_thickness_px": self.max_vertical_thickness_px,
            "horizontal_kernel_divisor": self.horizontal_kernel_divisor,
            "vertical_kernel_divisor": self.vertical_kernel_divisor,
        }


class FormLineDetectionImageInfo(BaseModel):
    path: str
    width: int
    height: int
    unit: Literal["px"] = "px"
    origin: Literal["top-left"] = "top-left"


class FormLineDetectionDetectorInfo(BaseModel):
    method: Literal["opencv_morphology"] = "opencv_morphology"
    config: dict[str, int]


class FormLineDetectionCounts(BaseModel):
    horizontal: int
    vertical: int
    total: int


class LineBBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class FormLineRecord(BaseModel):
    orientation: Literal["horizontal", "vertical"]
    x1: int
    y1: int
    x2: int
    y2: int
    bbox: LineBBox
    thickness: int


class FormLineDetectionResponse(BaseModel):
    """OpenCV morphology line detection result (same shape as written JSON)."""

    image: FormLineDetectionImageInfo
    detector: FormLineDetectionDetectorInfo
    counts: FormLineDetectionCounts
    lines: list[FormLineRecord]
    output_files: dict[str, str] | None = Field(
        default=None,
        description="When written under a job, relative paths for json and highlight PNG.",
    )


class FormLineDetectionPageResult(BaseModel):
    """One page within a job-wide line detection run."""

    page_index: int = Field(description="0-based page index.")
    detection: FormLineDetectionResponse


class DetectFormLinesJobRequest(BaseModel):
    """Optional JSON body for ``POST /jobs/{job_id}/detect-form-lines``."""

    model_config = {"extra": "ignore"}
    config: FormLineDetectorConfig | None = Field(
        default=None,
        description="Detector settings; omit or null for defaults.",
    )


class FormLineDetectionJobResponse(BaseModel):
    """Line detection for every converted page in a job."""

    job_id: str
    page_count: int
    pages: list[FormLineDetectionPageResult]


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


class VisionGroundingOptions(BaseModel):
    """Provider and vision model for field grounding (shared by convert-and-ground and ground-fields)."""

    provider: Literal["openai", "anthropic"] = "anthropic"
    model: str | None = Field(
        default=None,
        description="Vision model id; omit or null to use server default for the chosen provider.",
    )


class ConvertAndGroundRequest(VisionGroundingOptions):
    """JSON payload for the multipart ``request`` field on ``POST /convert-and-ground``."""


class GroundFieldsRequest(VisionGroundingOptions):
    """JSON body for ``POST /jobs/{job_id}/ground-fields``."""


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
    promoted: bool = Field(
        description="True when QA stopped because every page was acceptable.",
    )
    stopped_reason: str
    iterations_run: int
    refined_dir: str
    qa_session_dir: str
    final_preview_dir: str
    canonical_grounding_updated: bool = Field(
        description=(
            "True after this run: canonical field_grounding/page_*.fields.json were overwritten "
            "from refined/ so stamping matches the last QA iteration."
        ),
    )
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
