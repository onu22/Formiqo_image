"""Pydantic models for API responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ErrorResponse(BaseModel):
    detail: str


class FormLineDetectorConfig(BaseModel):
    """Optional OpenCV overrides for ``POST /user-uploads/process-convert-line-detect`` body ``config``."""

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
    path: str = Field(
        description="Path to the source page image, relative to the job output directory (e.g. converted_images/page_0001.png).",
    )
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
    line_id: str | None = None
    line_style: Literal["solid"] | None = None
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


class FormLineDetectionJobResponse(BaseModel):
    """Line detection for every converted page in a job."""

    job_id: str
    page_count: int
    pages: list[FormLineDetectionPageResult]


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


class UserUploadsConvertLineDetectRequest(BaseModel):
    """JSON body for ``POST /user-uploads/process-convert-line-detect`` (optional)."""

    model_config = {"extra": "ignore"}
    dpi: float = Field(default=200.0, gt=0, le=600)
    allow_rotated_pages: bool = False
    config: FormLineDetectorConfig | None = Field(
        default=None,
        description="OpenCV detector overrides; omit for defaults.",
    )
    include_lines: bool = Field(
        default=False,
        description="If true, include full ``lines`` arrays per page in ``line_detection`` (can be very large).",
    )


class ConvertLineDetectPageSummary(BaseModel):
    page_index: int
    counts: FormLineDetectionCounts
    output_files: dict[str, str] | None = None


class ConvertLineDetectJobSummary(BaseModel):
    job_id: str
    page_count: int
    pages: list[ConvertLineDetectPageSummary]


class UserUploadConvertLineDetectItem(BaseModel):
    source: str
    ok: bool
    job_id: str | None = None
    detected_pdf_type: str | None = None
    pipeline: str | None = None
    page_count: int | None = None
    dpi: float | None = None
    allow_rotated_pages: bool | None = None
    line_detection_summary: ConvertLineDetectJobSummary | None = None
    line_detection: FormLineDetectionJobResponse | None = Field(
        default=None,
        description="Populated only when the request set ``include_lines`` to true.",
    )
    error: str | None = None
    detail: dict[str, Any] | None = None


class ProcessUserUploadsConvertLineDetectResponse(BaseModel):
    processed: list[UserUploadConvertLineDetectItem]


class GroundFieldsFromLinesRequest(BaseModel):
    """JSON body for ``POST /jobs/{job_id}/ground-fields-from-lines``."""

    model_config = {"extra": "ignore"}
    provider: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="Vision provider for field grounding.",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Vision model id; omit for defaults (openai: gpt-5.5, anthropic: claude-opus-4-7) "
            "or env FORMIQO_GROUNDING_MODEL / FORMIQO_COMBINED_DEFAULT_*."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _apply_provider_model_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        prov = str(data.get("provider") or "openai").strip().lower()
        model = data.get("model")
        if model is None or (isinstance(model, str) and not model.strip()):
            data = dict(data)
            data["model"] = "claude-opus-4-7" if prov == "anthropic" else "gpt-5.5"
        return data


class SemanticGroundingPageResult(BaseModel):
    page_index: int
    status: str
    grounding_file: str | None = None
    error: str | None = None
    detail: Any | None = None


class GroundFieldsFromLinesResponse(BaseModel):
    job_id: str
    provider: str
    model: str
    run_dir: str
    manifest_path: str
    page_count: int
    succeeded_count: int
    failed_count: int
    pages: list[SemanticGroundingPageResult]
    failed_pages: list[SemanticGroundingPageResult] = Field(default_factory=list)
