"""User-upload batch processing routes (legacy CLI intake)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Body, Depends

from app.api_tags import TAG_PREPARE_PDF
from app.config import Settings
from app.dependencies import get_settings
from app.schemas import (
    ConvertLineDetectJobSummary,
    FormLineDetectionJobResponse,
    FormLineDetectorConfig,
    ProcessUserUploadsConvertLineDetectResponse,
    UserUploadConvertLineDetectItem,
    UserUploadsConvertLineDetectRequest,
)
from app.services.pdf_pipeline import scan_convert_and_detect_lines_user_uploads

LOG = logging.getLogger(__name__)

ingest_router = APIRouter(tags=[TAG_PREPARE_PDF])


@ingest_router.post(
    "/user-uploads/process-convert-line-detect",
    response_model=ProcessUserUploadsConvertLineDetectResponse,
    summary=(
        "[Legacy] Convert uploaded PDFs to page images and detect printed lines (no AI). "
        "Prefer POST /jobs for the UI pipeline."
    ),
    responses={
        200: {"description": "Per-PDF summary; jobs remain under the jobs directory for inspection"},
    },
)
async def process_user_uploads_convert_line_detect(
    body: UserUploadsConvertLineDetectRequest = Body(default_factory=UserUploadsConvertLineDetectRequest),
    settings: Settings = Depends(get_settings),
) -> ProcessUserUploadsConvertLineDetectResponse:
    det_cfg = body.config.to_detector_dict() if body.config is not None else FormLineDetectorConfig().to_detector_dict()
    raw = await asyncio.to_thread(
        scan_convert_and_detect_lines_user_uploads,
        settings=settings,
        dpi=body.dpi,
        allow_rotated_pages=body.allow_rotated_pages,
        detector_config=det_cfg,
        include_lines=body.include_lines,
    )
    items: list[UserUploadConvertLineDetectItem] = []
    for r in raw:
        detail = r.get("detail")
        if detail is not None and not isinstance(detail, dict):
            detail = {"note": str(detail)}
        ld_full = r.get("line_detection_full")
        line_detection = FormLineDetectionJobResponse.model_validate(ld_full) if ld_full is not None else None
        summary_raw = r.get("line_detection_summary")
        line_detection_summary = (
            ConvertLineDetectJobSummary.model_validate(summary_raw) if summary_raw is not None else None
        )
        items.append(
            UserUploadConvertLineDetectItem(
                source=r["source"],
                ok=r["ok"],
                job_id=r.get("job_id"),
                detected_pdf_type=r.get("detected_pdf_type"),
                pipeline=r.get("pipeline"),
                page_count=r.get("page_count"),
                dpi=r.get("dpi"),
                allow_rotated_pages=r.get("allow_rotated_pages"),
                line_detection_summary=line_detection_summary,
                line_detection=line_detection,
                error=r.get("error"),
                detail=detail if isinstance(detail, dict) else None,
            )
        )
    return ProcessUserUploadsConvertLineDetectResponse(processed=items)
