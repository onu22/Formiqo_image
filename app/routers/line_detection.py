"""HTTP routes for OpenCV form line detection on converted job pages."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Body, Depends, HTTPException

from app.config import Settings
from app.dependencies import get_settings
from app.schemas import (
    DetectFormLinesJobRequest,
    FormLineDetectionJobResponse,
    FormLineDetectorConfig,
)
from app.services.jobs import job_paths
from app.services.line_detection_job import NO_CONVERTED_PAGE_PNGS, run_detect_form_lines_for_job_output_dir

LOG = logging.getLogger(__name__)

router = APIRouter(tags=["line-detection"])


def _effective_detector_config(body: DetectFormLinesJobRequest) -> dict[str, int]:
    if body.config is not None:
        return body.config.to_detector_dict()
    return FormLineDetectorConfig().to_detector_dict()


@router.post(
    "/jobs/{job_id}/detect-form-lines",
    response_model=FormLineDetectionJobResponse,
    summary="Detect form lines on all converted pages for a job (OpenCV morphology)",
    responses={
        400: {"description": "Invalid job_id or missing conversion output"},
        404: {"description": "Job not found"},
        422: {"description": "A page failed detection or could not be written"},
    },
)
async def detect_form_lines_for_job(
    job_id: str,
    body: DetectFormLinesJobRequest = Body(default_factory=DetectFormLinesJobRequest),
    settings: Settings = Depends(get_settings),
) -> FormLineDetectionJobResponse:
    try:
        root, _, output_dir = job_paths(settings.jobs_dir, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if not output_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Job output folder not found: {output_dir}")

    det_cfg = _effective_detector_config(body)

    try:
        return await asyncio.to_thread(
            run_detect_form_lines_for_job_output_dir,
            job_id=job_id,
            output_dir=output_dir,
            detector_config=det_cfg,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        if str(exc) == NO_CONVERTED_PAGE_PNGS:
            raise HTTPException(
                status_code=400,
                detail="No converted page PNGs found under converted_images/ (expected page_0001.png, …).",
            ) from exc
        raise HTTPException(
            status_code=422,
            detail={"message": str(exc), "job_id": job_id},
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": str(exc), "job_id": job_id},
        ) from exc
