"""HTTP routes for hybrid line-map semantic field grounding."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Body, Depends, HTTPException

from app.config import Settings
from app.dependencies import get_settings
from app.schemas import (
    GroundFieldsFromLinesRequest,
    GroundFieldsFromLinesResponse,
    SemanticGroundingPageResult,
)
from app.services.jobs import job_paths
from app.services.line_detection_job import NO_CONVERTED_PAGE_PNGS, list_converted_page_pngs
from app.services.semantic_grounding import SemanticGroundingJobError, run_semantic_grounding_for_job

LOG = logging.getLogger(__name__)

router = APIRouter(tags=["grounding"])


@router.post(
    "/jobs/{job_id}/ground-fields-from-lines",
    response_model=GroundFieldsFromLinesResponse,
    summary="Ground all converted pages with OpenAI using line map + page images (geometry from OpenCV)",
    responses={
        400: {"description": "Invalid job, missing conversion/line detection, or bad request"},
        404: {"description": "Job not found"},
        422: {"description": "Grounding failed for every page"},
    },
)
async def ground_fields_from_lines(
    job_id: str,
    body: GroundFieldsFromLinesRequest = Body(default_factory=GroundFieldsFromLinesRequest),
    settings: Settings = Depends(get_settings),
) -> GroundFieldsFromLinesResponse:
    try:
        root, _, output_dir = job_paths(settings.jobs_dir, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if not output_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Job output folder not found: {output_dir}")

    try:
        pages = list_converted_page_pngs(output_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not pages:
        raise HTTPException(
            status_code=400,
            detail=(
                "No converted page PNGs found. Run "
                "POST /api/v1/user-uploads/process-convert-line-detect first."
            ),
        )

    line_root = output_dir / "line_detection"
    if not line_root.is_dir() or not any(line_root.glob("page_*/detected_lines.json")):
        raise HTTPException(
            status_code=400,
            detail=(
                "No line_detection output found. Run "
                "POST /api/v1/user-uploads/process-convert-line-detect first "
                "(or ensure this job has output/line_detection/ from that pipeline)."
            ),
        )

    try:
        raw = await asyncio.to_thread(
            run_semantic_grounding_for_job,
            job_id=job_id,
            output_dir=output_dir,
            settings=settings,
            provider=body.provider,
            model=body.model,
        )
    except ValueError as exc:
        if str(exc) == NO_CONVERTED_PAGE_PNGS:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SemanticGroundingJobError as exc:
        failed_models = [SemanticGroundingPageResult.model_validate(p) for p in exc.failed_pages]
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(exc),
                "job_id": job_id,
                "failed_pages": [p.model_dump() for p in failed_models],
            },
        ) from exc

    return GroundFieldsFromLinesResponse(**raw)
