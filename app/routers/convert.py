"""User-upload batch processing and job stamping routes."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import ValidationError

from app.config import Settings
from app.dependencies import get_settings
from app.schemas import (
    ConvertLineDetectJobSummary,
    FormLineDetectionJobResponse,
    FormLineDetectorConfig,
    ProcessUserUploadsConvertLineDetectResponse,
    StampImagesResponse,
    StampProviderRequest,
    StampPdfResponse,
    StampingJson,
    UserUploadConvertLineDetectItem,
    UserUploadsConvertLineDetectRequest,
)
from app.services.pdf_pipeline import scan_convert_and_detect_lines_user_uploads
from app.services.image_stamping import run_image_stamping_for_job
from app.services.jobs import job_paths
from app.services.pdf_stamping import StampPdfStyle, run_pdf_stamping_for_job
from app.services.stamping_config import (
    load_field_grounding_manifest,
    load_stamping_json_parsed,
    manifest_provider_model,
    stamping_json_to_image_style,
)

LOG = logging.getLogger(__name__)

router = APIRouter(tags=["conversion"])


def _http_load_field_grounding_manifest(output_dir: Path) -> dict[str, Any]:
    try:
        return load_field_grounding_manifest(output_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _http_load_stamping_json(output_dir: Path) -> StampingJson:
    try:
        return load_stamping_json_parsed(output_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": "Invalid field_grounding/stamping.json", "errors": exc.errors()},
        ) from exc


def _manifest_provider_must_match_route_or_400(manifest: dict[str, Any], route_provider: str) -> tuple[str, str]:
    try:
        prov, model = manifest_provider_model(manifest)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    want = route_provider.strip().lower()
    if prov != want:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Grounding manifest provider is {prov!r}; POST .../stamp-images or .../stamp-pdf "
                f'with JSON {{"provider":"{prov}"}}, not provider={want!r}.'
            ),
        )
    return prov, model


@router.post(
    "/user-uploads/process-convert-line-detect",
    response_model=ProcessUserUploadsConvertLineDetectResponse,
    summary=(
        "PDF inbox → rasterize each PDF and run OpenCV line detection (primary entry; no LLM). "
        "Use returned job_id with POST /jobs/{job_id}/ground-fields-from-lines next."
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


@router.post(
    "/jobs/{job_id}/stamp-images",
    response_model=StampImagesResponse,
    summary="Stamp values onto images (provider in JSON body; must match grounding manifest)",
    responses={
        400: {"description": "Invalid job, missing converted images, or missing grounding run"},
        404: {"description": "Job not found"},
        422: {"description": "No pages succeeded image stamping"},
    },
)
async def stamp_images_for_job(
    job_id: str,
    body: StampProviderRequest = Body(default_factory=StampProviderRequest),
    settings: Settings = Depends(get_settings),
) -> StampImagesResponse:
    route_provider = body.provider.strip().lower()
    return await _stamp_images_for_provider(
        job_id=job_id,
        route_provider=route_provider,
        settings=settings,
    )


async def _stamp_images_for_provider(
    *,
    job_id: str,
    route_provider: str,
    settings: Settings,
) -> StampImagesResponse:
    try:
        root, _, output_dir = job_paths(settings.jobs_dir, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if not output_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Job output folder not found: {output_dir}")

    manifest = _http_load_field_grounding_manifest(output_dir)
    provider, model = _manifest_provider_must_match_route_or_400(manifest, route_provider)
    stamping = _http_load_stamping_json(output_dir)
    style = stamping_json_to_image_style(stamping)

    try:
        result = await asyncio.to_thread(
            run_image_stamping_for_job,
            job_id=job_id,
            output_dir=output_dir,
            provider=provider,
            model=model,
            values=stamping.values,
            style=style,
            require_all_values=stamping.require_all_values,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result["succeeded_count"] == 0:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Image stamping failed for all pages.",
                "job_id": job_id,
                "provider": result["provider"],
                "model": result["model"],
                "page_count": result["page_count"],
                "failed_count": result["failed_count"],
                "pages": result["pages"],
            },
        )

    return StampImagesResponse(**result)


@router.post(
    "/jobs/{job_id}/stamp-pdf",
    response_model=StampPdfResponse,
    summary="Stamp values onto original PDF (provider in JSON body; must match grounding manifest)",
    responses={
        400: {"description": "Invalid job, missing converted manifests, or missing grounding run"},
        404: {"description": "Job not found"},
        422: {"description": "No pages succeeded PDF stamping"},
    },
)
async def stamp_pdf_for_job(
    job_id: str,
    body: StampProviderRequest = Body(default_factory=StampProviderRequest),
    settings: Settings = Depends(get_settings),
) -> StampPdfResponse:
    route_provider = body.provider.strip().lower()
    return await _stamp_pdf_for_provider(
        job_id=job_id,
        route_provider=route_provider,
        settings=settings,
    )


async def _stamp_pdf_for_provider(
    *,
    job_id: str,
    route_provider: str,
    settings: Settings,
) -> StampPdfResponse:
    try:
        root, input_pdf, output_dir = job_paths(settings.jobs_dir, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if not output_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Job output folder not found: {output_dir}")
    if not input_pdf.is_file():
        raise HTTPException(status_code=400, detail=f"Input PDF not found for job: {input_pdf}")

    manifest = _http_load_field_grounding_manifest(output_dir)
    provider, model = _manifest_provider_must_match_route_or_400(manifest, route_provider)
    stamping = _http_load_stamping_json(output_dir)
    style = StampPdfStyle()

    try:
        result = await asyncio.to_thread(
            run_pdf_stamping_for_job,
            job_id=job_id,
            input_pdf=input_pdf,
            output_dir=output_dir,
            provider=provider,
            model=model,
            values=stamping.values,
            style=style,
            require_all_values=stamping.require_all_values,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result["succeeded_count"] == 0:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "PDF stamping failed for all pages.",
                "job_id": job_id,
                "provider": result["provider"],
                "model": result["model"],
                "page_count": result["page_count"],
                "failed_count": result["failed_count"],
                "pages": result["pages"],
            },
        )

    return StampPdfResponse(**result)
