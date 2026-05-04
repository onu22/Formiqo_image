"""PDF upload → images + manifests."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.config import Settings
from app.dependencies import get_settings
from app.schemas import (
    ConvertResponse,
    ConvertAndGroundResponse,
    GroundFieldsResponse,
    RefineGroundingIterationPage,
    RefineGroundingResponse,
    StampImagesResponse,
    StampPdfResponse,
    StampingJson,
)
from app.services.conversion import run_convert_pdf_to_images
from app.services.field_grounding import run_field_grounding_for_job
from app.services.grounding_qa import run_grounding_qa_refinement_loop
from app.services.image_stamping import _assert_grounding_run_matches, run_image_stamping_for_job
from app.services.jobs import job_paths, provider_model_dir_name, read_document_manifest
from app.services.pdf_stamping import StampPdfStyle, run_pdf_stamping_for_job
from app.services.stamping_config import (
    load_field_grounding_manifest,
    load_stamping_json_parsed,
    manifest_provider_model,
    stamping_json_to_image_style,
)

LOG = logging.getLogger(__name__)

router = APIRouter(tags=["conversion"])


async def _save_upload_limited(
    upload: UploadFile,
    dest: Path,
    *,
    max_bytes: int,
) -> None:
    """Stream ``upload`` to ``dest`` and abort if the stream exceeds ``max_bytes``."""
    total = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds max_upload_bytes ({max_bytes}).",
                    )
                out.write(chunk)
    finally:
        await upload.close()


def _cleanup_job_dir(root: Path) -> None:
    if root.is_dir():
        shutil.rmtree(root, ignore_errors=True)


def _write_provider_metadata_to_document_manifest(
    *,
    output_dir: Path,
    provider: str,
    model: str,
) -> dict:
    manifest_path = output_dir / "document_manifest.json"
    doc = read_document_manifest(output_dir)
    doc["provider"] = provider
    doc["model"] = model
    doc["provider_model"] = provider_model_dir_name(provider, model)
    manifest_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


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
                f"Grounding manifest provider is {prov!r}; call POST .../stamp-images/{prov} "
                f"(or stamp-pdf/{prov}), not {want!r}."
            ),
        )
    return prov, model


@router.post(
    "/jobs/{job_id}/refine-grounding",
    response_model=RefineGroundingResponse,
    summary="Vision QA loop: refine grounding bboxes via stamped previews (delta patches)",
    responses={
        400: {"description": "Invalid job, manifest mismatch, or missing outputs"},
        404: {"description": "Job not found"},
    },
)
async def refine_grounding_for_job(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> RefineGroundingResponse:
    try:
        root, _, output_dir = job_paths(settings.jobs_dir, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if not output_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Job output folder not found: {output_dir}")

    grounding_dir = output_dir / "field_grounding"
    if not grounding_dir.is_dir():
        raise HTTPException(status_code=400, detail="field_grounding not found for this job.")

    manifest = _http_load_field_grounding_manifest(output_dir)
    try:
        provider, model = manifest_provider_model(manifest)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        _assert_grounding_run_matches(grounding_dir, provider=provider, model=model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stamping = _http_load_stamping_json(output_dir)
    style = stamping_json_to_image_style(stamping)

    try:
        raw = await asyncio.to_thread(
            run_grounding_qa_refinement_loop,
            job_id=job_id,
            output_dir=output_dir,
            provider=provider,
            model=model,
            values=stamping.values,
            style=style,
            require_all_values=stamping.require_all_values,
            openai_api_key=settings.openai_api_key,
            openai_timeout_seconds=settings.openai_timeout_seconds,
            openai_max_output_tokens=settings.grounding_openai_max_output_tokens,
            anthropic_api_key=settings.anthropic_api_key,
            anthropic_timeout_seconds=settings.anthropic_timeout_seconds,
            anthropic_max_tokens=settings.grounding_anthropic_max_tokens,
            max_iterations=settings.grounding_qa_max_iterations,
            max_bbox_delta_px=settings.grounding_qa_max_bbox_delta_px,
            consensus_translation_enabled=settings.grounding_qa_consensus_translation_enabled,
            consensus_min_fields=settings.grounding_qa_consensus_min_fields,
            consensus_max_spread_px=settings.grounding_qa_consensus_max_spread_px,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    iteration_models: list[list[RefineGroundingIterationPage]] = []
    for outer in raw["iterations"]:
        row: list[RefineGroundingIterationPage] = []
        for p in outer:
            row.append(
                RefineGroundingIterationPage(
                    page_index=p["page_index"],
                    qa_status=p["qa_status"],
                    corrections_requested=p.get("corrections_requested", 0),
                    preview_image=p.get("preview_image"),
                    note=p.get("note"),
                )
            )
        iteration_models.append(row)

    return RefineGroundingResponse(
        job_id=raw["job_id"],
        provider=raw["provider"],
        model=raw["model"],
        session_id=raw["session_id"],
        promoted=raw["promoted"],
        stopped_reason=raw["stopped_reason"],
        iterations_run=raw["iterations_run"],
        refined_dir=raw["refined_dir"],
        qa_session_dir=raw["qa_session_dir"],
        final_preview_dir=raw["final_preview_dir"],
        canonical_grounding_updated=raw["canonical_grounding_updated"],
        iterations=iteration_models,
    )


@router.post(
    "/convert-and-ground/anthropic",
    response_model=ConvertAndGroundResponse,
    summary="Upload, convert, then ground with Anthropic in one request",
    responses={
        400: {"description": "Invalid PDF upload or request parameters"},
        413: {"description": "Upload too large"},
        422: {"description": "Conversion or grounding failed"},
    },
)
async def convert_and_ground_anthropic(
    file: UploadFile = File(..., description="Input PDF"),
    dpi: float = Form(200.0, ge=1.0, le=1200.0, description="Rasterization DPI"),
    allow_rotated_pages: bool = Form(
        False,
        description="If true, allow non-zero page.rotation (see manifest mapping.status).",
    ),
    settings: Settings = Depends(get_settings),
) -> ConvertAndGroundResponse:
    return await _convert_and_ground_for_provider(
        file=file,
        dpi=dpi,
        allow_rotated_pages=allow_rotated_pages,
        provider="anthropic",
        model=settings.combined_default_anthropic_model.strip(),
        settings=settings,
    )


@router.post(
    "/convert-and-ground/openai",
    response_model=ConvertAndGroundResponse,
    summary="Upload, convert, then ground with OpenAI in one request",
    responses={
        400: {"description": "Invalid PDF upload or request parameters"},
        413: {"description": "Upload too large"},
        422: {"description": "Conversion or grounding failed"},
    },
)
async def convert_and_ground_openai(
    file: UploadFile = File(..., description="Input PDF"),
    dpi: float = Form(200.0, ge=1.0, le=1200.0, description="Rasterization DPI"),
    allow_rotated_pages: bool = Form(
        False,
        description="If true, allow non-zero page.rotation (see manifest mapping.status).",
    ),
    settings: Settings = Depends(get_settings),
) -> ConvertAndGroundResponse:
    return await _convert_and_ground_for_provider(
        file=file,
        dpi=dpi,
        allow_rotated_pages=allow_rotated_pages,
        provider="openai",
        model=settings.combined_default_openai_model.strip(),
        settings=settings,
    )


async def _convert_and_ground_for_provider(
    *,
    file: UploadFile,
    dpi: float,
    allow_rotated_pages: bool,
    provider: str,
    model: str,
    settings: Settings,
) -> ConvertAndGroundResponse:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        await file.close()
        raise HTTPException(status_code=400, detail="Upload filename must end with .pdf")

    job_id = str(uuid.uuid4())
    root, input_pdf, output_dir = job_paths(settings.jobs_dir, job_id)
    root.mkdir(parents=True, exist_ok=True)

    try:
        await _save_upload_limited(file, input_pdf, max_bytes=settings.max_upload_bytes)

        with input_pdf.open("rb") as handle:
            header = handle.read(5)
        if header != b"%PDF-":
            raise HTTPException(status_code=400, detail="File does not look like a PDF (missing %PDF- header).")

        try:
            convert_result = await asyncio.to_thread(
                run_convert_pdf_to_images,
                str(input_pdf),
                str(output_dir),
                dpi,
                overwrite=True,
                allow_rotated_pages=allow_rotated_pages,
                job_id=job_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        doc = _write_provider_metadata_to_document_manifest(
            output_dir=output_dir,
            provider=provider,
            model=model,
        )
        page_count = len(convert_result.get("pages", []))
        convert_payload = ConvertResponse(
            job_id=job_id,
            page_count=page_count,
            dpi=dpi,
            allow_rotated_pages=allow_rotated_pages,
            source_filename=filename,
            document_manifest=doc,
        )

        try:
            ground_result = await asyncio.to_thread(
                run_field_grounding_for_job,
                job_id=job_id,
                output_dir=output_dir,
                provider=provider,
                model=model,
                openai_api_key=settings.openai_api_key,
                openai_timeout_seconds=settings.openai_timeout_seconds,
                openai_max_output_tokens=settings.grounding_openai_max_output_tokens,
                anthropic_api_key=settings.anthropic_api_key,
                anthropic_timeout_seconds=settings.anthropic_timeout_seconds,
                anthropic_max_tokens=settings.grounding_anthropic_max_tokens,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        grounding_payload = GroundFieldsResponse(**ground_result)
        if ground_result["succeeded_count"] == 0:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Convert succeeded but grounding failed for all pages.",
                    "job_id": job_id,
                    "convert": convert_payload.model_dump(),
                    "grounding": {
                        "provider": provider,
                        "model": model,
                        "page_count": ground_result["page_count"],
                        "failed_count": ground_result["failed_count"],
                        "pages": ground_result["pages"],
                    },
                },
            )

        LOG.info(
            "Convert+ground provider=%s model=%s job=%s pages=%d file=%s",
            provider,
            model,
            job_id,
            page_count,
            filename,
        )
        return ConvertAndGroundResponse(
            job_id=job_id,
            convert=convert_payload,
            grounding=grounding_payload,
        )
    except HTTPException:
        _cleanup_job_dir(root)
        raise
    except Exception:
        _cleanup_job_dir(root)
        raise


@router.post(
    "/jobs/{job_id}/stamp-images/anthropic",
    response_model=StampImagesResponse,
    summary="Stamp values onto images using Anthropic grounding JSON",
    responses={
        400: {"description": "Invalid job, missing converted images, or missing grounding run"},
        404: {"description": "Job not found"},
        422: {"description": "No pages succeeded image stamping"},
    },
)
async def stamp_images_anthropic_for_job(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> StampImagesResponse:
    return await _stamp_images_for_provider(
        job_id=job_id,
        route_provider="anthropic",
        settings=settings,
    )


@router.post(
    "/jobs/{job_id}/stamp-images/openai",
    response_model=StampImagesResponse,
    summary="Stamp values onto images using OpenAI grounding JSON",
    responses={
        400: {"description": "Invalid job, missing converted images, or missing grounding run"},
        404: {"description": "Job not found"},
        422: {"description": "No pages succeeded image stamping"},
    },
)
async def stamp_images_openai_for_job(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> StampImagesResponse:
    return await _stamp_images_for_provider(
        job_id=job_id,
        route_provider="openai",
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
    "/jobs/{job_id}/stamp-pdf/anthropic",
    response_model=StampPdfResponse,
    summary="Stamp values onto original PDF using Anthropic grounding JSON",
    responses={
        400: {"description": "Invalid job, missing converted manifests, or missing grounding run"},
        404: {"description": "Job not found"},
        422: {"description": "No pages succeeded PDF stamping"},
    },
)
async def stamp_pdf_anthropic_for_job(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> StampPdfResponse:
    return await _stamp_pdf_for_provider(
        job_id=job_id,
        route_provider="anthropic",
        settings=settings,
    )


@router.post(
    "/jobs/{job_id}/stamp-pdf/openai",
    response_model=StampPdfResponse,
    summary="Stamp values onto original PDF using OpenAI grounding JSON",
    responses={
        400: {"description": "Invalid job, missing converted manifests, or missing grounding run"},
        404: {"description": "Job not found"},
        422: {"description": "No pages succeeded PDF stamping"},
    },
)
async def stamp_pdf_openai_for_job(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> StampPdfResponse:
    return await _stamp_pdf_for_provider(
        job_id=job_id,
        route_provider="openai",
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
