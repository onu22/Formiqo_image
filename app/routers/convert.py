"""PDF upload → images + manifests."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from app.config import Settings
from app.dependencies import get_settings
from app.schemas import (
    ConvertResponse,
    GroundFieldsRequest,
    GroundFieldsResponse,
    PrepareStampRequest,
    PrepareStampResponse,
    StampPdfRequest,
    StampPdfResponse,
)
from app.services.conversion import run_convert_pdf_to_images
from app.services.field_grounding import run_field_grounding_for_job
from app.services.jobs import job_paths, read_document_manifest, zip_output_folder
from app.services.stamp_writer import (
    StampOptions,
    resolve_stamped_pdf_path,
    run_stamp_pdf_for_job,
)
from app.services.stamping import run_prepare_stamp_for_job

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


@router.post(
    "/convert",
    response_model=ConvertResponse,
    summary="Convert PDF to page images and manifests",
    responses={
        400: {"description": "Invalid PDF or conversion parameters"},
        413: {"description": "Upload too large"},
        422: {"description": "Conversion failed (e.g. uniform scale check)"},
    },
)
async def convert_pdf(
    request: Request,
    file: UploadFile = File(..., description="Input PDF"),
    dpi: float = Form(200.0, ge=1.0, le=1200.0, description="Rasterization DPI"),
    allow_rotated_pages: bool = Form(
        False,
        description="If true, allow non-zero page.rotation (see manifest mapping.status).",
    ),
    settings: Settings = Depends(get_settings),
) -> ConvertResponse:
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
            result = await asyncio.to_thread(
                run_convert_pdf_to_images,
                str(input_pdf),
                str(output_dir),
                dpi,
                overwrite=True,
                allow_rotated_pages=allow_rotated_pages,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        doc = read_document_manifest(output_dir)
        page_count = len(result.get("pages", []))
        base = str(request.base_url).rstrip("/")
        links = {
            "archive_zip": f"{base}/api/v1/jobs/{job_id}/archive.zip",
            "document_manifest_json": f"{base}/api/v1/jobs/{job_id}/document_manifest.json",
        }

        LOG.info(
            "Converted job=%s pages=%d dpi=%s file=%s",
            job_id,
            page_count,
            dpi,
            filename,
        )

        return ConvertResponse(
            job_id=job_id,
            page_count=page_count,
            dpi=dpi,
            allow_rotated_pages=allow_rotated_pages,
            source_filename=filename,
            document_manifest=doc,
            links=links,
        )
    except HTTPException:
        _cleanup_job_dir(root)
        raise
    except Exception:
        _cleanup_job_dir(root)
        raise


@router.get(
    "/jobs/{job_id}/archive.zip",
    summary="Download conversion output as a zip",
    response_class=Response,
    responses={404: {"description": "Job or output not found"}},
)
async def download_archive(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> Response:
    try:
        _, _, output_dir = job_paths(settings.jobs_dir, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        data = await asyncio.to_thread(zip_output_folder, output_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
    )


@router.get(
    "/jobs/{job_id}/document_manifest.json",
    summary="Download document_manifest.json for a job",
    responses={404: {"description": "Job or manifest not found"}},
)
async def download_document_manifest(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    try:
        _, _, output_dir = job_paths(settings.jobs_dir, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    path = output_dir / "document_manifest.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="document_manifest.json not found for this job_id")

    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@router.post(
    "/jobs/{job_id}/ground-fields",
    response_model=GroundFieldsResponse,
    summary="Ground form fields from existing converted page images",
    responses={
        400: {"description": "Job exists but converted images are missing"},
        404: {"description": "Job not found"},
        422: {"description": "No pages succeeded grounding"},
    },
)
async def ground_fields_for_job(
    job_id: str,
    request_body: GroundFieldsRequest | None = None,
    settings: Settings = Depends(get_settings),
) -> GroundFieldsResponse:
    try:
        root, _, output_dir = job_paths(settings.jobs_dir, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if not output_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Job output folder not found: {output_dir}")

    converted_images_dir = output_dir / "converted_images"
    if not converted_images_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Converted images folder not found: {converted_images_dir}",
        )

    try:
        provider = (
            request_body.provider.strip().lower()
            if request_body and request_body.provider
            else settings.grounding_provider.strip().lower()
        )
        model = (
            request_body.model.strip()
            if request_body and request_body.model
            else settings.grounding_model.strip()
        )
        result = await asyncio.to_thread(
            run_field_grounding_for_job,
            job_id=job_id,
            output_dir=output_dir,
            provider=provider,
            model=model,
            openai_api_key=settings.openai_api_key,
            openai_timeout_seconds=settings.openai_timeout_seconds,
            anthropic_api_key=settings.anthropic_api_key,
            anthropic_timeout_seconds=settings.anthropic_timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result["succeeded_count"] == 0:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Field grounding failed for all pages.",
                "job_id": job_id,
                "page_count": result["page_count"],
                "failed_count": result["failed_count"],
                "pages": result["pages"],
            },
        )

    return GroundFieldsResponse(**result)


@router.post(
    "/jobs/{job_id}/prepare-stamp",
    response_model=PrepareStampResponse,
    summary="Map grounded image bboxes to PDF coordinates and emit a stamp plan",
    responses={
        400: {"description": "Invalid job, missing output folder, or missing grounding run"},
        404: {"description": "Job not found"},
        422: {"description": "No pages succeeded stamp-plan generation"},
    },
)
async def prepare_stamp_for_job(
    job_id: str,
    request_body: PrepareStampRequest | None = None,
    settings: Settings = Depends(get_settings),
) -> PrepareStampResponse:
    try:
        root, input_pdf, output_dir = job_paths(settings.jobs_dir, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if not output_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Job output folder not found: {output_dir}")

    provider = (
        request_body.provider.strip().lower()
        if request_body and request_body.provider
        else settings.grounding_provider.strip().lower()
    )
    model = (
        request_body.model.strip()
        if request_body and request_body.model
        else settings.grounding_model.strip()
    )

    try:
        result = await asyncio.to_thread(
            run_prepare_stamp_for_job,
            job_id=job_id,
            output_dir=output_dir,
            source_pdf=str(input_pdf),
            provider=provider,
            model=model,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result["succeeded_count"] == 0:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Stamp plan generation failed for all pages.",
                "job_id": job_id,
                "provider": result["provider"],
                "model": result["model"],
                "page_count": result["page_count"],
                "failed_count": result["failed_count"],
                "pages": result["pages"],
            },
        )

    return PrepareStampResponse(**result)


def _resolve_stamp_provider_model(
    body: StampPdfRequest | None,
    settings: Settings,
) -> tuple[str, str]:
    provider = (
        body.provider.strip().lower()
        if body and body.provider
        else settings.grounding_provider.strip().lower()
    )
    model = (
        body.model.strip()
        if body and body.model
        else settings.grounding_model.strip()
    )
    if not provider:
        raise HTTPException(status_code=400, detail="provider must not be empty.")
    if not model:
        raise HTTPException(status_code=400, detail="model must not be empty.")
    return provider, model


@router.post(
    "/jobs/{job_id}/stamp-pdf",
    response_model=StampPdfResponse,
    summary="Stamp values onto the original PDF using the stamp plan",
    responses={
        400: {"description": "Invalid job, missing stamp plan, or strict-mode violation"},
        404: {"description": "Job not found"},
        422: {"description": "Failed to save stamped PDF"},
    },
)
async def stamp_pdf_for_job(
    job_id: str,
    request: Request,
    request_body: StampPdfRequest | None = None,
    settings: Settings = Depends(get_settings),
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
        raise HTTPException(status_code=400, detail=f"Source PDF not found: {input_pdf}")

    provider, model = _resolve_stamp_provider_model(request_body, settings)

    values = dict(request_body.values) if request_body and request_body.values else {}
    strict = bool(request_body.strict) if request_body else False

    try:
        stamp_options = (
            StampOptions(
                fontsize=float(request_body.options.fontsize),
                fontname=request_body.options.fontname,
                color_rgb=tuple(float(c) for c in request_body.options.color_rgb),
                align=request_body.options.align,
                autoshrink=bool(request_body.options.autoshrink),
                min_fontsize=float(request_body.options.min_fontsize),
            )
            if request_body and request_body.options is not None
            else None
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid stamp options: {exc}") from exc

    try:
        result = await asyncio.to_thread(
            run_stamp_pdf_for_job,
            job_id=job_id,
            input_pdf=input_pdf,
            output_dir=output_dir,
            provider=provider,
            model=model,
            values=values,
            strict=strict,
            options=stamp_options,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    base = str(request.base_url).rstrip("/")
    download_url = (
        f"{base}/api/v1/jobs/{job_id}/stamped.pdf"
        f"?provider={provider}&model={model}"
    )

    return StampPdfResponse(
        download_url=download_url,
        **result,
    )


@router.get(
    "/jobs/{job_id}/stamped.pdf",
    summary="Download the stamped PDF for a given provider/model run",
    response_class=Response,
    responses={
        400: {"description": "Invalid job id or missing provider/model"},
        404: {"description": "Stamped PDF not found for this provider/model"},
    },
)
async def download_stamped_pdf(
    job_id: str,
    provider: str | None = None,
    model: str | None = None,
    settings: Settings = Depends(get_settings),
) -> Response:
    try:
        root, _, output_dir = job_paths(settings.jobs_dir, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    resolved_provider = (provider or settings.grounding_provider).strip().lower()
    resolved_model = (model or settings.grounding_model).strip()
    if not resolved_provider or not resolved_model:
        raise HTTPException(status_code=400, detail="provider and model must be provided.")

    path = resolve_stamped_pdf_path(output_dir, resolved_provider, resolved_model)
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Stamped PDF not found for provider={resolved_provider}, model={resolved_model}. "
                f"Run POST /stamp-pdf first."
            ),
        )

    data = await asyncio.to_thread(path.read_bytes)
    return Response(
        content=data,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="stamped.pdf"'},
    )
