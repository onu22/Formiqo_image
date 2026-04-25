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
from app.schemas import ConvertResponse, GroundFieldsRequest, GroundFieldsResponse
from app.services.conversion import run_convert_pdf_to_images
from app.services.field_grounding import run_field_grounding_for_job
from app.services.jobs import job_paths, read_document_manifest, zip_output_folder

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
