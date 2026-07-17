"""E3 job lifecycle, fields, and export routes."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from app.api_tags import TAG_FILL_EXPORT, TAG_JOBS
from app.config import Settings
from app.dependencies import get_settings
from app.http_errors import ApiHttpError, job_not_found
from app.schemas import (
    CreateJobResponse,
    FieldsResponse,
    JobDetailResponse,
    JobListResponse,
    PatchFieldsRequest,
    PatchFieldsResponse,
    PatchValuesRequest,
    PatchValuesResponse,
    RefineGroundingResponse,
    StampImagesRunResponse,
    StampPdfRunResponse,
    StampingJson,
)
from app.services.fields_service import (
    load_fields_payload,
    patch_fields,
    patch_values,
    resolve_export_pdf_path,
    resolve_page_image_path,
)
from app.services.image_stamping import run_image_stamping_for_job
from app.services.job_pipeline import (
    delete_job_tree,
    detect_upload_pdf_type,
    run_full_job_pipeline,
    validate_upload_pdf_bytes,
)
from app.services.jobs import (
    assert_valid_job_id,
    job_detail_projection,
    job_paths,
    list_job_summaries,
    new_job_manifest,
    read_job_manifest,
    write_job_manifest,
)
from app.services.pdf_stamping import run_pdf_stamping_for_job
from app.services.stamping_config import (
    load_job_grounding_info,
    load_stamping_json_parsed,
    stamping_overrides,
    stamping_style,
)

LOG = logging.getLogger(__name__)

router = APIRouter(tags=[TAG_JOBS, TAG_FILL_EXPORT])


def _resolve_job(settings: Settings, job_id: str) -> tuple[Path, Path, Path]:
    try:
        assert_valid_job_id(job_id)
        root, input_pdf, output_dir = job_paths(settings.jobs_dir, job_id)
    except ValueError as exc:
        raise ApiHttpError(400, "invalid_job_id", "job_id must be a canonical UUID string") from exc
    if not root.is_dir():
        raise job_not_found(job_id)
    return root, input_pdf, output_dir


def _load_stamping(output_dir: Path) -> StampingJson:
    try:
        return load_stamping_json_parsed(output_dir)
    except FileNotFoundError as exc:
        raise ApiHttpError(400, "stamping_not_found", "field_grounding/stamping.json not found") from exc
    except ValueError as exc:
        raise ApiHttpError(400, "invalid_stamping_json", "Invalid stamping.json") from exc
    except ValidationError as exc:
        raise ApiHttpError(400, "invalid_stamping_json", "Invalid stamping.json") from exc


@router.post(
    "/jobs",
    response_model=CreateJobResponse,
    status_code=201,
    summary="Upload a PDF and start the background processing pipeline",
)
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> CreateJobResponse:
    raw_name = file.filename or "upload.pdf"
    if not raw_name.lower().endswith(".pdf"):
        raise ApiHttpError(400, "invalid_pdf", "Upload must be a PDF file.")

    data = await file.read()
    validate_upload_pdf_bytes(data, max_bytes=settings.max_upload_bytes)
    detected_type = detect_upload_pdf_type(data)

    job_id = str(uuid.uuid4())
    root, input_pdf, output_dir = job_paths(settings.jobs_dir, job_id)
    root.mkdir(parents=True, exist_ok=True)
    write_job_manifest(
        root,
        new_job_manifest(
            job_id=job_id,
            source_filename=Path(raw_name).name,
            dpi=200,
            detected_pdf_type=detected_type,
        ),
    )
    input_pdf.write_bytes(data)
    output_dir.mkdir(parents=True, exist_ok=True)

    background_tasks.add_task(
        run_full_job_pipeline,
        job_id=job_id,
        job_root=root,
        input_pdf=input_pdf,
        output_dir=output_dir,
        settings=settings,
        source_filename=Path(raw_name).name,
    )

    return CreateJobResponse(job_id=job_id, status="converting")


@router.get("/jobs", response_model=JobListResponse, summary="List all jobs")
async def list_jobs(settings: Settings = Depends(get_settings)) -> JobListResponse:
    return JobListResponse(jobs=list_job_summaries(settings.jobs_dir))


@router.get("/jobs/{job_id}", response_model=JobDetailResponse, summary="Get job status for polling")
async def get_job(job_id: str, settings: Settings = Depends(get_settings)) -> JobDetailResponse:
    root, _, _ = _resolve_job(settings, job_id)
    manifest = read_job_manifest(root)
    return JobDetailResponse(**job_detail_projection(manifest))


@router.delete("/jobs/{job_id}", status_code=204, summary="Delete a job and all artifacts")
async def delete_job(job_id: str, settings: Settings = Depends(get_settings)) -> Response:
    root, _, _ = _resolve_job(settings, job_id)
    delete_job_tree(root)
    return Response(status_code=204)


@router.get(
    "/jobs/{job_id}/pages/{page_number}/image",
    summary="Get a page PNG for the editor canvas",
    responses={200: {"content": {"image/png": {}}}},
)
async def get_page_image(
    job_id: str,
    page_number: int,
    variant: str = Query(default="source", pattern="^(source|stamped)$"),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    if page_number < 1:
        raise ApiHttpError(400, "invalid_page_number", "page_number must be >= 1.")

    root, _, output_dir = _resolve_job(settings, job_id)
    try:
        image_path = resolve_page_image_path(
            output_dir=output_dir,
            job_root=root,
            page_number=page_number,
            variant=variant,
        )
    except ValueError as exc:
        if "escapes job output directory" in str(exc):
            raise ApiHttpError(400, "invalid_artifact_path", "Job artifact path is not allowed") from exc
        raise ApiHttpError(400, "invalid_request", "Invalid page image request") from exc
    except FileNotFoundError as exc:
        raise ApiHttpError(404, "image_not_found", "Page image not found") from exc

    if not image_path.is_file():
        raise ApiHttpError(404, "image_not_found", f"Page image not found for page {page_number}.")

    return FileResponse(image_path, media_type="image/png")


@router.get("/jobs/{job_id}/fields", response_model=FieldsResponse, summary="Get all fields and values")
async def get_fields(job_id: str, settings: Settings = Depends(get_settings)) -> FieldsResponse:
    root, _, output_dir = _resolve_job(settings, job_id)
    try:
        payload = load_fields_payload(job_id=job_id, output_dir=output_dir)
    except FileNotFoundError as exc:
        raise ApiHttpError(400, "fields_not_found", "Grounded fields not found for this job") from exc
    except ValueError as exc:
        raise ApiHttpError(400, "invalid_fields", "Invalid grounded fields data") from exc
    return FieldsResponse(**payload)


@router.patch("/jobs/{job_id}/fields", response_model=PatchFieldsResponse, summary="Update field geometry")
async def patch_job_fields(
    job_id: str,
    body: PatchFieldsRequest,
    settings: Settings = Depends(get_settings),
) -> PatchFieldsResponse:
    _, _, output_dir = _resolve_job(settings, job_id)
    try:
        updated = patch_fields(
            output_dir=output_dir,
            field_updates=[item.model_dump(exclude_none=True) for item in body.fields],
        )
    except FileNotFoundError as exc:
        raise ApiHttpError(404, "field_not_found", "Field not found on the requested page") from exc
    except ValueError as exc:
        raise ApiHttpError(400, "invalid_field_patch", "Invalid field patch payload") from exc

    # E7: a save through the editor is the human-correction signal — capture the touched
    # pages as reusable templates. Best-effort; a capture failure never breaks the save.
    if settings.template_memory_enabled:
        try:
            from app.services.template_memory import capture_corrected_pages

            page_indices = sorted({item.page_number - 1 for item in body.fields})
            capture_corrected_pages(output_dir, settings, page_indices=page_indices)
        except Exception:  # noqa: BLE001 - capture is advisory, never user-facing
            LOG.warning("template capture after PATCH fields failed for job %s", job_id, exc_info=True)

    return PatchFieldsResponse(fields=updated)


@router.patch("/jobs/{job_id}/values", response_model=PatchValuesResponse, summary="Update field values")
async def patch_job_values(
    job_id: str,
    body: PatchValuesRequest,
    settings: Settings = Depends(get_settings),
) -> PatchValuesResponse:
    _, _, output_dir = _resolve_job(settings, job_id)
    try:
        result = patch_values(output_dir=output_dir, values=body.values, style=body.style)
    except FileNotFoundError as exc:
        raise ApiHttpError(404, "stamping_not_found", "field_grounding/stamping.json not found") from exc
    except ApiHttpError:
        raise
    except ValueError as exc:
        raise ApiHttpError(400, "invalid_values_patch", "Invalid values patch payload") from exc
    return PatchValuesResponse(**result)


@router.post(
    "/jobs/{job_id}/stamp-images",
    response_model=StampImagesRunResponse,
    summary="Generate server-rendered preview PNGs",
)
async def stamp_images(job_id: str, settings: Settings = Depends(get_settings)) -> StampImagesRunResponse:
    root, _, output_dir = _resolve_job(settings, job_id)
    try:
        provider, model = load_job_grounding_info(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ApiHttpError(400, "grounding_not_found", "Job grounding metadata not found") from exc

    stamping = _load_stamping(output_dir)
    style = stamping_style(stamping)
    overrides = stamping_overrides(stamping)

    try:
        result = await asyncio.to_thread(
            run_image_stamping_for_job,
            job_id=job_id,
            output_dir=output_dir,
            provider=provider,
            model=model,
            values=stamping.values,
            style=style,
            overrides=overrides,
            require_all_values=stamping.require_all_values,
        )
    except FileNotFoundError as exc:
        raise ApiHttpError(400, "stamp_failed", "Stamping prerequisites not found") from exc
    except ValueError as exc:
        raise ApiHttpError(400, "stamp_failed", "Image stamping failed") from exc

    if result["succeeded_count"] == 0:
        raise ApiHttpError(422, "stamp_failed", "Image stamping failed for all pages.")

    pages = [
        {
            "page_number": page["page_index"] + 1,
            "image_url": f"/api/v1/jobs/{job_id}/pages/{page['page_index'] + 1}/image?variant=stamped",
        }
        for page in result["pages"]
        if page.get("status") == "succeeded"
    ]
    return StampImagesRunResponse(run_id=result["stamp_run_id"], pages=pages)


@router.post(
    "/jobs/{job_id}/stamp-pdf",
    response_model=StampPdfRunResponse,
    summary="Export a flattened PDF",
)
async def stamp_pdf(job_id: str, settings: Settings = Depends(get_settings)) -> StampPdfRunResponse:
    root, input_pdf, output_dir = _resolve_job(settings, job_id)
    if not input_pdf.is_file():
        raise ApiHttpError(400, "input_pdf_missing", f"Input PDF not found for job: {job_id}")

    try:
        provider, model = load_job_grounding_info(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ApiHttpError(400, "grounding_not_found", "Job grounding metadata not found") from exc

    stamping = _load_stamping(output_dir)
    style = stamping_style(stamping)
    overrides = stamping_overrides(stamping)

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
            overrides=overrides,
            require_all_values=stamping.require_all_values,
        )
    except FileNotFoundError as exc:
        raise ApiHttpError(400, "stamp_failed", "Stamping prerequisites not found") from exc
    except ValueError as exc:
        raise ApiHttpError(400, "stamp_failed", "PDF stamping failed") from exc

    return StampPdfRunResponse(
        run_id=result["stamp_run_id"],
        download_url=f"/api/v1/jobs/{job_id}/export",
    )


@router.post(
    "/jobs/{job_id}/refine-grounding",
    response_model=RefineGroundingResponse,
    status_code=202,
    summary="Run the vision QA refinement loop (E5)",
)
async def refine_grounding(
    job_id: str,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> RefineGroundingResponse:
    root, _, output_dir = _resolve_job(settings, job_id)
    if not (output_dir / "field_grounding").is_dir():
        raise ApiHttpError(400, "fields_not_found", "Grounded fields not found for this job")

    from app.services.grounding_qa import run_refine_grounding_task

    background_tasks.add_task(
        run_refine_grounding_task,
        job_id=job_id,
        job_root=root,
        output_dir=output_dir,
        settings=settings,
    )
    return RefineGroundingResponse(status="running")


@router.get(
    "/jobs/{job_id}/export",
    summary="Download the latest stamped PDF",
    responses={200: {"content": {"application/pdf": {}}}},
)
async def export_pdf(job_id: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    root, _, output_dir = _resolve_job(settings, job_id)
    try:
        pdf_path = resolve_export_pdf_path(output_dir=output_dir, job_root=root)
    except ValueError as exc:
        if "escapes job output directory" in str(exc):
            raise ApiHttpError(400, "invalid_artifact_path", "Job artifact path is not allowed") from exc
        raise
    except FileNotFoundError as exc:
        raise ApiHttpError(404, "export_not_found", "Exported PDF not found") from exc

    if not pdf_path.is_file():
        raise ApiHttpError(404, "export_not_found", "Exported PDF file is missing on disk.")

    return FileResponse(pdf_path, media_type="application/pdf", filename="form.pdf")
