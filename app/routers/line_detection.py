"""HTTP routes for OpenCV form line detection on converted job pages."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException

from app.config import Settings
from app.dependencies import get_settings
from app.schemas import (
    DetectFormLinesJobRequest,
    FormLineDetectionJobResponse,
    FormLineDetectionPageResult,
    FormLineDetectionResponse,
    FormLineDetectorConfig,
)
from app.services.jobs import job_paths
from app.services.line_detector import compute_raw_line_detection, write_raw_line_detection

LOG = logging.getLogger(__name__)

router = APIRouter(tags=["line-detection"])

_PAGE_PNG_RE = re.compile(r"^page_(\d+)\.png$", re.IGNORECASE)


def _list_converted_page_pngs(output_dir: Path) -> list[tuple[int, Path]]:
    images_dir = output_dir / "converted_images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"converted_images directory not found: {images_dir}")

    pages: list[tuple[int, Path]] = []
    for path in sorted(images_dir.glob("page_*.png")):
        m = _PAGE_PNG_RE.match(path.name)
        if not m:
            continue
        page_index = int(m.group(1)) - 1
        pages.append((page_index, path))
    pages.sort(key=lambda t: t[0])
    return pages


def _effective_detector_config(body: DetectFormLinesJobRequest) -> dict[str, int]:
    if body.config is not None:
        return body.config.to_detector_dict()
    return FormLineDetectorConfig().to_detector_dict()


def _process_page_sync(
    *,
    in_path: Path,
    out_dir: Path,
    detector_config: dict[str, int],
) -> dict:
    """Run raw line detection and write JSON + highlight PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_json = out_dir / "detected_lines.json"
    raw_png = out_dir / "lines_highlighted.png"

    raw, bgr = compute_raw_line_detection(str(in_path), detector_config)
    write_raw_line_detection(raw, bgr, str(raw_json), str(raw_png))
    return raw


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

    try:
        page_entries = _list_converted_page_pngs(output_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not page_entries:
        raise HTTPException(
            status_code=400,
            detail="No converted page PNGs found under converted_images/ (expected page_0001.png, …).",
        )

    det_cfg = _effective_detector_config(body)

    async def _run_one(page_index: int, in_path: Path) -> FormLineDetectionPageResult:
        stem = f"page_{page_index + 1:04d}"
        out_dir = output_dir / "line_detection" / stem

        try:
            raw = await asyncio.to_thread(
                _process_page_sync,
                in_path=in_path,
                out_dir=out_dir,
                detector_config=det_cfg,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"message": str(exc), "job_id": job_id, "page_index": page_index},
            ) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=422,
                detail={"message": str(exc), "job_id": job_id, "page_index": page_index},
            ) from exc

        rel_json = str((out_dir / "detected_lines.json").relative_to(output_dir)).replace("\\", "/")
        rel_png = str((out_dir / "lines_highlighted.png").relative_to(output_dir)).replace("\\", "/")

        detection = FormLineDetectionResponse.model_validate(
            {
                **raw,
                "output_files": {"json": rel_json, "highlight_png": rel_png},
            }
        )
        return FormLineDetectionPageResult(page_index=page_index, detection=detection)

    results = await asyncio.gather(*[_run_one(idx, p) for idx, p in page_entries])
    pages = list(results)

    total_lines = sum(p.detection.counts.total for p in pages)
    LOG.info(
        "detect-form-lines job=%s pages=%d total_raw_lines=%d",
        job_id,
        len(pages),
        total_lines,
    )

    return FormLineDetectionJobResponse(job_id=job_id, page_count=len(pages), pages=pages)
