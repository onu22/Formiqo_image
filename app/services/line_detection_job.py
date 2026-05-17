"""Run OpenCV form line detection across all pages under a job ``output_dir``."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from app.schemas import (
    FormLineDetectionJobResponse,
    FormLineDetectionPageResult,
    FormLineDetectionResponse,
)
from app.services.jobs import to_output_relative_path
from app.services.line_detector import compute_raw_line_detection, write_raw_line_detection

LOG = logging.getLogger(__name__)

_PAGE_PNG_RE = re.compile(r"^page_(\d+)\.png$", re.IGNORECASE)

NO_CONVERTED_PAGE_PNGS = "NO_CONVERTED_PAGE_PNGS"


def list_converted_page_pngs(output_dir: Path) -> list[tuple[int, Path]]:
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


def write_line_detection_for_page(
    *,
    output_dir: Path,
    in_path: Path,
    line_detection_page_dir: Path,
    detector_config: dict[str, int],
) -> dict:
    """Run raw line detection and write JSON + highlight PNG; return raw payload (no ``output_files``)."""
    line_detection_page_dir.mkdir(parents=True, exist_ok=True)
    raw_json = line_detection_page_dir / "detected_lines.json"
    raw_png = line_detection_page_dir / "lines_highlighted.png"

    image_rel = to_output_relative_path(output_dir, in_path)
    raw, bgr = compute_raw_line_detection(
        str(in_path),
        detector_config,
        image_path_in_json=image_rel,
    )
    write_raw_line_detection(raw, bgr, str(raw_json), str(raw_png))
    return raw


def run_detect_form_lines_for_job_output_dir(
    *,
    job_id: str,
    output_dir: Path,
    detector_config: dict[str, int],
) -> FormLineDetectionJobResponse:
    """
    Synchronous line detection for every ``converted_images/page_*.png``.

    Raises ``FileNotFoundError`` if ``converted_images`` is missing,
    ``ValueError`` with :data:`NO_CONVERTED_PAGE_PNGS` if no matching PNGs exist,
    or ``ValueError`` / ``OSError`` from the detector when a page fails.
    """
    page_entries = list_converted_page_pngs(output_dir)
    if not page_entries:
        raise ValueError(NO_CONVERTED_PAGE_PNGS)

    pages: list[FormLineDetectionPageResult] = []
    for page_index, in_path in page_entries:
        stem = f"page_{page_index + 1:04d}"
        out_dir = output_dir / "line_detection" / stem
        raw = write_line_detection_for_page(
            output_dir=output_dir,
            in_path=in_path,
            line_detection_page_dir=out_dir,
            detector_config=detector_config,
        )
        rel_json = str((out_dir / "detected_lines.json").relative_to(output_dir)).replace("\\", "/")
        rel_png = str((out_dir / "lines_highlighted.png").relative_to(output_dir)).replace("\\", "/")
        detection = FormLineDetectionResponse.model_validate(
            {
                **raw,
                "output_files": {"json": rel_json, "highlight_png": rel_png},
            }
        )
        pages.append(FormLineDetectionPageResult(page_index=page_index, detection=detection))

    total_lines = sum(p.detection.counts.total for p in pages)
    LOG.info(
        "detect-form-lines job=%s pages=%d total_raw_lines=%d",
        job_id,
        len(pages),
        total_lines,
    )
    return FormLineDetectionJobResponse(job_id=job_id, page_count=len(pages), pages=pages)
