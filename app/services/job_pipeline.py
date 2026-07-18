"""Background job pipeline: convert → line-detect → semantic grounding."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from app.config import Settings
from app.schemas import FormLineDetectorConfig
from app.services.convert_and_ground_job import run_convert_sync
from app.services.jobs import (
    read_job_manifest,
    set_job_status,
    update_job_after_convert,
    update_job_after_line_detect,
    update_job_stage,
    write_job_manifest,
)
from app.services.line_detection_job import run_detect_form_lines_for_job_output_dir
from app.services.pdf_pipeline.detector import PdfTypeDetector
from app.services.pdf_pipeline.errors import PdfPipelineError, XFA_UNSUPPORTED_USER_MESSAGE
from app.services.pdf_pipeline.types import PdfPipelineKind
from app.services.semantic_grounding import SemanticGroundingJobError, run_semantic_grounding_for_job

LOG = logging.getLogger(__name__)


def validate_upload_pdf_bytes(data: bytes, *, max_bytes: int) -> None:
    from app.http_errors import ApiHttpError

    if len(data) > max_bytes:
        raise ApiHttpError(
            413,
            "file_too_large",
            f"Upload exceeds maximum size of {max_bytes} bytes.",
        )
    if not data.startswith(b"%PDF-"):
        raise ApiHttpError(400, "invalid_pdf", "File does not look like a PDF (missing %PDF- header).")


def detect_upload_pdf_type(data: bytes) -> str:
    from app.http_errors import ApiHttpError

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        kind = PdfTypeDetector.detect(tmp_path)
        if kind == PdfPipelineKind.XFA:
            raise ApiHttpError(400, "xfa_not_supported", XFA_UNSUPPORTED_USER_MESSAGE)
        return kind.value
    finally:
        tmp_path.unlink(missing_ok=True)


def run_full_job_pipeline(
    *,
    job_id: str,
    job_root: Path,
    input_pdf: Path,
    output_dir: Path,
    settings: Settings,
    source_filename: str,
    dpi: float = 200.0,
    allow_rotated_pages: bool = False,
) -> None:
    """Run convert, line detection, and semantic grounding; update ``job.json`` stages."""
    det_cfg = FormLineDetectorConfig().to_detector_dict()
    try:
        update_job_stage(job_root, "convert", status="running")
        conv = run_convert_sync(
            job_id=job_id,
            input_pdf=input_pdf,
            output_dir=output_dir,
            dpi=dpi,
            allow_rotated_pages=allow_rotated_pages,
            source_filename=source_filename,
        )
        update_job_after_convert(job_root, page_count=conv["page_count"], dpi=int(dpi))

        update_job_stage(job_root, "line_detect", status="running")
        run_detect_form_lines_for_job_output_dir(
            job_id=job_id,
            output_dir=output_dir,
            detector_config=det_cfg,
        )
        update_job_after_line_detect(job_root)

        run_semantic_grounding_for_job(
            job_id=job_id,
            output_dir=output_dir,
            settings=settings,
            provider=settings.grounding_provider,
            model=settings.grounding_model,
        )

        # E5: optionally auto-refine as the final stage (config toggle, off by default).
        # Never let a refinement failure sink an otherwise-ready job.
        if settings.grounding_qa_enabled:
            run_qa_refine_background(
                job_id=job_id,
                job_root=job_root,
                output_dir=output_dir,
                settings=settings,
            )

        # Dev/test: stamp with fixture values after final boxes exist. Never fail the job.
        if settings.dev_auto_stamp_after_grounding:
            run_dev_auto_stamp_background(
                job_id=job_id,
                job_root=job_root,
                input_pdf=input_pdf,
                output_dir=output_dir,
                settings=settings,
                source_filename=source_filename,
            )
    except SemanticGroundingJobError as exc:
        LOG.warning("job pipeline grounding failed job_id=%s: %s", job_id, exc)
        manifest = read_job_manifest(job_root)
        manifest["status"] = "failed"
        manifest["stages"]["grounding"]["status"] = "failed"
        manifest["stages"]["grounding"]["error"] = str(exc)
        write_job_manifest(job_root, manifest)
    except Exception as exc:
        LOG.exception("job pipeline failed job_id=%s", job_id)
        try:
            manifest = read_job_manifest(job_root)
            manifest["status"] = "failed"
            for stage in ("convert", "line_detect", "grounding"):
                node = manifest.get("stages", {}).get(stage, {})
                if node.get("status") not in ("done", "skipped"):
                    node["status"] = "failed"
                    node["error"] = str(exc)
            write_job_manifest(job_root, manifest)
        except FileNotFoundError:
            pass
        raise


def run_qa_refine_background(
    *,
    job_id: str,
    job_root: Path,
    output_dir: Path,
    settings: Settings,
) -> None:
    """Background wrapper for the E5 refinement loop; records failures on ``stages.qa_refine``."""
    from app.services.qa_refinement import QaRefinementError, run_qa_refinement_for_job

    try:
        run_qa_refinement_for_job(job_id=job_id, output_dir=output_dir, settings=settings)
    except (QaRefinementError, Exception) as exc:  # noqa: BLE001 - isolate the manual re-run
        LOG.warning("qa refinement failed job_id=%s: %s", job_id, exc)
        try:
            update_job_stage(job_root, "qa_refine", status="failed", error=str(exc))
        except FileNotFoundError:
            pass


def run_dev_auto_stamp_background(
    *,
    job_id: str,
    job_root: Path,
    input_pdf: Path,
    output_dir: Path,
    settings: Settings,
    source_filename: str,
) -> None:
    """Apply fixture values + stamp after grounding; isolate failures from job readiness."""
    from app.services.dev_auto_stamp import run_dev_auto_stamp_after_grounding

    try:
        run_dev_auto_stamp_after_grounding(
            job_id=job_id,
            job_root=job_root,
            input_pdf=input_pdf,
            output_dir=output_dir,
            settings=settings,
            source_filename=source_filename,
        )
    except Exception as exc:  # noqa: BLE001 - never fail a ready job for a dev stamp hook
        LOG.warning("dev auto-stamp failed job_id=%s: %s", job_id, exc)


def delete_job_tree(job_root: Path) -> None:
    if job_root.is_dir():
        shutil.rmtree(job_root, ignore_errors=True)
