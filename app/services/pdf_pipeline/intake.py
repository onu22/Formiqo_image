"""Create a job from a PDF on disk and run the pipeline router."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.convert_and_ground_job import run_convert_sync
from app.services.jobs import job_paths
from app.services.line_detection_job import run_detect_form_lines_for_job_output_dir
from app.services.pdf_pipeline.detector import PdfTypeDetector
from app.services.pdf_pipeline.errors import PdfIntakeArchiveError, PdfPipelineError, XFA_UNSUPPORTED_USER_MESSAGE
from app.services.pdf_pipeline.router import PdfPipelineRouter
from app.services.pdf_pipeline.types import PdfPipelineKind

LOG = logging.getLogger(__name__)


def process_pdf_from_path(
    *,
    source_pdf: Path,
    settings: Settings,
    dpi: float = 200.0,
    allow_rotated_pages: bool = False,
) -> dict[str, Any]:
    """
    Allocate a new job, copy ``source_pdf`` to ``input.pdf``, detect kind, run pipeline.

    The source file on disk is unchanged — callers move/delete it after success or failure.
    """
    src = source_pdf.resolve()
    if not src.is_file():
        raise FileNotFoundError(str(src))

    job_id = str(uuid.uuid4())
    root, input_pdf, output_dir = job_paths(settings.jobs_dir, job_id)
    router = PdfPipelineRouter()

    try:
        root.mkdir(parents=True, exist_ok=True)
        kind = PdfTypeDetector.detect(src)
        LOG.info(
            "pdf_intake detected_pdf_type=%s job_id=%s source=%s",
            kind.value,
            job_id,
            src.name,
        )
        shutil.copy2(src, input_pdf)
        result = router.run(
            kind=kind,
            job_id=job_id,
            input_pdf=input_pdf,
            output_dir=output_dir,
            settings=settings,
            dpi=dpi,
            allow_rotated_pages=allow_rotated_pages,
        )
        result["detected_pdf_type"] = kind.value
        result["job_id"] = job_id
        return result
    except Exception as exc:
        shutil.rmtree(root, ignore_errors=True)
        raise PdfIntakeArchiveError(job_id=job_id) from exc


def iter_pending_upload_pdfs(user_uploads_dir: Path) -> list[Path]:
    """PDF files directly under ``user_uploads_dir`` (not ``processed`` / ``failed`` subfolders)."""
    base = user_uploads_dir.resolve()
    if not base.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(base.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".pdf":
            continue
        if p.name.startswith("."):
            continue
        out.append(p)
    return out


def archive_upload_dest(dest_dir: Path, original: Path, job_id: str) -> Path:
    """Archive path ``{original.stem}_{job_id}{suffix}``, with ``_2``, ``_3``, … on collision."""
    stem = original.stem
    suf = original.suffix
    primary = dest_dir / f"{stem}_{job_id}{suf}"
    if not primary.exists():
        return primary
    for n in range(2, 10_000):
        candidate = dest_dir / f"{stem}_{job_id}_{n}{suf}"
        if not candidate.exists():
            return candidate
    return dest_dir / f"{stem}_{job_id}_dup{uuid.uuid4().hex[:4]}{suf}"


def process_pdf_convert_and_line_detect_from_path(
    *,
    source_pdf: Path,
    settings: Settings,
    dpi: float = 200.0,
    allow_rotated_pages: bool = False,
    detector_config: dict[str, int],
    include_lines: bool = False,
) -> dict[str, Any]:
    """
    Allocate a job, copy ``source_pdf`` to ``input.pdf``, rasterize, run OpenCV line detection.

    No LLM or vision API calls. Raises :class:`PdfPipelineError` for XFA PDFs.
    On failure after the job directory is created, the job folder is removed and
    :class:`PdfIntakeArchiveError` is raised for unexpected errors.
    """
    src = source_pdf.resolve()
    if not src.is_file():
        raise FileNotFoundError(str(src))

    job_id = str(uuid.uuid4())
    root, input_pdf, output_dir = job_paths(settings.jobs_dir, job_id)

    try:
        root.mkdir(parents=True, exist_ok=True)
        kind = PdfTypeDetector.detect(src)
        LOG.info(
            "pdf_intake_convert_line_detect detected_pdf_type=%s job_id=%s source=%s",
            kind.value,
            job_id,
            src.name,
        )
        if kind == PdfPipelineKind.XFA:
            raise PdfPipelineError(XFA_UNSUPPORTED_USER_MESSAGE)
        shutil.copy2(src, input_pdf)
        conv = run_convert_sync(
            job_id=job_id,
            input_pdf=input_pdf,
            output_dir=output_dir,
            dpi=dpi,
            allow_rotated_pages=allow_rotated_pages,
            source_filename=src.name,
        )
        line = run_detect_form_lines_for_job_output_dir(
            job_id=job_id,
            output_dir=output_dir,
            detector_config=detector_config,
        )
        summary_pages = [
            {
                "page_index": p.page_index,
                "counts": p.detection.counts.model_dump(),
                "output_files": p.detection.output_files,
            }
            for p in line.pages
        ]
        outcome: dict[str, Any] = {
            "job_id": job_id,
            "detected_pdf_type": kind.value,
            "pipeline": "convert_line_detection",
            "page_count": conv["page_count"],
            "dpi": dpi,
            "allow_rotated_pages": allow_rotated_pages,
            "line_detection_summary": {
                "job_id": job_id,
                "page_count": line.page_count,
                "pages": summary_pages,
            },
        }
        if include_lines:
            outcome["line_detection_full"] = line.model_dump()
        return outcome
    except Exception as exc:
        shutil.rmtree(root, ignore_errors=True)
        if isinstance(exc, PdfPipelineError):
            raise
        raise PdfIntakeArchiveError(job_id=job_id) from exc


def scan_convert_and_detect_lines_user_uploads(
    *,
    settings: Settings,
    dpi: float = 200.0,
    allow_rotated_pages: bool = False,
    detector_config: dict[str, int],
    include_lines: bool = False,
) -> list[dict[str, Any]]:
    """
    For each inbox PDF: rasterize (``run_convert_sync``) and OpenCV line detection only.

    Successful uploads move to ``processed/``; failures (including XFA) to ``failed/``.
    """
    pending = iter_pending_upload_pdfs(settings.user_uploads_dir)
    processed_dir = settings.user_uploads_dir / "processed"
    failed_dir = settings.user_uploads_dir / "failed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for pdf_path in pending:
        item: dict[str, Any] = {
            "source": str(pdf_path),
            "ok": False,
            "job_id": None,
            "detected_pdf_type": None,
            "pipeline": None,
            "page_count": None,
            "dpi": None,
            "allow_rotated_pages": None,
            "line_detection_summary": None,
            "line_detection_full": None,
            "error": None,
            "detail": None,
        }
        try:
            outcome = process_pdf_convert_and_line_detect_from_path(
                source_pdf=pdf_path,
                settings=settings,
                dpi=dpi,
                allow_rotated_pages=allow_rotated_pages,
                detector_config=detector_config,
                include_lines=include_lines,
            )
            item.update(
                ok=True,
                job_id=outcome.get("job_id"),
                detected_pdf_type=outcome.get("detected_pdf_type"),
                pipeline=outcome.get("pipeline"),
                page_count=outcome.get("page_count"),
                dpi=outcome.get("dpi"),
                allow_rotated_pages=outcome.get("allow_rotated_pages"),
                line_detection_summary=outcome.get("line_detection_summary"),
                line_detection_full=outcome.get("line_detection_full"),
                detail={
                    "page_count": outcome.get("page_count"),
                    "dpi": outcome.get("dpi"),
                    "allow_rotated_pages": outcome.get("allow_rotated_pages"),
                },
            )
            dest = archive_upload_dest(processed_dir, pdf_path, outcome["job_id"])
            shutil.move(str(pdf_path), str(dest))
            LOG.info(
                "user_uploads convert+line_detect ok job_id=%s archived=%s",
                outcome.get("job_id"),
                dest.name,
            )
        except PdfIntakeArchiveError as exc:
            item["job_id"] = exc.job_id
            item["error"] = str(exc.__cause__) if exc.__cause__ is not None else str(exc)
            item["detail"] = {"message": item["error"], "job_id": exc.job_id}
            if pdf_path.is_file():
                shutil.move(
                    str(pdf_path),
                    str(archive_upload_dest(failed_dir, pdf_path, exc.job_id)),
                )
            LOG.warning(
                "user_uploads convert+line_detect failed source=%s job_id=%s error=%s",
                pdf_path.name,
                exc.job_id,
                item["error"],
            )
        except PdfPipelineError as exc:
            item["error"] = str(exc)
            item["detail"] = {"message": str(exc), "type": "PdfPipelineError"}
            if pdf_path.is_file():
                fallback_job_id = str(uuid.uuid4())
                shutil.move(
                    str(pdf_path),
                    str(archive_upload_dest(failed_dir, pdf_path, fallback_job_id)),
                )
            LOG.warning("user_uploads convert+line_detect rejected source=%s (%s)", pdf_path.name, item["error"])
        except Exception as exc:
            item["error"] = type(exc).__name__
            item["detail"] = {"message": str(exc), "type": type(exc).__name__}
            if pdf_path.is_file():
                try:
                    fallback_job_id = str(uuid.uuid4())
                    shutil.move(
                        str(pdf_path),
                        str(archive_upload_dest(failed_dir, pdf_path, fallback_job_id)),
                    )
                except OSError:
                    LOG.warning("could not move failed upload: %s", pdf_path)
            LOG.exception("user_uploads convert+line_detect unexpected failure source=%s", pdf_path.name)

        results.append(item)

    return results


def scan_and_process_user_uploads(
    *,
    settings: Settings,
    dpi: float = 200.0,
    allow_rotated_pages: bool = False,
) -> list[dict[str, Any]]:
    """
    Process each ``*.pdf`` in ``settings.user_uploads_dir``.

    Successful uploads are moved to ``user-uploads/processed``; failures to ``user-uploads/failed``.
    Archived PDFs use ``{filename_stem}_{job_id}.pdf`` under each folder; rare name clashes append ``_2``, ``_3``, … before the extension.
    """
    pending = iter_pending_upload_pdfs(settings.user_uploads_dir)
    processed_dir = settings.user_uploads_dir / "processed"
    failed_dir = settings.user_uploads_dir / "failed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for pdf_path in pending:
        item: dict[str, Any] = {
            "source": str(pdf_path),
            "ok": False,
            "job_id": None,
            "detected_pdf_type": None,
            "pipeline": None,
            "error": None,
            "detail": None,
        }
        try:
            outcome = process_pdf_from_path(
                source_pdf=pdf_path,
                settings=settings,
                dpi=dpi,
                allow_rotated_pages=allow_rotated_pages,
            )
            item.update(
                ok=True,
                job_id=outcome.get("job_id"),
                detected_pdf_type=outcome.get("detected_pdf_type"),
                pipeline=outcome.get("pipeline"),
                detail=outcome,
            )
            dest = archive_upload_dest(processed_dir, pdf_path, outcome["job_id"])
            shutil.move(str(pdf_path), str(dest))
            LOG.info(
                "user_uploads processed ok job_id=%s pipeline=%s archived=%s",
                outcome.get("job_id"),
                outcome.get("pipeline"),
                dest.name,
            )
        except PdfIntakeArchiveError as exc:
            item["job_id"] = exc.job_id
            item["error"] = str(exc.__cause__) if exc.__cause__ is not None else str(exc)
            item["detail"] = {"message": item["error"], "job_id": exc.job_id}
            if pdf_path.is_file():
                shutil.move(
                    str(pdf_path),
                    str(archive_upload_dest(failed_dir, pdf_path, exc.job_id)),
                )
            LOG.warning(
                "user_uploads failed source=%s job_id=%s error=%s",
                pdf_path.name,
                exc.job_id,
                item["error"],
            )
        except Exception as exc:
            item["error"] = type(exc).__name__
            item["detail"] = {"message": str(exc), "type": type(exc).__name__}
            if pdf_path.is_file():
                try:
                    fallback_job_id = str(uuid.uuid4())
                    shutil.move(
                        str(pdf_path),
                        str(archive_upload_dest(failed_dir, pdf_path, fallback_job_id)),
                    )
                except OSError:
                    LOG.warning("could not move failed upload: %s", pdf_path)
            LOG.exception("user_uploads unexpected failure source=%s", pdf_path.name)

        results.append(item)

    return results
