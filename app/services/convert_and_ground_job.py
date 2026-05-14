"""Synchronous PDF conversion and field grounding for a job with ``input.pdf`` on disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.conversion import run_convert_pdf_to_images
from app.services.document_manifest import write_provider_metadata_to_document_manifest
from app.services.field_grounding import run_field_grounding_for_job
from app.services.jobs import read_document_manifest

LOG = logging.getLogger(__name__)


def assert_pdf_header(input_pdf: Path) -> None:
    with input_pdf.open("rb") as handle:
        header = handle.read(5)
    if header != b"%PDF-":
        raise ValueError("File does not look like a PDF (missing %PDF- header).")


def run_convert_sync(
    *,
    job_id: str,
    input_pdf: Path,
    output_dir: Path,
    dpi: float,
    allow_rotated_pages: bool,
    source_filename: str,
) -> dict[str, Any]:
    """
    Rasterize ``input_pdf`` into page PNGs and write manifests under ``output_dir``.

    Does not call vision APIs. Returns ``document_manifest`` as written by the converter
    (no ``provider`` / ``model`` yet; those are added by :func:`run_ground_sync`).
    """
    assert_pdf_header(input_pdf)

    try:
        convert_result = run_convert_pdf_to_images(
            str(input_pdf),
            str(output_dir),
            dpi,
            overwrite=True,
            allow_rotated_pages=allow_rotated_pages,
            job_id=job_id,
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    except FileExistsError:
        raise
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    doc = read_document_manifest(output_dir)
    page_count = len(convert_result.get("pages", []))

    LOG.info(
        "convert_sync job=%s pages=%d file=%s",
        job_id,
        page_count,
        source_filename,
    )

    return {
        "convert_result": convert_result,
        "document_manifest": doc,
        "page_count": page_count,
        "source_filename": source_filename,
        "dpi": dpi,
        "allow_rotated_pages": allow_rotated_pages,
    }


def run_ground_sync(
    *,
    job_id: str,
    output_dir: Path,
    provider: str,
    model: str,
    settings: Settings,
) -> dict[str, Any]:
    """
    Run vision field grounding on an already-converted job.

    Writes provider/model onto ``document_manifest.json`` then populates ``field_grounding/``.
    """
    doc = write_provider_metadata_to_document_manifest(
        output_dir=output_dir,
        provider=provider,
        model=model,
    )

    try:
        ground_result = run_field_grounding_for_job(
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
        raise ValueError(str(exc)) from exc

    if ground_result["succeeded_count"] == 0:
        raise RuntimeError(
            "Grounding failed for all pages.",
        )

    LOG.info(
        "ground_sync provider=%s model=%s job=%s pages=%d succeeded=%d",
        provider,
        model,
        job_id,
        ground_result.get("page_count", 0),
        ground_result.get("succeeded_count", 0),
    )

    return {
        "document_manifest": doc,
        "ground_result": ground_result,
        "provider": provider,
        "model": model,
    }


def run_convert_and_ground_sync(
    *,
    job_id: str,
    input_pdf: Path,
    output_dir: Path,
    dpi: float,
    allow_rotated_pages: bool,
    provider: str,
    model: str,
    settings: Settings,
    source_filename: str,
) -> dict[str, Any]:
    """
    Convert PDF pages to images and run vision grounding in one call.

    Returns a dict with ``convert_result``, ``document_manifest`` (with provider metadata),
    ``ground_result``, ``page_count``, ``source_filename``, ``dpi``, ``allow_rotated_pages``,
    ``provider``, ``model``.
    """
    convert_part = run_convert_sync(
        job_id=job_id,
        input_pdf=input_pdf,
        output_dir=output_dir,
        dpi=dpi,
        allow_rotated_pages=allow_rotated_pages,
        source_filename=source_filename,
    )
    ground_part = run_ground_sync(
        job_id=job_id,
        output_dir=output_dir,
        provider=provider,
        model=model,
        settings=settings,
    )

    LOG.info(
        "convert_and_ground_sync provider=%s model=%s job=%s pages=%d file=%s",
        provider,
        model,
        job_id,
        convert_part["page_count"],
        source_filename,
    )

    return {
        **convert_part,
        "document_manifest": ground_part["document_manifest"],
        "ground_result": ground_part["ground_result"],
        "provider": provider,
        "model": model,
    }
