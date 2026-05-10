"""Synchronous convert + field grounding for a job with ``input.pdf`` already on disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.conversion import run_convert_pdf_to_images
from app.services.document_manifest import write_provider_metadata_to_document_manifest
from app.services.field_grounding import run_field_grounding_for_job

LOG = logging.getLogger(__name__)


def assert_pdf_header(input_pdf: Path) -> None:
    with input_pdf.open("rb") as handle:
        header = handle.read(5)
    if header != b"%PDF-":
        raise ValueError("File does not look like a PDF (missing %PDF- header).")


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
    Convert PDF pages to images and run vision grounding.

    Returns a dict with ``convert_result``, ``document_manifest``, ``ground_result``,
    ``page_count``, ``source_filename``.
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
    except FileExistsError as exc:
        raise RuntimeError(str(exc)) from exc
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    doc = write_provider_metadata_to_document_manifest(
        output_dir=output_dir,
        provider=provider,
        model=model,
    )
    page_count = len(convert_result.get("pages", []))

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
            "Convert succeeded but grounding failed for all pages.",
        )

    LOG.info(
        "convert_and_ground_sync provider=%s model=%s job=%s pages=%d file=%s",
        provider,
        model,
        job_id,
        page_count,
        source_filename,
    )

    return {
        "convert_result": convert_result,
        "document_manifest": doc,
        "ground_result": ground_result,
        "page_count": page_count,
        "source_filename": source_filename,
        "dpi": dpi,
        "allow_rotated_pages": allow_rotated_pages,
        "provider": provider,
        "model": model,
    }
