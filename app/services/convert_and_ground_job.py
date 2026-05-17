"""Synchronous PDF conversion for a job with ``input.pdf`` on disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.services.conversion import run_convert_pdf_to_images
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

    Does not call vision APIs.
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
            source_pdf_record="../input.pdf",
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
