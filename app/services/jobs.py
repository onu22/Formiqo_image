"""Per-job filesystem layout helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import UUID

_JOB_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)


def assert_valid_job_id(job_id: str) -> UUID:
    """Return a :class:`UUID` or raise ``ValueError`` if ``job_id`` is not a strict UUID string."""
    if not _JOB_ID_RE.match(job_id):
        raise ValueError("job_id must be a canonical UUID string")
    return UUID(job_id)


def job_root(jobs_dir: Path, job_id: str) -> Path:
    """Resolved job directory ``jobs_dir / job_id`` (validates UUID)."""
    jid = assert_valid_job_id(job_id)
    return (jobs_dir / str(jid)).resolve()


def job_paths(jobs_dir: Path, job_id: str) -> tuple[Path, Path, Path]:
    """
    Return ``(root, input_pdf, output_dir)`` for a job.

    ``output_dir`` is where :func:`convert_pdf_to_images` writes manifests and PNGs.
    """
    root = job_root(jobs_dir, job_id)
    return root, root / "input.pdf", root / "output"


def read_document_manifest(output_dir: Path) -> dict:
    """Load ``document_manifest.json`` from a completed job."""
    p = output_dir / "document_manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"document_manifest.json missing under {output_dir}")
    return json.loads(p.read_text(encoding="utf-8"))
