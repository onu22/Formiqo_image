"""Per-job filesystem layout and job.json manifest helpers."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

_JOB_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.I,
)

DEFAULT_MAX_STAMP_RUNS = 3


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

    ``output_dir`` is where conversion writes manifests and PNGs.
    """
    root = job_root(jobs_dir, job_id)
    return root, root / "input.pdf", root / "output"


def job_manifest_path(job_root_dir: Path) -> Path:
    return job_root_dir / "job.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_manifest(
    *,
    job_id: str,
    source_filename: str,
    dpi: int = 200,
    detected_pdf_type: str = "flat",
) -> dict[str, Any]:
    """Build a new job manifest matching ``harness/specs/job-manifest-schema.md``."""
    now = _utc_now_iso()
    return {
        "job_id": job_id,
        "source_filename": source_filename,
        "created_at": now,
        "updated_at": now,
        "status": "converting",
        "page_count": 0,
        "dpi": dpi,
        "detected_pdf_type": detected_pdf_type,
        "stages": {
            "convert": {"status": "pending", "error": None, "failed_pages": []},
            "line_detect": {"status": "pending", "error": None, "failed_pages": []},
            "grounding": {
                "status": "pending",
                "error": None,
                "grounded_pages": 0,
                "total_pages": 0,
                "failed_pages": [],
            },
            "qa_refine": {"status": "skipped", "iterations": 0, "error": None},
        },
        "artifacts": {
            "input_pdf": "input.pdf",
            "converted_images_dir": "output/converted_images",
            "page_manifests_dir": "output/converted_images/pages",
            "fields_dir": "output/field_grounding",
            "stamping_json": "output/field_grounding/stamping.json",
            "latest_stamp_run_id": None,
            "latest_stamped_pdf": None,
            "latest_stamped_images_dir": None,
        },
        "grounding": {"provider": None, "model": None, "run_id": None},
        "retention": {"max_stamp_runs": DEFAULT_MAX_STAMP_RUNS},
    }


def read_job_manifest(job_root_dir: Path) -> dict[str, Any]:
    path = job_manifest_path(job_root_dir)
    if not path.is_file():
        raise FileNotFoundError(f"job.json missing under {job_root_dir}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"job.json must be a JSON object: {path}")
    return data


def write_job_manifest(job_root_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now_iso()
    path = job_manifest_path(job_root_dir)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def update_job_stage(
    job_root_dir: Path,
    stage: str,
    *,
    status: str | None = None,
    error: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    manifest = read_job_manifest(job_root_dir)
    stages = manifest.setdefault("stages", {})
    node = stages.setdefault(stage, {})
    if status is not None:
        node["status"] = status
    if error is not None:
        node["error"] = error
    node.update(extra)
    write_job_manifest(job_root_dir, manifest)
    return manifest


def set_job_status(job_root_dir: Path, status: str) -> dict[str, Any]:
    manifest = read_job_manifest(job_root_dir)
    manifest["status"] = status
    write_job_manifest(job_root_dir, manifest)
    return manifest


def set_job_grounding(
    job_root_dir: Path,
    *,
    provider: str,
    model: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    manifest = read_job_manifest(job_root_dir)
    manifest["grounding"] = {
        "provider": provider.strip().lower(),
        "model": model.strip(),
        "run_id": run_id,
    }
    write_job_manifest(job_root_dir, manifest)
    return manifest


def job_grounding_provider_model(job_root_dir: Path) -> tuple[str, str]:
    manifest = read_job_manifest(job_root_dir)
    grounding = manifest.get("grounding")
    if not isinstance(grounding, dict):
        raise ValueError("job.json missing grounding object.")
    prov = grounding.get("provider")
    model = grounding.get("model")
    if not isinstance(prov, str) or not prov.strip():
        raise ValueError("job.json missing non-empty grounding.provider.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("job.json missing non-empty grounding.model.")
    return prov.strip().lower(), model.strip()


def update_job_after_convert(
    job_root_dir: Path,
    *,
    page_count: int,
    dpi: int,
) -> dict[str, Any]:
    manifest = read_job_manifest(job_root_dir)
    manifest["page_count"] = page_count
    manifest["dpi"] = dpi
    manifest["stages"]["convert"]["status"] = "done"
    manifest["stages"]["convert"]["error"] = None
    write_job_manifest(job_root_dir, manifest)
    return manifest


def update_job_after_line_detect(job_root_dir: Path) -> dict[str, Any]:
    manifest = read_job_manifest(job_root_dir)
    manifest["stages"]["line_detect"]["status"] = "done"
    manifest["stages"]["line_detect"]["error"] = None
    manifest["status"] = "grounding"
    manifest["stages"]["grounding"]["status"] = "pending"
    manifest["stages"]["grounding"]["total_pages"] = manifest.get("page_count", 0)
    write_job_manifest(job_root_dir, manifest)
    return manifest


def update_job_after_grounding(
    job_root_dir: Path,
    *,
    provider: str,
    model: str,
    grounded_pages: int,
    total_pages: int,
    failed_pages: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = read_job_manifest(job_root_dir)
    manifest["grounding"] = {
        "provider": provider.strip().lower(),
        "model": model.strip(),
        "run_id": _utc_now_iso(),
    }
    manifest["stages"]["grounding"].update(
        {
            "status": "done",
            "error": None,
            "grounded_pages": grounded_pages,
            "total_pages": total_pages,
            "failed_pages": failed_pages,
        }
    )
    manifest["status"] = "ready"
    write_job_manifest(job_root_dir, manifest)
    return manifest


def update_job_after_image_stamp(
    job_root_dir: Path,
    *,
    stamp_run_id: str,
    run_dir_rel: str,
) -> dict[str, Any]:
    manifest = read_job_manifest(job_root_dir)
    artifacts = manifest.setdefault("artifacts", {})
    artifacts["latest_stamp_run_id"] = stamp_run_id
    artifacts["latest_stamped_images_dir"] = run_dir_rel
    write_job_manifest(job_root_dir, manifest)
    return manifest


def update_job_after_pdf_stamp(
    job_root_dir: Path,
    *,
    stamp_run_id: str,
    pdf_rel: str,
) -> dict[str, Any]:
    manifest = read_job_manifest(job_root_dir)
    artifacts = manifest.setdefault("artifacts", {})
    artifacts["latest_stamp_run_id"] = stamp_run_id
    artifacts["latest_stamped_pdf"] = pdf_rel
    manifest["status"] = "exported"
    write_job_manifest(job_root_dir, manifest)
    return manifest


def prune_stamp_runs(output_dir: Path, subdir: str, max_runs: int) -> None:
    """Keep only the newest ``max_runs`` directories under ``output_dir/subdir``."""
    base = output_dir / subdir
    if not base.is_dir() or max_runs < 1:
        return
    run_dirs = sorted(
        (p for p in base.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    for old in run_dirs[max_runs:]:
        shutil.rmtree(old, ignore_errors=True)


def page_manifest_image_px(page_manifest: dict[str, Any]) -> tuple[int, int]:
    image_node = page_manifest.get("image")
    if not isinstance(image_node, dict):
        raise ValueError("Page manifest missing image object.")
    try:
        return int(image_node["width_px"]), int(image_node["height_px"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Page manifest missing width_px/height_px.") from exc


def grounding_page_px(grounding: dict[str, Any]) -> tuple[int, int]:
    w = grounding.get("width_px", grounding.get("width"))
    h = grounding.get("height_px", grounding.get("height"))
    try:
        return int(w), int(h)
    except (TypeError, ValueError) as exc:
        raise ValueError("Grounding page missing width_px/height_px.") from exc


def list_job_summaries(jobs_dir: Path) -> list[dict[str, Any]]:
    """Scan ``jobs_dir`` for ``job.json`` files and return list summaries."""
    if not jobs_dir.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for child in sorted(jobs_dir.iterdir()):
        if not child.is_dir():
            continue
        try:
            manifest = read_job_manifest(child)
        except (FileNotFoundError, ValueError):
            continue
        summaries.append(
            {
                "job_id": manifest.get("job_id", child.name),
                "source_filename": manifest.get("source_filename", ""),
                "page_count": int(manifest.get("page_count") or 0),
                "status": manifest.get("status", "unknown"),
                "created_at": manifest.get("created_at", ""),
            }
        )
    summaries.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return summaries


def job_detail_projection(manifest: dict[str, Any]) -> dict[str, Any]:
    """Shape ``job.json`` for ``GET /jobs/{id}`` polling."""
    stages_out: dict[str, Any] = {}
    stages = manifest.get("stages") or {}
    grounding = stages.get("grounding")
    if isinstance(grounding, dict):
        stages_out["grounding"] = {
            "grounded_pages": int(grounding.get("grounded_pages") or 0),
            "total_pages": int(grounding.get("total_pages") or 0),
        }
        if grounding.get("status"):
            stages_out["grounding"]["status"] = grounding["status"]

    qa_refine = stages.get("qa_refine")
    if isinstance(qa_refine, dict):
        qa_out: dict[str, Any] = {"status": qa_refine.get("status", "skipped")}
        for key in ("iterations", "converged", "counts", "cost", "page_count"):
            if key in qa_refine:
                qa_out[key] = qa_refine[key]
        stages_out["qa_refine"] = qa_out

    errors: list[Any] = []
    for stage_name, node in stages.items():
        if not isinstance(node, dict):
            continue
        err = node.get("error")
        if isinstance(err, str) and err.strip():
            errors.append({"stage": stage_name, "message": err})
        failed_pages = node.get("failed_pages")
        if isinstance(failed_pages, list):
            for fp in failed_pages:
                if isinstance(fp, dict):
                    errors.append(fp)

    artifacts = manifest.get("artifacts") or {}
    return {
        "job_id": manifest.get("job_id"),
        "source_filename": manifest.get("source_filename"),
        "status": manifest.get("status"),
        "page_count": int(manifest.get("page_count") or 0),
        "stages": stages_out,
        "errors": errors,
        "artifacts": {
            "has_stamped_preview": bool(artifacts.get("latest_stamped_images_dir")),
            "has_export_pdf": bool(artifacts.get("latest_stamped_pdf")),
        },
    }


def to_output_relative_path(output_dir: Path, path: Path) -> str:
    """Return ``path`` relative to ``output_dir`` using forward slashes (portable JSON)."""
    out_root = output_dir.resolve()
    rel = path.resolve().relative_to(out_root)
    return rel.as_posix()


def resolve_under_output_dir(output_dir: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``output_dir``; reject paths that escape the output root."""
    out_root = output_dir.resolve()
    candidate = (out_root / rel).resolve()
    try:
        candidate.relative_to(out_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes job output directory: {rel!r}") from exc
    return candidate


def normalize_stored_path(output_dir: Path, stored: str) -> Path:
    """
    Resolve a path from JSON: output-relative (preferred) or legacy absolute if the file exists.
    """
    p = Path(stored)
    if p.is_absolute() and p.is_file():
        return p.resolve()
    return resolve_under_output_dir(output_dir, stored)
