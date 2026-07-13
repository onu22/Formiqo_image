"""Load field grounding and stamping.json for job-only stamp/refine APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas import StampingJson, StampStyleSchema
from app.services.jobs import job_grounding_provider_model
from app.services.stamping_common import StampStyle


def build_stamping_json_sample(field_grounding_dir: Path) -> dict[str, Any]:
    """
    Build sample stamping values from ``page_*.fields.json`` under *field_grounding_dir*.

    All fields default to empty strings.
    """
    values: dict[str, str] = {}
    for page_path in sorted(field_grounding_dir.glob("page_*.fields.json")):
        try:
            payload = json.loads(page_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {page_path}: {exc}") from exc
        if not isinstance(payload, dict):
            continue
        fields = payload.get("fields")
        if not isinstance(fields, list):
            continue
        for field in fields:
            if not isinstance(field, dict):
                continue
            field_id = field.get("field_id")
            if not isinstance(field_id, str) or not field_id.strip() or field_id in values:
                continue
            values[field_id] = ""
    return {
        "values": values,
        "require_all_values": False,
        "style": StampStyleSchema().model_dump(),
        "overrides": {},
    }


def write_stamping_json_sample(field_grounding_dir: Path) -> Path:
    """Overwrite ``stamping.json`` with sample values derived from grounded fields."""
    field_grounding_dir.mkdir(parents=True, exist_ok=True)
    path = field_grounding_dir / "stamping.json"
    payload = build_stamping_json_sample(field_grounding_dir)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_job_grounding_info(job_root_dir: Path) -> tuple[str, str]:
    """Return ``(provider, model)`` from ``job.json``."""
    return job_grounding_provider_model(job_root_dir)


def load_stamping_json_parsed(output_dir: Path) -> StampingJson:
    path = output_dir / "field_grounding" / "stamping.json"
    if not path.is_file():
        raise FileNotFoundError("field_grounding/stamping.json not found")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"stamping.json must be a JSON object: {path}")
    return StampingJson.model_validate(raw)


def stamping_style(stamping: StampingJson) -> StampStyle:
    """Convert the persisted (PDF-point) style schema into the internal render style."""
    return StampStyle(
        font_size_pt=stamping.style.font_size_pt,
        text_color=stamping.style.text_color,
    )


def stamping_overrides(stamping: StampingJson) -> dict[str, Any]:
    """Flatten ``stamping.json.overrides`` into plain dicts for the stampers."""
    return {field_id: override.model_dump() for field_id, override in stamping.overrides.items()}
