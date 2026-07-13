"""Load field grounding and stamping.json for job-only stamp/refine APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas import StampImagesStyle as StampImagesStyleSchema
from app.schemas import StampingJson
from app.services.image_stamping import StampImageStyle
from app.services.jobs import job_grounding_provider_model


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
        "image_style": StampImagesStyleSchema().model_dump(),
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


def stamping_json_to_image_style(stamping: StampingJson) -> StampImageStyle:
    sch = stamping.image_style
    if sch is None:
        sch = StampImagesStyleSchema()
    return StampImageStyle(
        font_size_px=sch.font_size_px,
        font_color=sch.font_color,
        padding_px=sch.padding_px,
        draw_debug_boxes=sch.draw_debug_boxes,
        debug_box_color=sch.debug_box_color,
    )
