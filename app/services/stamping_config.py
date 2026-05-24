"""Load field grounding manifest and stamping.json for job-only stamp/refine APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.grounding_field_types import TOGGLE_GROUNDING_TYPES
from app.schemas import StampImagesStyle as StampImagesStyleSchema
from app.schemas import StampingJson
from app.services.image_stamping import StampImageStyle


def build_stamping_json_sample(field_grounding_dir: Path) -> dict[str, Any]:
    """
    Build sample stamping values from ``page_*.fields.json`` under *field_grounding_dir*.

    Checkbox/radio fields get ``"true"``; other fields get a short placeholder from ``field_id``.
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
            ftype = field.get("type")
            if isinstance(ftype, str) and ftype in TOGGLE_GROUNDING_TYPES:
                values[field_id] = "true"
            else:
                values[field_id] = field_id[:10]
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


def load_field_grounding_manifest(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "field_grounding" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"field_grounding/manifest.json not found under {output_dir}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"manifest.json must be a JSON object: {path}")
    return data


def load_stamping_json_parsed(output_dir: Path) -> StampingJson:
    path = output_dir / "field_grounding" / "stamping.json"
    if not path.is_file():
        raise FileNotFoundError(f"field_grounding/stamping.json not found under {output_dir}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"stamping.json must be a JSON object: {path}")
    return StampingJson.model_validate(raw)


def manifest_provider_model(manifest: dict[str, Any]) -> tuple[str, str]:
    prov = manifest.get("provider")
    model = manifest.get("model")
    if not isinstance(prov, str) or not prov.strip():
        raise ValueError("manifest.json missing non-empty provider.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("manifest.json missing non-empty model.")
    return prov.strip().lower(), model.strip()


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
