"""Load field grounding manifest and stamping.json for job-only stamp/refine APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas import StampImagesStyle as StampImagesStyleSchema
from app.schemas import StampingJson
from app.services.image_stamping import StampImageStyle


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
