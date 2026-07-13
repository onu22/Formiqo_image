"""Load and persist grounded fields + stamping.json for the E3 editor API."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.schemas import StampingJson, StampStyleSchema
from app.services.jobs import grounding_page_px, page_manifest_image_px, resolve_under_output_dir

_PAGE_FIELDS_RE = re.compile(r"^page_(\d{4})\.fields\.json$")


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _page_number_from_filename(name: str) -> int:
    match = _PAGE_FIELDS_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected fields file name: {name}")
    return int(match.group(1))


def _style_projection(stamping: StampingJson) -> dict[str, Any]:
    """``style`` reads directly from ``stamping.json`` (PDF points) after E2."""
    return {"font_size_pt": stamping.style.font_size_pt, "text_color": stamping.style.text_color}


def load_fields_payload(*, job_id: str, output_dir: Path) -> dict[str, Any]:
    fg_dir = output_dir / "field_grounding"
    if not fg_dir.is_dir():
        raise FileNotFoundError("field_grounding directory not found")

    stamping_path = fg_dir / "stamping.json"
    if not stamping_path.is_file():
        raise FileNotFoundError("field_grounding/stamping.json not found")
    stamping = StampingJson.model_validate(_load_json(stamping_path))

    pages: list[dict[str, Any]] = []
    for path in sorted(fg_dir.glob("page_*.fields.json")):
        page_number = _page_number_from_filename(path.name)
        payload = _load_json(path)
        page_manifest_path = output_dir / "converted_images" / "pages" / f"page_{page_number:04d}.json"
        if page_manifest_path.is_file():
            manifest = _load_json(page_manifest_path)
            width_px, height_px = page_manifest_image_px(manifest)
        else:
            width_px, height_px = grounding_page_px(payload)

        fields = payload.get("fields")
        if not isinstance(fields, list):
            fields = []

        pages.append(
            {
                "page_number": page_number,
                "width_px": width_px,
                "height_px": height_px,
                "fields": fields,
            }
        )

    return {
        "job_id": job_id,
        "style": _style_projection(stamping),
        "pages": pages,
        "values": dict(stamping.values),
    }


def patch_fields(
    *,
    output_dir: Path,
    field_updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for item in field_updates:
        field_id = item.get("field_id")
        page_number = item.get("page_number")
        if not isinstance(field_id, str) or not field_id.strip():
            raise ValueError("Each field update requires a non-empty field_id.")
        if not isinstance(page_number, int) or page_number < 1:
            raise ValueError("Each field update requires page_number >= 1.")

        path = output_dir / "field_grounding" / f"page_{page_number:04d}.fields.json"
        if not path.is_file():
            raise FileNotFoundError(f"Fields file not found for page {page_number}")

        payload = _load_json(path)
        fields = payload.get("fields")
        if not isinstance(fields, list):
            raise ValueError(f"Invalid fields list in {path.name}")

        matched = False
        for field in fields:
            if not isinstance(field, dict):
                continue
            if field.get("field_id") != field_id:
                continue
            matched = True
            if "bbox" in item:
                bbox = item["bbox"]
                if not isinstance(bbox, dict):
                    raise ValueError("bbox must be an object.")
                field["bbox"] = bbox
            if "font_size_pt" in item:
                field["font_size_pt"] = item["font_size_pt"]
            if "reviewed" in item:
                field["reviewed"] = bool(item["reviewed"])
            updated.append(
                {
                    "field_id": field_id,
                    "page_number": page_number,
                    "bbox": field.get("bbox"),
                    "font_size_pt": field.get("font_size_pt"),
                    "reviewed": field.get("reviewed"),
                }
            )
            break

        if not matched:
            raise ValueError(f"field_id {field_id!r} not found on page {page_number}")

        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return updated


def patch_values(
    *,
    output_dir: Path,
    values: dict[str, str] | None = None,
    style: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = output_dir / "field_grounding" / "stamping.json"
    if not path.is_file():
        raise FileNotFoundError("field_grounding/stamping.json not found")

    raw = _load_json(path)
    stamping = StampingJson.model_validate(raw)

    if values:
        merged = dict(stamping.values)
        merged.update(values)
        stamping.values = merged

    if style is not None:
        current = stamping.style.model_dump()
        current.update({k: v for k, v in style.items() if k in current})
        stamping.style = StampStyleSchema.model_validate(current)

    path.write_text(json.dumps(stamping.model_dump(), indent=2) + "\n", encoding="utf-8")
    return {"values": stamping.values, "style": _style_projection(stamping)}


def resolve_page_image_path(
    *,
    output_dir: Path,
    job_root: Path,
    page_number: int,
    variant: str,
) -> Path:
    if page_number < 1:
        raise ValueError("page_number must be >= 1")

    if variant == "source":
        rel = f"converted_images/page_{page_number:04d}.png"
        return resolve_under_output_dir(output_dir, rel)

    if variant != "stamped":
        raise ValueError('variant must be "source" or "stamped"')

    from app.services.jobs import read_job_manifest

    manifest = read_job_manifest(job_root)
    artifacts = manifest.get("artifacts") or {}
    stamped_dir_rel = artifacts.get("latest_stamped_images_dir")
    if not isinstance(stamped_dir_rel, str) or not stamped_dir_rel:
        raise FileNotFoundError("No stamped preview available for this job.")

    grounding = manifest.get("grounding") or {}
    provider = grounding.get("provider") or "openai"
    if not isinstance(provider, str):
        provider = "openai"

    rel = f"{stamped_dir_rel}/page_{page_number:04d}.{provider.strip().lower()}.stamped.png"
    return resolve_under_output_dir(output_dir, rel)


def resolve_export_pdf_path(*, output_dir: Path, job_root: Path) -> Path:
    from app.services.jobs import read_job_manifest

    manifest = read_job_manifest(job_root)
    artifacts = manifest.get("artifacts") or {}
    pdf_rel = artifacts.get("latest_stamped_pdf")
    if not isinstance(pdf_rel, str) or not pdf_rel:
        raise FileNotFoundError("No exported PDF available for this job.")
    return resolve_under_output_dir(output_dir, pdf_rel)
