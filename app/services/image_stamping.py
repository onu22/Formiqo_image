"""Stamp user-provided values onto converted PNG page images."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_GROUNDING_PAGE_RE = re.compile(r"^page_(\d{4})\.fields\.json$")
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class StampImageStyle:
    """Drawing options for image stamping."""

    font_size_px: int = 22
    font_color: str = "#111111"
    padding_px: int = 3
    draw_debug_boxes: bool = False
    debug_box_color: str = "#ff0000"


def _validate_hex_color(value: str, *, field_name: str) -> str:
    if not _HEX_COLOR_RE.match(value):
        raise ValueError(f"{field_name} must be a hex color like #111111.")
    return value


def _load_font(size_px: int) -> ImageFont.ImageFont:
    """Load a scalable font when available, otherwise fall back to Pillow default."""
    candidates = [
        "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size_px)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _fit_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    max_width_px: int,
    preferred_font_size_px: int,
    min_font_size_px: int = 8,
) -> tuple[str, ImageFont.ImageFont]:
    """Shrink text, then truncate with ellipsis if it still does not fit."""
    for size in range(preferred_font_size_px, min_font_size_px - 1, -1):
        font = _load_font(size)
        width, _ = _text_size(draw, text, font)
        if width <= max_width_px:
            return text, font

    font = _load_font(min_font_size_px)
    ellipsis = "..."
    if _text_size(draw, ellipsis, font)[0] > max_width_px:
        return "", font

    clipped = text
    while clipped:
        candidate = f"{clipped}{ellipsis}"
        if _text_size(draw, candidate, font)[0] <= max_width_px:
            return candidate, font
        clipped = clipped[:-1]
    return ellipsis, font


def _bbox_from_field(field: dict[str, Any], *, width_px: int, height_px: int, field_index: int) -> dict[str, int]:
    bbox = field.get("bbox")
    if not isinstance(bbox, dict):
        raise ValueError(f"fields[{field_index}].bbox must be an object.")
    expected = {"x", "y", "w", "h"}
    if set(bbox.keys()) != expected:
        raise ValueError(f"fields[{field_index}].bbox keys must be exactly {sorted(expected)}.")
    try:
        x = int(bbox["x"])
        y = int(bbox["y"])
        w = int(bbox["w"])
        h = int(bbox["h"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fields[{field_index}].bbox values must be integers.") from exc

    if w <= 0 or h <= 0:
        raise ValueError(f"fields[{field_index}].bbox requires w > 0 and h > 0.")
    if x < 0 or y < 0:
        raise ValueError(f"fields[{field_index}].bbox requires x >= 0 and y >= 0.")
    if x + w > width_px or y + h > height_px:
        raise ValueError(f"fields[{field_index}].bbox exceeds image bounds ({width_px}x{height_px}).")
    return {"x": x, "y": y, "w": w, "h": h}


def stamp_text_into_bbox(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: dict[str, int],
    text: str,
    style: StampImageStyle,
) -> bool:
    """Draw text inside a pixel bbox. Returns true when visible text was drawn."""
    if text == "":
        return False

    inner_w = max(1, bbox["w"] - (style.padding_px * 2))
    text_to_draw, font = _fit_text_to_width(
        draw,
        text,
        max_width_px=inner_w,
        preferred_font_size_px=style.font_size_px,
    )
    if not text_to_draw:
        return False

    _, text_h = _text_size(draw, text_to_draw, font)
    x = bbox["x"] + style.padding_px
    y = bbox["y"] + max(style.padding_px, (bbox["h"] - text_h) // 2)
    draw.text((x, y), text_to_draw, fill=style.font_color, font=font)
    return True


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _read_page_manifest(output_dir: Path, page_index: int) -> dict[str, Any]:
    rel = Path("converted_images") / "pages" / f"page_{page_index + 1:04d}.json"
    path = output_dir / rel
    if not path.is_file():
        raise FileNotFoundError(f"Conversion page manifest not found: {rel}")
    return _load_json(path)


def _image_path_from_manifest(output_dir: Path, page_manifest: dict[str, Any]) -> Path:
    image_node = page_manifest.get("image")
    if not isinstance(image_node, dict):
        raise ValueError("Conversion page manifest missing image object.")
    image_rel = image_node.get("path")
    if not isinstance(image_rel, str) or not image_rel:
        raise ValueError("Conversion page manifest image.path must be a non-empty string.")
    output_root = output_dir.resolve()
    image_path = (output_root / image_rel).resolve()
    try:
        image_path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"Source image path escapes job output directory: {image_rel}") from exc
    if not image_path.is_file():
        raise FileNotFoundError(f"Source image not found: {image_rel}")
    return image_path


def _validate_page_inputs(
    *,
    page_manifest: dict[str, Any],
    grounding: dict[str, Any],
    page_index: int,
) -> tuple[int, int, str]:
    image_node = page_manifest.get("image")
    if not isinstance(image_node, dict):
        raise ValueError("Conversion page manifest missing image object.")
    try:
        img_w = int(image_node["saved_image_width_px"])
        img_h = int(image_node["saved_image_height_px"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Conversion page manifest missing saved image dimensions.") from exc

    g_page_index = int(grounding.get("page_index", -1))
    g_width = int(grounding.get("width", -1))
    g_height = int(grounding.get("height", -1))
    g_unit = grounding.get("unit")
    g_origin = grounding.get("origin")

    if g_page_index != page_index:
        raise ValueError(f"Grounding page_index ({g_page_index}) does not match expected {page_index}.")
    if g_width != img_w or g_height != img_h:
        raise ValueError(
            f"Grounding dimensions ({g_width}x{g_height}) do not match converted image ({img_w}x{img_h})."
        )
    if g_unit != "px":
        raise ValueError('Grounding unit must be "px".')
    if g_origin != "top-left":
        raise ValueError('Grounding origin must be "top-left".')
    image_rel = str(image_node.get("path", ""))
    return img_w, img_h, image_rel


def stamp_page_image(
    *,
    output_dir: Path,
    page_index: int,
    grounding_path: Path,
    output_image_path: Path,
    values: dict[str, str],
    style: StampImageStyle,
    require_all_values: bool,
) -> dict[str, Any]:
    """Stamp one page image and return a per-page result."""
    page_manifest = _read_page_manifest(output_dir, page_index)
    grounding = _load_json(grounding_path)
    width_px, height_px, image_rel = _validate_page_inputs(
        page_manifest=page_manifest,
        grounding=grounding,
        page_index=page_index,
    )
    source_image_path = _image_path_from_manifest(output_dir, page_manifest)

    fields = grounding.get("fields")
    if not isinstance(fields, list):
        raise ValueError("Grounding fields must be a list.")

    missing_values: list[str] = []
    warnings: list[str] = []
    stamped_count = 0
    unsupported_count = 0

    with Image.open(source_image_path) as source:
        image = source.convert("RGBA")
    if image.size != (width_px, height_px):
        raise ValueError(f"PNG dimensions {image.size} do not match manifest dimensions ({width_px}, {height_px}).")

    draw = ImageDraw.Draw(image)
    if style.draw_debug_boxes:
        _validate_hex_color(style.debug_box_color, field_name="debug_box_color")
    _validate_hex_color(style.font_color, field_name="font_color")

    for idx, field in enumerate(fields):
        if not isinstance(field, dict):
            raise ValueError(f"fields[{idx}] must be an object.")
        field_id = field.get("field_id")
        field_type = field.get("type")
        if not isinstance(field_id, str) or not field_id.strip():
            raise ValueError(f"fields[{idx}].field_id must be a non-empty string.")
        bbox = _bbox_from_field(field, width_px=width_px, height_px=height_px, field_index=idx)

        if style.draw_debug_boxes:
            draw.rectangle(
                (bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]),
                outline=style.debug_box_color,
                width=2,
            )

        if field_type != "text":
            unsupported_count += 1
            warnings.append(f"Skipped unsupported field type for {field_id}: {field_type}")
            continue

        if field_id not in values:
            missing_values.append(field_id)
            continue

        if stamp_text_into_bbox(draw, bbox=bbox, text=values[field_id], style=style):
            stamped_count += 1

    if require_all_values and missing_values:
        raise ValueError(f"Missing values for field_id(s): {', '.join(sorted(missing_values))}")

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_image_path, format="PNG")

    return {
        "page_index": page_index,
        "status": "succeeded",
        "source_image": image_rel,
        "grounding_file": str(grounding_path.relative_to(output_dir)).replace("\\", "/"),
        "output_image": str(output_image_path.relative_to(output_dir)).replace("\\", "/"),
        "field_count": len(fields),
        "stamped_count": stamped_count,
        "missing_value_count": len(missing_values),
        "unsupported_field_count": unsupported_count,
        "warnings": warnings,
    }


def _discover_grounding_pages(grounding_dir: Path) -> list[tuple[int, Path]]:
    pages: list[tuple[int, Path]] = []
    for path in sorted(grounding_dir.glob("page_*.fields.json")):
        match = _GROUNDING_PAGE_RE.match(path.name)
        if not match:
            continue
        pages.append((int(match.group(1)) - 1, path))
    if not pages:
        raise FileNotFoundError(f"No grounded page files found under: {grounding_dir}")
    return pages


def _assert_grounding_run_matches(grounding_dir: Path, *, provider: str, model: str) -> None:
    manifest_path = grounding_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = _load_json(manifest_path)
    if manifest.get("provider") != provider or manifest.get("model") != model:
        raise FileNotFoundError(
            f"Field grounding run not found for provider={provider}, model={model} "
            f"(found provider={manifest.get('provider')}, model={manifest.get('model')})."
        )


def run_image_stamping_for_job(
    *,
    job_id: str,
    output_dir: Path,
    provider: str,
    model: str,
    values: dict[str, str],
    style: StampImageStyle,
    require_all_values: bool,
) -> dict[str, Any]:
    """Stamp all grounded converted images for a job."""
    provider_norm = provider.strip().lower()
    if not provider_norm:
        raise ValueError("provider must be a non-empty string.")
    if not model.strip():
        raise ValueError("model must be a non-empty string.")

    converted_pages_dir = output_dir / "converted_images" / "pages"
    if not converted_pages_dir.is_dir():
        raise FileNotFoundError(f"Conversion page manifests not found: converted_images/pages")

    grounding_dir = output_dir / "field_grounding"
    if not grounding_dir.is_dir():
        raise FileNotFoundError(
            f"Field grounding run not found for provider={provider_norm}, model={model} "
            f"(expected directory: field_grounding)."
        )
    _assert_grounding_run_matches(grounding_dir, provider=provider_norm, model=model)

    stamp_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir_rel = f"stamped_images/{stamp_run_id}"
    run_dir = output_dir / run_dir_rel
    grounding_pages = _discover_grounding_pages(grounding_dir)

    page_results: list[dict[str, Any]] = []
    output_files: list[str] = []

    for page_index, grounding_path in grounding_pages:
        try:
            output_image_path = run_dir / f"page_{page_index + 1:04d}.{provider_norm}.stamped.png"
            page_result = stamp_page_image(
                output_dir=output_dir,
                page_index=page_index,
                grounding_path=grounding_path,
                output_image_path=output_image_path,
                values=values,
                style=style,
                require_all_values=require_all_values,
            )
            page_results.append(page_result)
            output_files.append(page_result["output_image"])
        except Exception as exc:  # noqa: BLE001 - preserve per-page errors in API response
            page_results.append(
                {
                    "page_index": page_index,
                    "status": "failed",
                    "grounding_file": str(grounding_path.relative_to(output_dir)).replace("\\", "/"),
                    "error": str(exc),
                }
            )

    succeeded_count = sum(1 for page in page_results if page["status"] == "succeeded")
    failed_count = len(page_results) - succeeded_count
    manifest_rel = f"{run_dir_rel}/manifest.json"
    manifest = {
        "job_id": job_id,
        "provider": provider_norm,
        "model": model,
        "stamp_run_id": stamp_run_id,
        "run_dir": run_dir_rel,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(page_results),
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "files": output_files,
        "pages": page_results,
        "style": {
            "font_size_px": style.font_size_px,
            "font_color": style.font_color,
            "padding_px": style.padding_px,
            "draw_debug_boxes": style.draw_debug_boxes,
            "debug_box_color": style.debug_box_color,
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "job_id": job_id,
        "provider": provider_norm,
        "model": model,
        "stamp_run_id": stamp_run_id,
        "run_dir": run_dir_rel,
        "manifest_path": manifest_rel,
        "page_count": len(page_results),
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "pages": page_results,
    }
