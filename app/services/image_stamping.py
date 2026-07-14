"""Render stamped values onto converted PNG page images (preview path, PRD E2)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from app.grounding_field_types import (
    is_supported_grounding_field_type,
    is_toggle_value_truthy,
    stamps_as_text,
    stamps_as_toggle,
)
from app.services.stamping_common import (
    PageGeometry,
    StampStyle,
    bbox_from_field,
    bottom_padding_pt,
    build_page_geometry,
    discover_grounding_pages,
    fit_multiline_to_box_pt,
    fit_text_to_width_pt,
    image_path_from_manifest,
    load_json,
    load_pil_font,
    prepare_stamp_run,
    read_page_manifest,
    rel_to_output,
    resolve_field_font_size_pt,
    run_stamp_pages,
    summarize_page_results,
    validate_hex_color,
    write_run_manifest,
    finalize_job_after_stamp,
    DEFAULT_LINE_HEIGHT_FACTOR,
)
from app.vector_tick import tick_points_in_rect, tick_stroke_width_px


def stamp_text_into_bbox(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: dict[str, float],
    text: str,
    style: StampStyle,
    font_size_pt: float,
    geometry: PageGeometry,
    wrap: bool,
) -> bool:
    """Draw text inside a pixel bbox using PDF-point sized fitting (parity with pdf_stamping)."""
    if text == "":
        return False

    padding_px = style.padding_pt / geometry.scale_y
    bottom_px = bottom_padding_pt(style, font_size_pt) / geometry.scale_y

    inner_x = bbox["x"] + padding_px
    inner_y = bbox["y"] + padding_px
    inner_w_px = max(1.0, bbox["w"] - (2 * padding_px))
    inner_h_px = max(1.0, bbox["h"] - padding_px - bottom_px)

    max_width_pt = inner_w_px * geometry.scale_x
    max_height_pt = inner_h_px * geometry.scale_y

    if wrap:
        lines, size_pt = fit_multiline_to_box_pt(
            text,
            max_width_pt=max_width_pt,
            max_height_pt=max_height_pt,
            preferred_size_pt=font_size_pt,
        )
    else:
        fitted, size_pt = fit_text_to_width_pt(text, max_width_pt=max_width_pt, preferred_size_pt=font_size_pt)
        lines = [fitted] if fitted else []

    lines = [line for line in lines if line]
    if not lines:
        return False

    size_px = size_pt / geometry.scale_y
    font = load_pil_font(max(1, round(size_px)))
    line_height_px = (size_pt * DEFAULT_LINE_HEIGHT_FACTOR) / geometry.scale_y

    baseline_y = inner_y + inner_h_px
    start_baseline = baseline_y - (len(lines) - 1) * line_height_px

    drew = False
    for i, line in enumerate(lines):
        y = start_baseline + (i * line_height_px)
        draw.text((inner_x, y), line, font=font, fill=style.text_color, anchor="ls")
        drew = True
    return drew


def stamp_toggle_mark_into_bbox(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: dict[str, float],
    style: StampStyle,
    geometry: PageGeometry,
) -> bool:
    """Draw a vector check mark (two strokes) for checkbox/radio regions."""
    padding_px = style.padding_pt / geometry.scale_y
    iw = bbox["w"] - (2 * padding_px)
    ih = bbox["h"] - (2 * padding_px)
    if iw < 4 or ih < 4:
        return False
    ix = bbox["x"] + padding_px
    iy = bbox["y"] + padding_px
    (x1, y1), (x2, y2), (x3, y3) = tick_points_in_rect(ix, iy, iw, ih)
    lw = tick_stroke_width_px(min(iw, ih))
    draw.line([(x1, y1), (x2, y2)], fill=style.text_color, width=lw)
    draw.line([(x2, y2), (x3, y3)], fill=style.text_color, width=lw)
    return True


def stamp_page_image(
    *,
    output_dir: Path,
    page_index: int,
    grounding_path: Path,
    output_image_path: Path,
    values: dict[str, str],
    style: StampStyle,
    require_all_values: bool,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stamp one page image and return a per-page result."""
    overrides = overrides or {}
    _, page_manifest = read_page_manifest(output_dir, page_index)
    grounding = load_json(grounding_path)
    geometry = build_page_geometry(page_manifest=page_manifest, grounding=grounding, page_index=page_index)
    source_image_path = image_path_from_manifest(output_dir, page_manifest)

    fields = grounding.get("fields")
    if not isinstance(fields, list):
        raise ValueError("Grounding fields must be a list.")

    missing_values: list[str] = []
    warnings: list[str] = []
    stamped_count = 0
    unsupported_count = 0

    with Image.open(source_image_path) as source:
        image = source.convert("RGBA")
    if image.size != (geometry.width_px, geometry.height_px):
        raise ValueError(
            f"PNG dimensions {image.size} do not match manifest dimensions "
            f"({geometry.width_px}, {geometry.height_px})."
        )

    draw = ImageDraw.Draw(image)
    if style.draw_debug_boxes:
        validate_hex_color(style.debug_box_color, field_name="debug_box_color")
    validate_hex_color(style.text_color, field_name="text_color")

    for idx, field in enumerate(fields):
        if not isinstance(field, dict):
            raise ValueError(f"fields[{idx}] must be an object.")
        field_id = field.get("field_id")
        field_type_raw = field.get("type")
        field_type = field_type_raw if isinstance(field_type_raw, str) else ""
        if not isinstance(field_id, str) or not field_id.strip():
            raise ValueError(f"fields[{idx}].field_id must be a non-empty string.")
        bbox = bbox_from_field(field, width_px=geometry.width_px, height_px=geometry.height_px, field_index=idx)

        if style.draw_debug_boxes:
            draw.rectangle(
                (bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]),
                outline=style.debug_box_color,
                width=2,
            )

        if not is_supported_grounding_field_type(field_type):
            unsupported_count += 1
            warnings.append(f"Skipped unsupported field type for {field_id}: {field_type_raw!r}")
            continue

        if stamps_as_text(field_type):
            if field_id not in values:
                missing_values.append(field_id)
                continue
            text_val = values[field_id]
            if text_val == "":
                continue
            font_size_pt = resolve_field_font_size_pt(
                field, field_id=field_id, overrides=overrides, default_pt=style.font_size_pt
            )
            if stamp_text_into_bbox(
                draw,
                bbox=bbox,
                text=text_val,
                style=style,
                font_size_pt=font_size_pt,
                geometry=geometry,
                wrap=field_type == "multiline_text",
            ):
                stamped_count += 1
        elif stamps_as_toggle(field_type):
            if field_id not in values:
                missing_values.append(field_id)
                continue
            raw_val = values[field_id]
            if not is_toggle_value_truthy(raw_val):
                continue
            if stamp_toggle_mark_into_bbox(draw, bbox=bbox, style=style, geometry=geometry):
                stamped_count += 1

    if require_all_values and missing_values:
        raise ValueError(f"Missing values for field_id(s): {', '.join(sorted(missing_values))}")

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_image_path, format="PNG")

    return {
        "page_index": page_index,
        "status": "succeeded",
        "source_image": geometry.image_rel,
        "grounding_file": rel_to_output(output_dir, grounding_path),
        "output_image": rel_to_output(output_dir, output_image_path),
        "field_count": len(fields),
        "stamped_count": stamped_count,
        "missing_value_count": len(missing_values),
        "unsupported_field_count": unsupported_count,
        "warnings": warnings,
    }


def stamp_qa_preview_pages(
    *,
    output_dir: Path,
    provider: str,
    refined_grounding_dir: Path,
    preview_run_dir: Path,
    values: dict[str, str],
    style: StampStyle,
    require_all_values: bool,
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Stamp previews using grounding JSON files under ``refined_grounding_dir`` (per-page paths).

    Writes ``page_XXXX.{provider}.stamped.png`` under ``preview_run_dir``.
    """
    provider_norm = provider.strip().lower()
    preview_run_dir.mkdir(parents=True, exist_ok=True)
    pages = discover_grounding_pages(refined_grounding_dir)
    results: list[dict[str, Any]] = []
    for page_index, grounding_path in pages:
        output_image_path = preview_run_dir / f"page_{page_index + 1:04d}.{provider_norm}.stamped.png"
        page_result = stamp_page_image(
            output_dir=output_dir,
            page_index=page_index,
            grounding_path=grounding_path,
            output_image_path=output_image_path,
            values=values,
            style=style,
            overrides=overrides,
            require_all_values=require_all_values,
        )
        results.append(page_result)
    return results


def run_image_stamping_for_job(
    *,
    job_id: str,
    output_dir: Path,
    provider: str,
    model: str,
    values: dict[str, str],
    style: StampStyle,
    require_all_values: bool,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stamp all grounded converted images for a job."""
    overrides = overrides or {}
    ctx = prepare_stamp_run(output_dir=output_dir, provider=provider, model=model, subdir="stamped_images")

    def _render(page_index: int, grounding_path: Path) -> dict[str, Any]:
        output_image_path = ctx.run_dir / f"page_{page_index + 1:04d}.{ctx.provider}.stamped.png"
        return stamp_page_image(
            output_dir=output_dir,
            page_index=page_index,
            grounding_path=grounding_path,
            output_image_path=output_image_path,
            values=values,
            style=style,
            overrides=overrides,
            require_all_values=require_all_values,
        )

    page_results = run_stamp_pages(output_dir=output_dir, grounding_pages=ctx.grounding_pages, render_page=_render)
    succeeded_count, failed_count = summarize_page_results(page_results)
    output_files = [p["output_image"] for p in page_results if p.get("status") == "succeeded"]

    manifest_rel = f"{ctx.run_dir_rel}/manifest.json"
    manifest = {
        "job_id": job_id,
        "provider": ctx.provider,
        "model": ctx.model,
        "stamp_run_id": ctx.stamp_run_id,
        "run_dir": ctx.run_dir_rel,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(page_results),
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "files": output_files,
        "pages": page_results,
        "style": {
            "font_size_pt": style.font_size_pt,
            "text_color": style.text_color,
            "draw_debug_boxes": style.draw_debug_boxes,
            "debug_box_color": style.debug_box_color,
        },
    }
    write_run_manifest(ctx.run_dir, manifest)

    from app.services.jobs import update_job_after_image_stamp

    finalize_job_after_stamp(
        output_dir=output_dir,
        subdir="stamped_images",
        update_job=lambda job_root_dir: update_job_after_image_stamp(
            job_root_dir, stamp_run_id=ctx.stamp_run_id, run_dir_rel=ctx.run_dir_rel
        ),
    )

    return {
        "job_id": job_id,
        "provider": ctx.provider,
        "model": ctx.model,
        "stamp_run_id": ctx.stamp_run_id,
        "run_dir": ctx.run_dir_rel,
        "manifest_path": manifest_rel,
        "page_count": len(page_results),
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "pages": page_results,
    }
