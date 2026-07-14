"""Render stamped values onto original PDF pages (export path, PRD E2)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz

from app.grounding_field_types import (
    is_supported_grounding_field_type,
    is_toggle_value_truthy,
    stamps_as_text,
    stamps_as_toggle,
)
from app.services.stamping_common import (
    FONT_NAME,
    FONT_PATH,
    StampStyle,
    bbox_from_field,
    bottom_padding_pt,
    build_page_geometry,
    fit_multiline_to_box_pt,
    fit_text_to_width_pt,
    hex_to_rgb_tuple,
    load_json,
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
from app.vector_tick import tick_points_in_rect, tick_stroke_width_pt

# Fall back to a PDF base-14 font when the shipped TTF is unavailable on disk
# (keeps stamp-pdf functional even if the font asset is missing in an environment).
_INSERT_TEXT_FONT_KWARGS: dict[str, str] = (
    {"fontname": FONT_NAME, "fontfile": FONT_PATH} if FONT_PATH else {"fontname": "helv"}
)


def _map_bbox_to_pdf_points(
    *,
    bbox: dict[str, float],
    pdf_w_pt: float,
    pdf_h_pt: float,
    image_w_px: int,
    image_h_px: int,
) -> tuple[float, float, float, float]:
    scale_x = pdf_w_pt / float(image_w_px)
    scale_y = pdf_h_pt / float(image_h_px)
    pdf_x = bbox["x"] * scale_x
    pdf_y = pdf_h_pt - ((bbox["y"] + bbox["h"]) * scale_y)
    pdf_w = bbox["w"] * scale_x
    pdf_h = bbox["h"] * scale_y
    return pdf_x, pdf_y, pdf_w, pdf_h


def _pdf_bl_rect_to_pymupdf_rect(
    *,
    pdf_x: float,
    pdf_y: float,
    pdf_w: float,
    pdf_h: float,
    pdf_h_pt: float,
) -> fitz.Rect:
    """Convert a bottom-left PDF rect to PyMuPDF's top-left page coordinate space."""
    y0 = pdf_h_pt - (pdf_y + pdf_h)
    y1 = pdf_h_pt - pdf_y
    return fitz.Rect(pdf_x, y0, pdf_x + pdf_w, y1)


def _stamp_text_into_rect(
    page: fitz.Page,
    *,
    rect: fitz.Rect,
    text: str,
    style: StampStyle,
    font_size_pt: float,
    font_color: tuple[float, float, float],
    wrap: bool,
) -> bool:
    """Draw text inside a PDF-point rect using the same fitting/wrapping as the preview path."""
    if text == "":
        return False

    bottom_gap = bottom_padding_pt(style, font_size_pt)
    inner = fitz.Rect(
        rect.x0 + style.padding_pt,
        rect.y0 + style.padding_pt,
        max(rect.x0 + style.padding_pt + 1.0, rect.x1 - style.padding_pt),
        max(rect.y0 + style.padding_pt + 1.0, rect.y1 - bottom_gap),
    )

    if wrap:
        lines, size_pt = fit_multiline_to_box_pt(
            text,
            max_width_pt=inner.width,
            max_height_pt=inner.height,
            preferred_size_pt=font_size_pt,
        )
    else:
        fitted, size_pt = fit_text_to_width_pt(text, max_width_pt=inner.width, preferred_size_pt=font_size_pt)
        lines = [fitted] if fitted else []

    lines = [line for line in lines if line]
    if not lines:
        return False

    line_height_pt = size_pt * DEFAULT_LINE_HEIGHT_FACTOR
    baseline_y = inner.y1
    start_baseline = baseline_y - (len(lines) - 1) * line_height_pt

    drew = False
    for i, line in enumerate(lines):
        y = start_baseline + (i * line_height_pt)
        page.insert_text(
            fitz.Point(inner.x0, y),
            text=line,
            fontsize=size_pt,
            color=font_color,
            overlay=True,
            **_INSERT_TEXT_FONT_KWARGS,
        )
        drew = True
    return drew


def _stamp_toggle_mark_pdf(
    page: fitz.Page,
    *,
    rect: fitz.Rect,
    style: StampStyle,
    font_color: tuple[float, float, float],
) -> bool:
    """Draw a vector check mark using two line segments (PDF points)."""
    p = style.padding_pt
    inner_w = rect.width - (2 * p)
    inner_h = rect.height - (2 * p)
    if inner_w < 3 or inner_h < 3:
        return False
    x0 = rect.x0 + p
    y0 = rect.y0 + p
    (x1, y1), (x2, y2), (x3, y3) = tick_points_in_rect(x0, y0, inner_w, inner_h)
    lw = tick_stroke_width_pt(min(inner_w, inner_h))
    page.draw_line(fitz.Point(x1, y1), fitz.Point(x2, y2), color=font_color, width=lw, overlay=True)
    page.draw_line(fitz.Point(x2, y2), fitz.Point(x3, y3), color=font_color, width=lw, overlay=True)
    return True


def run_pdf_stamping_for_job(
    *,
    job_id: str,
    input_pdf: Path,
    output_dir: Path,
    provider: str,
    model: str,
    values: dict[str, str],
    style: StampStyle,
    require_all_values: bool,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    if not input_pdf.is_file():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    ctx = prepare_stamp_run(output_dir=output_dir, provider=provider, model=model, subdir="stamped_pdfs")
    output_pdf_rel = f"{ctx.run_dir_rel}/stamped.{ctx.provider}.pdf"
    output_pdf = output_dir / output_pdf_rel

    validate_hex_color(style.text_color, field_name="text_color")
    validate_hex_color(style.debug_box_color, field_name="debug_box_color")
    font_color = hex_to_rgb_tuple(style.text_color)
    debug_color = hex_to_rgb_tuple(style.debug_box_color)

    with fitz.open(input_pdf) as src_doc:

        def _render(page_index: int, grounding_path: Path) -> dict[str, Any]:
            if page_index >= src_doc.page_count:
                raise ValueError(f"Grounding page_index {page_index} exceeds PDF page count {src_doc.page_count}.")
            page_manifest_rel, page_manifest = read_page_manifest(output_dir, page_index)
            grounding = load_json(grounding_path)
            geometry = build_page_geometry(page_manifest=page_manifest, grounding=grounding, page_index=page_index)
            page = src_doc[page_index]
            fields = grounding.get("fields")
            if not isinstance(fields, list):
                raise ValueError("Grounding fields must be a list.")

            missing_values: list[str] = []
            warnings: list[str] = []
            stamped_count = 0
            unsupported_count = 0

            for idx, field in enumerate(fields):
                if not isinstance(field, dict):
                    raise ValueError(f"fields[{idx}] must be an object.")
                field_id = field.get("field_id")
                field_type_raw = field.get("type")
                field_type = field_type_raw if isinstance(field_type_raw, str) else ""
                if not isinstance(field_id, str) or not field_id.strip():
                    raise ValueError(f"fields[{idx}].field_id must be a non-empty string.")

                bbox = bbox_from_field(
                    field, width_px=geometry.width_px, height_px=geometry.height_px, field_index=idx
                )
                pdf_x, pdf_y, pdf_w, pdf_h = _map_bbox_to_pdf_points(
                    bbox=bbox,
                    pdf_w_pt=geometry.pdf_width_pt,
                    pdf_h_pt=geometry.pdf_height_pt,
                    image_w_px=geometry.width_px,
                    image_h_px=geometry.height_px,
                )
                field_rect = _pdf_bl_rect_to_pymupdf_rect(
                    pdf_x=pdf_x,
                    pdf_y=pdf_y,
                    pdf_w=pdf_w,
                    pdf_h=pdf_h,
                    pdf_h_pt=geometry.pdf_height_pt,
                )

                if style.draw_debug_boxes:
                    page.draw_rect(field_rect, color=debug_color, width=0.8, overlay=True)

                if not is_supported_grounding_field_type(field_type):
                    unsupported_count += 1
                    warnings.append(f"Skipped unsupported field type for {field_id}: {field_type_raw!r}")
                    continue

                if stamps_as_text(field_type):
                    if field_id not in values:
                        missing_values.append(field_id)
                        continue
                    text_value = values[field_id]
                    if text_value == "":
                        continue
                    font_size_pt = resolve_field_font_size_pt(
                        field, field_id=field_id, overrides=overrides, default_pt=style.font_size_pt
                    )
                    if _stamp_text_into_rect(
                        page,
                        rect=field_rect,
                        text=text_value,
                        style=style,
                        font_size_pt=font_size_pt,
                        font_color=font_color,
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
                    if _stamp_toggle_mark_pdf(page, rect=field_rect, style=style, font_color=font_color):
                        stamped_count += 1

            if require_all_values and missing_values:
                raise ValueError(f"Missing values for field_id(s): {', '.join(sorted(missing_values))}")

            return {
                "page_index": page_index,
                "status": "succeeded",
                "grounding_file": rel_to_output(output_dir, grounding_path),
                "page_manifest": str(page_manifest_rel).replace("\\", "/"),
                "source_pdf": str(input_pdf.name),
                "output_pdf": output_pdf_rel,
                "field_count": len(fields),
                "stamped_count": stamped_count,
                "missing_value_count": len(missing_values),
                "unsupported_field_count": unsupported_count,
                "warnings": warnings,
            }

        page_results = run_stamp_pages(
            output_dir=output_dir, grounding_pages=ctx.grounding_pages, render_page=_render
        )

        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        src_doc.save(output_pdf)

    succeeded_count, failed_count = summarize_page_results(page_results)
    manifest_rel = f"{ctx.run_dir_rel}/manifest.json"
    manifest = {
        "job_id": job_id,
        "provider": ctx.provider,
        "model": ctx.model,
        "stamp_run_id": ctx.stamp_run_id,
        "run_dir": ctx.run_dir_rel,
        "output_pdf": output_pdf_rel,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(page_results),
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "files": [output_pdf_rel],
        "pages": page_results,
        "style": {
            "font_size_pt": style.font_size_pt,
            "text_color": style.text_color,
            "draw_debug_boxes": style.draw_debug_boxes,
            "debug_box_color": style.debug_box_color,
        },
    }
    write_run_manifest(ctx.run_dir, manifest)

    from app.services.jobs import update_job_after_pdf_stamp

    finalize_job_after_stamp(
        output_dir=output_dir,
        subdir="stamped_pdfs",
        update_job=lambda job_root_dir: update_job_after_pdf_stamp(
            job_root_dir, stamp_run_id=ctx.stamp_run_id, pdf_rel=output_pdf_rel
        ),
    )

    return {
        "job_id": job_id,
        "provider": ctx.provider,
        "model": ctx.model,
        "stamp_run_id": ctx.stamp_run_id,
        "run_dir": ctx.run_dir_rel,
        "manifest_path": manifest_rel,
        "output_pdf": output_pdf_rel,
        "page_count": len(page_results),
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "pages": page_results,
    }
