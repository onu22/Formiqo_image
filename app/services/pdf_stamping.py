"""Stamp user-provided values onto original PDF pages."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz

_GROUNDING_PAGE_RE = re.compile(r"^page_(\d{4})\.fields\.json$")
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class StampPdfStyle:
    font_size_pt: float = 11.0
    font_color: str = "#111111"
    padding_pt: float = 1.0
    draw_debug_boxes: bool = False
    debug_box_color: str = "#ff0000"


def _validate_hex_color(value: str, *, field_name: str) -> str:
    if not _HEX_COLOR_RE.match(value):
        raise ValueError(f"{field_name} must be a hex color like #111111.")
    return value


def _hex_to_rgb_tuple(value: str) -> tuple[float, float, float]:
    value_norm = _validate_hex_color(value, field_name="color")
    return (
        int(value_norm[1:3], 16) / 255.0,
        int(value_norm[3:5], 16) / 255.0,
        int(value_norm[5:7], 16) / 255.0,
    )


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


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


def _read_page_manifest(output_dir: Path, page_index: int) -> tuple[Path, dict[str, Any]]:
    rel = Path("converted_images") / "pages" / f"page_{page_index + 1:04d}.json"
    path = output_dir / rel
    if not path.is_file():
        raise FileNotFoundError(f"Conversion page manifest not found: {rel}")
    return rel, _load_json(path)


def _validate_page_inputs(
    *,
    page_manifest: dict[str, Any],
    grounding: dict[str, Any],
    page_index: int,
) -> tuple[float, float, int, int]:
    pdf_node = page_manifest.get("pdf")
    image_node = page_manifest.get("image")
    if not isinstance(pdf_node, dict):
        raise ValueError("Conversion page manifest missing pdf object.")
    if not isinstance(image_node, dict):
        raise ValueError("Conversion page manifest missing image object.")

    try:
        pdf_w_pt = float(pdf_node["width_pt"])
        pdf_h_pt = float(pdf_node["height_pt"])
        image_w_px = int(image_node["saved_image_width_px"])
        image_h_px = int(image_node["saved_image_height_px"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Conversion page manifest is missing required page dimensions.") from exc

    g_page_index = int(grounding.get("page_index", -1))
    g_width = int(grounding.get("width", -1))
    g_height = int(grounding.get("height", -1))
    g_unit = grounding.get("unit")
    g_origin = grounding.get("origin")

    if g_page_index != page_index:
        raise ValueError(f"Grounding page_index ({g_page_index}) does not match expected {page_index}.")
    if g_width != image_w_px or g_height != image_h_px:
        raise ValueError(
            f"Grounding dimensions ({g_width}x{g_height}) do not match converted image ({image_w_px}x{image_h_px})."
        )
    if g_unit != "px":
        raise ValueError('Grounding unit must be "px".')
    if g_origin != "top-left":
        raise ValueError('Grounding origin must be "top-left".')

    if pdf_w_pt <= 0 or pdf_h_pt <= 0:
        raise ValueError("PDF dimensions must be positive.")
    if image_w_px <= 0 or image_h_px <= 0:
        raise ValueError("Image dimensions must be positive.")
    return pdf_w_pt, pdf_h_pt, image_w_px, image_h_px


def _bbox_from_field(field: dict[str, Any], *, width_px: int, height_px: int, field_index: int) -> dict[str, float]:
    bbox = field.get("bbox")
    if not isinstance(bbox, dict):
        raise ValueError(f"fields[{field_index}].bbox must be an object.")
    expected = {"x", "y", "w", "h"}
    if set(bbox.keys()) != expected:
        raise ValueError(f"fields[{field_index}].bbox keys must be exactly {sorted(expected)}.")
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["w"])
        h = float(bbox["h"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fields[{field_index}].bbox values must be numeric.") from exc

    if w <= 0 or h <= 0:
        raise ValueError(f"fields[{field_index}].bbox requires w > 0 and h > 0.")
    if x < 0 or y < 0:
        raise ValueError(f"fields[{field_index}].bbox requires x >= 0 and y >= 0.")
    if x + w > width_px or y + h > height_px:
        raise ValueError(f"fields[{field_index}].bbox exceeds image bounds ({width_px}x{height_px}).")
    return {"x": x, "y": y, "w": w, "h": h}


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


def _fit_fontsize_for_rect(value: str, *, rect: fitz.Rect, preferred_size: float) -> float:
    size = preferred_size
    while size >= 5.0:
        w = fitz.get_text_length(value, fontsize=size)
        if w <= rect.width and size <= rect.height + 1:
            return size
        size -= 0.5
    return 5.0


def run_pdf_stamping_for_job(
    *,
    job_id: str,
    input_pdf: Path,
    output_dir: Path,
    provider: str,
    model: str,
    values: dict[str, str],
    style: StampPdfStyle,
    require_all_values: bool,
) -> dict[str, Any]:
    provider_norm = provider.strip().lower()
    if not provider_norm:
        raise ValueError("provider must be a non-empty string.")
    if not model.strip():
        raise ValueError("model must be a non-empty string.")
    if not input_pdf.is_file():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    converted_pages_dir = output_dir / "converted_images" / "pages"
    if not converted_pages_dir.is_dir():
        raise FileNotFoundError("Conversion page manifests not found: converted_images/pages")

    grounding_dir = output_dir / "field_grounding"
    if not grounding_dir.is_dir():
        raise FileNotFoundError(
            f"Field grounding run not found for provider={provider_norm}, model={model} "
            f"(expected directory: field_grounding)."
        )
    _assert_grounding_run_matches(grounding_dir, provider=provider_norm, model=model)

    stamp_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir_rel = f"stamped_pdfs/{stamp_run_id}"
    run_dir = output_dir / run_dir_rel
    output_pdf_rel = f"{run_dir_rel}/stamped.{provider_norm}.pdf"
    output_pdf = output_dir / output_pdf_rel
    grounding_pages = _discover_grounding_pages(grounding_dir)

    _validate_hex_color(style.font_color, field_name="font_color")
    _validate_hex_color(style.debug_box_color, field_name="debug_box_color")
    font_color = _hex_to_rgb_tuple(style.font_color)
    debug_color = _hex_to_rgb_tuple(style.debug_box_color)

    page_results: list[dict[str, Any]] = []
    output_files = [output_pdf_rel]

    with fitz.open(input_pdf) as src_doc:
        for page_index, grounding_path in grounding_pages:
            try:
                if page_index >= src_doc.page_count:
                    raise ValueError(f"Grounding page_index {page_index} exceeds PDF page count {src_doc.page_count}.")
                page_manifest_rel, page_manifest = _read_page_manifest(output_dir, page_index)
                grounding = _load_json(grounding_path)
                pdf_w_pt, pdf_h_pt, image_w_px, image_h_px = _validate_page_inputs(
                    page_manifest=page_manifest,
                    grounding=grounding,
                    page_index=page_index,
                )
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
                    field_type = field.get("type")
                    if not isinstance(field_id, str) or not field_id.strip():
                        raise ValueError(f"fields[{idx}].field_id must be a non-empty string.")

                    bbox = _bbox_from_field(field, width_px=image_w_px, height_px=image_h_px, field_index=idx)
                    pdf_x, pdf_y, pdf_w, pdf_h = _map_bbox_to_pdf_points(
                        bbox=bbox,
                        pdf_w_pt=pdf_w_pt,
                        pdf_h_pt=pdf_h_pt,
                        image_w_px=image_w_px,
                        image_h_px=image_h_px,
                    )
                    pymupdf_rect = _pdf_bl_rect_to_pymupdf_rect(
                        pdf_x=pdf_x,
                        pdf_y=pdf_y,
                        pdf_w=pdf_w,
                        pdf_h=pdf_h,
                        pdf_h_pt=pdf_h_pt,
                    )
                    rect = fitz.Rect(
                        pymupdf_rect.x0 + style.padding_pt,
                        pymupdf_rect.y0 + style.padding_pt,
                        max(pymupdf_rect.x0 + 1.0, pymupdf_rect.x1 - style.padding_pt),
                        max(pymupdf_rect.y0 + 1.0, pymupdf_rect.y1 - style.padding_pt),
                    )

                    if style.draw_debug_boxes:
                        page.draw_rect(
                            pymupdf_rect,
                            color=debug_color,
                            width=0.8,
                            overlay=True,
                        )

                    if field_type != "text":
                        unsupported_count += 1
                        warnings.append(f"Skipped unsupported field type for {field_id}: {field_type}")
                        continue
                    if field_id not in values:
                        missing_values.append(field_id)
                        continue

                    text_value = values[field_id]
                    if text_value == "":
                        continue
                    font_size = _fit_fontsize_for_rect(text_value, rect=rect, preferred_size=style.font_size_pt)
                    page.insert_text(
                        fitz.Point(rect.x0, rect.y0 + font_size),
                        fontsize=font_size,
                        text=text_value,
                        color=font_color,
                        overlay=True,
                    )
                    stamped_count += 1

                if require_all_values and missing_values:
                    raise ValueError(f"Missing values for field_id(s): {', '.join(sorted(missing_values))}")

                page_results.append(
                    {
                        "page_index": page_index,
                        "status": "succeeded",
                        "grounding_file": str(grounding_path.relative_to(output_dir)).replace("\\", "/"),
                        "page_manifest": str(page_manifest_rel).replace("\\", "/"),
                        "source_pdf": str(input_pdf.name),
                        "output_pdf": output_pdf_rel,
                        "field_count": len(fields),
                        "stamped_count": stamped_count,
                        "missing_value_count": len(missing_values),
                        "unsupported_field_count": unsupported_count,
                        "warnings": warnings,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - preserve per-page errors for API response
                page_results.append(
                    {
                        "page_index": page_index,
                        "status": "failed",
                        "grounding_file": str(grounding_path.relative_to(output_dir)).replace("\\", "/"),
                        "error": str(exc),
                    }
                )

        run_dir.mkdir(parents=True, exist_ok=True)
        src_doc.save(output_pdf)

    succeeded_count = sum(1 for page in page_results if page["status"] == "succeeded")
    failed_count = len(page_results) - succeeded_count
    manifest_rel = f"{run_dir_rel}/manifest.json"
    manifest = {
        "job_id": job_id,
        "provider": provider_norm,
        "model": model,
        "stamp_run_id": stamp_run_id,
        "run_dir": run_dir_rel,
        "output_pdf": output_pdf_rel,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(page_results),
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "files": output_files,
        "pages": page_results,
        "style": {
            "font_size_pt": style.font_size_pt,
            "font_color": style.font_color,
            "padding_pt": style.padding_pt,
            "draw_debug_boxes": style.draw_debug_boxes,
            "debug_box_color": style.debug_box_color,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "job_id": job_id,
        "provider": provider_norm,
        "model": model,
        "stamp_run_id": stamp_run_id,
        "run_dir": run_dir_rel,
        "manifest_path": manifest_rel,
        "output_pdf": output_pdf_rel,
        "page_count": len(page_results),
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "pages": page_results,
    }
