"""Deterministic geometry index from detected_lines.json."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

# Shared with grounding_validator toggle checks (avoid importing validator here).
MAX_TOGGLE_AREA = 2500

BOUNDED_SURFACES = frozenset({"solid_box", "character_boxes"})
LINE_SURFACES = frozenset({"underline", "dotted_line", "signature_line"})
TOGGLE_SURFACES = frozenset({"checkbox", "radio_circle"})


def _bbox_intersects(a: dict[str, int], b: dict[str, int]) -> bool:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    if a["x"] >= bx2 or b["x"] >= ax2:
        return False
    if a["y"] >= by2 or b["y"] >= ay2:
        return False
    return True


def _expand_bbox(bbox: dict[str, int], padding: int) -> dict[str, int]:
    return {
        "x": bbox["x"] - padding,
        "y": bbox["y"] - padding,
        "w": bbox["w"] + 2 * padding,
        "h": bbox["h"] + 2 * padding,
    }


def build_geometry_index(
    detected_lines: dict[str, Any],
    *,
    line_padding_px: int = 3,
    min_writable_height_px: int = 20,
    header_margin_ratio: float = 0.08,
) -> dict[str, Any]:
    """
    Build lines_by_id, writable_bands, cells, and forbidden_zones from detector output.
    """
    image = detected_lines.get("image") or {}
    width = int(image.get("width", 0))
    height = int(image.get("height", 0))
    raw_lines = detected_lines.get("lines") or []
    if not isinstance(raw_lines, list):
        raise ValueError("detected_lines.lines must be a list.")

    lines_by_id: dict[str, dict[str, Any]] = {}
    horizontals: list[dict[str, Any]] = []
    verticals: list[dict[str, Any]] = []
    for ln in raw_lines:
        if not isinstance(ln, dict):
            continue
        line_id = ln.get("line_id")
        if isinstance(line_id, str):
            lines_by_id[line_id] = ln
        if ln.get("orientation") == "horizontal":
            horizontals.append(ln)
        elif ln.get("orientation") == "vertical":
            verticals.append(ln)

    horizontals.sort(key=lambda ln: (ln["bbox"]["y"], ln["bbox"]["x"]))
    verticals.sort(key=lambda ln: (ln["bbox"]["x"], ln["bbox"]["y"]))

    forbidden_zones: list[dict[str, Any]] = []
    for ln in raw_lines:
        if not isinstance(ln, dict):
            continue
        bbox = ln.get("bbox")
        if not isinstance(bbox, dict):
            continue
        try:
            bb = {k: int(bbox[k]) for k in ("x", "y", "w", "h")}
        except (KeyError, TypeError, ValueError):
            continue
        forbidden_zones.append(
            {
                "source_line_id": ln.get("line_id"),
                "bbox": _expand_bbox(bb, line_padding_px),
            }
        )

    header_h = int(height * header_margin_ratio) if height > 0 else 0
    if header_h > 0:
        forbidden_zones.append(
            {
                "source_line_id": None,
                "bbox": {"x": 0, "y": 0, "w": width, "h": header_h},
                "kind": "header_margin",
            }
        )

    writable_bands: list[dict[str, Any]] = []
    margin_x = max(8, width // 100)
    for i in range(len(horizontals) - 1):
        top = horizontals[i]
        bottom = horizontals[i + 1]
        y_min = top["bbox"]["y"] + top["bbox"]["h"]
        y_max = bottom["bbox"]["y"]
        band_h = y_max - y_min
        if band_h < min_writable_height_px:
            continue
        writable_bands.append(
            {
                "y_min": y_min,
                "y_max": y_max,
                "x_min": margin_x,
                "x_max": width - margin_x,
                "between_lines": [top.get("line_id"), bottom.get("line_id")],
            }
        )

    cells: list[dict[str, Any]] = []
    for hi, top in enumerate(horizontals):
        for hj, bottom in enumerate(horizontals):
            if hj <= hi:
                continue
            y0 = top["bbox"]["y"] + top["bbox"]["h"]
            y1 = bottom["bbox"]["y"]
            if y1 - y0 < min_writable_height_px:
                continue
            for vi, left in enumerate(verticals):
                for vj, right in enumerate(verticals):
                    if vj <= vi:
                        continue
                    x0 = left["bbox"]["x"] + left["bbox"]["w"]
                    x1 = right["bbox"]["x"]
                    if x1 - x0 < min_writable_height_px:
                        continue
                    cell_index = len(cells)
                    cells.append(
                        {
                            "cell_index": cell_index,
                            "bbox": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
                            "bounding_lines": [
                                top.get("line_id"),
                                bottom.get("line_id"),
                                left.get("line_id"),
                                right.get("line_id"),
                            ],
                        }
                    )

    return {
        "page_width": width,
        "page_height": height,
        "lines_by_id": lines_by_id,
        "writable_bands": writable_bands,
        "cells": cells,
        "forbidden_zones": forbidden_zones,
    }


def geometry_summary_for_prompt(geometry: dict[str, Any], *, max_cells: int = 40) -> str:
    """Compact text block for the LLM developer/user context."""
    lines_by_id = geometry.get("lines_by_id") or {}
    line_rows = []
    for lid, ln in sorted(lines_by_id.items()):
        bb = ln.get("bbox") or {}
        line_rows.append(f"  {lid}: {ln.get('orientation')} bbox={bb}")
    bands = geometry.get("writable_bands") or []
    cells = (geometry.get("cells") or [])[:max_cells]
    parts = [
        f"Geometry summary ({geometry.get('page_width')}x{geometry.get('page_height')} px):",
        f"Lines ({len(line_rows)}):",
        *line_rows[:80],
        f"Writable bands ({len(bands)}): {bands[:15]}",
        f"Cells (showing {len(cells)} of {len(geometry.get('cells') or [])}): {cells}",
    ]
    return "\n".join(parts)


def load_detected_lines(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bbox_intersects_line(bbox: dict[str, int], line_bbox: dict[str, int], *, padding: int = 0) -> bool:
    expanded = _expand_bbox(line_bbox, padding) if padding else line_bbox
    return _bbox_intersects(bbox, expanded)


def bbox_center(bbox: dict[str, int]) -> tuple[float, float]:
    return bbox["x"] + bbox["w"] / 2.0, bbox["y"] + bbox["h"] / 2.0


def point_in_rect(px: float, py: float, rect: dict[str, int]) -> bool:
    return rect["x"] <= px <= rect["x"] + rect["w"] and rect["y"] <= py <= rect["y"] + rect["h"]


def bbox_subset_of(
    inner: dict[str, int],
    outer: dict[str, int],
    *,
    tolerance_px: int = 1,
) -> bool:
    """True when ``inner`` lies inside ``outer`` (integer bbox), within tolerance."""
    tol = tolerance_px
    return (
        inner["x"] >= outer["x"] - tol
        and inner["y"] >= outer["y"] - tol
        and inner["x"] + inner["w"] <= outer["x"] + outer["w"] + tol
        and inner["y"] + inner["h"] <= outer["y"] + outer["h"] + tol
    )


def find_containing_cell(
    cx: float,
    cy: float,
    cells: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Smallest-area cell whose bbox contains ``(cx, cy)``."""
    best: dict[str, Any] | None = None
    best_area = -1
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        cbb = cell.get("bbox")
        if not isinstance(cbb, dict):
            continue
        try:
            rect = {k: int(cbb[k]) for k in ("x", "y", "w", "h")}
        except (KeyError, TypeError, ValueError):
            continue
        if not point_in_rect(cx, cy, rect):
            continue
        area = rect["w"] * rect["h"]
        if best is None or area < best_area:
            best = cell
            best_area = area
    return best


def find_writable_band(
    cx: float,
    cy: float,
    bands: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for band in bands:
        if not isinstance(band, dict):
            continue
        try:
            x_min = int(band["x_min"])
            y_min = int(band["y_min"])
            x_max = int(band["x_max"])
            y_max = int(band["y_max"])
        except (KeyError, TypeError, ValueError):
            continue
        if x_min <= cx <= x_max and y_min <= cy <= y_max:
            return band
    return None


def clamp_bbox_to_page(bbox: dict[str, int], page_w: int, page_h: int) -> dict[str, int]:
    x = max(0, min(bbox["x"], page_w - 1))
    y = max(0, min(bbox["y"], page_h - 1))
    w = max(1, min(bbox["w"], page_w - x))
    h = max(1, min(bbox["h"], page_h - y))
    return {"x": x, "y": y, "w": w, "h": h}


def inset_bbox(
    bbox: dict[str, int],
    inset_px: int,
    *,
    page_w: int,
    page_h: int,
) -> dict[str, int]:
    if inset_px <= 0:
        return clamp_bbox_to_page(dict(bbox), page_w, page_h)
    shrunk = {
        "x": bbox["x"] + inset_px,
        "y": bbox["y"] + inset_px,
        "w": bbox["w"] - 2 * inset_px,
        "h": bbox["h"] - 2 * inset_px,
    }
    if shrunk["w"] < 1 or shrunk["h"] < 1:
        return clamp_bbox_to_page(dict(bbox), page_w, page_h)
    return clamp_bbox_to_page(shrunk, page_w, page_h)


def _effective_surface(field: dict[str, Any]) -> str:
    fsurface = field.get("field_surface")
    if isinstance(fsurface, str) and fsurface != "unknown":
        return fsurface
    ftype = field.get("type")
    if ftype in ("checkbox", "radio"):
        return "checkbox" if ftype == "checkbox" else "radio_circle"
    if ftype == "multiline_text":
        return "solid_box"
    return "unknown"


def _parse_field_bbox(field: dict[str, Any]) -> dict[str, int] | None:
    bbox = field.get("bbox")
    if not isinstance(bbox, dict):
        return None
    try:
        return {k: int(bbox[k]) for k in ("x", "y", "w", "h")}
    except (KeyError, TypeError, ValueError):
        return None


def _set_supporting_lines(field: dict[str, Any], line_ids: list[Any]) -> None:
    ids = [lid for lid in line_ids if isinstance(lid, str) and lid]
    if not ids:
        return
    existing = field.get("supporting_lines")
    if not isinstance(existing, list) or not existing:
        field["supporting_lines"] = ids


def normalize_field_bbox(
    field: dict[str, Any],
    geometry: dict[str, Any],
    *,
    stamp_inset_px: int,
    page_w: int,
    page_h: int,
    aggressive: bool = False,
) -> dict[str, Any]:
    """Return a field copy with bbox snapped to geometry when applicable."""
    out = copy.deepcopy(field)
    bbox = _parse_field_bbox(out)
    if bbox is None:
        return out

    cells = geometry.get("cells") or []
    bands = geometry.get("writable_bands") or []
    cx, cy = bbox_center(bbox)
    surface = _effective_surface(out)
    ftype = out.get("type")

    cell = find_containing_cell(cx, cy, cells)
    band = find_writable_band(cx, cy, bands)

    if surface in BOUNDED_SURFACES or (
        surface == "unknown" and ftype in ("text", "numeric", "date", "multiline_text")
    ):
        if cell is not None:
            cbb = cell.get("bbox")
            if isinstance(cbb, dict):
                snapped = {k: int(cbb[k]) for k in ("x", "y", "w", "h")}
                snapped = inset_bbox(snapped, stamp_inset_px, page_w=page_w, page_h=page_h)
                out["bbox"] = snapped
                _set_supporting_lines(out, cell.get("bounding_lines") or [])

    elif surface in TOGGLE_SURFACES or ftype in ("checkbox", "radio"):
        if cell is not None:
            cbb = cell.get("bbox")
            if isinstance(cbb, dict):
                rect = {k: int(cbb[k]) for k in ("x", "y", "w", "h")}
                if rect["w"] * rect["h"] <= MAX_TOGGLE_AREA or aggressive:
                    inset = 1 if stamp_inset_px > 0 else 0
                    out["bbox"] = inset_bbox(rect, inset, page_w=page_w, page_h=page_h)
                    _set_supporting_lines(out, cell.get("bounding_lines") or [])

    elif surface in LINE_SURFACES:
        if band is not None:
            try:
                y_min = int(band["y_min"])
                y_max = int(band["y_max"])
                x_min = int(band["x_min"])
                x_max = int(band["x_max"])
            except (KeyError, TypeError, ValueError):
                pass
            else:
                band_h = y_max - y_min
                inset = stamp_inset_px
                new_y = y_min + inset
                new_h = band_h - 2 * inset
                if new_h < 1:
                    new_y = y_min
                    new_h = max(1, band_h)
                new_x = max(x_min, bbox["x"])
                new_w = bbox["w"]
                if new_x + new_w > x_max:
                    new_w = max(1, x_max - new_x)
                if aggressive:
                    new_x = x_min + inset
                    new_w = max(1, (x_max - x_min) - 2 * inset)
                out["bbox"] = clamp_bbox_to_page(
                    {"x": new_x, "y": new_y, "w": new_w, "h": new_h},
                    page_w,
                    page_h,
                )
                _set_supporting_lines(out, band.get("between_lines") or [])

    elif aggressive and cell is not None:
        cbb = cell.get("bbox")
        if isinstance(cbb, dict):
            snapped = {k: int(cbb[k]) for k in ("x", "y", "w", "h")}
            out["bbox"] = inset_bbox(snapped, stamp_inset_px, page_w=page_w, page_h=page_h)
            _set_supporting_lines(out, cell.get("bounding_lines") or [])

    else:
        out["bbox"] = clamp_bbox_to_page(bbox, page_w, page_h)

    return out


def normalize_page_grounding(
    payload: dict[str, Any],
    geometry: dict[str, Any],
    *,
    stamp_inset_px: int,
    page_w: int,
    page_h: int,
    aggressive: bool = False,
) -> dict[str, Any]:
    """Return payload copy with field bboxes normalized against ``geometry``."""
    out = copy.deepcopy(payload)
    fields = out.get("fields")
    if not isinstance(fields, list):
        return out
    normalized_fields: list[Any] = []
    for field in fields:
        if not isinstance(field, dict):
            normalized_fields.append(field)
            continue
        normalized_fields.append(
            normalize_field_bbox(
                field,
                geometry,
                stamp_inset_px=stamp_inset_px,
                page_w=page_w,
                page_h=page_h,
                aggressive=aggressive,
            )
        )
    out["fields"] = normalized_fields
    return out
