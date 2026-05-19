"""Deterministic validation of AI field grounding against line geometry."""

from __future__ import annotations

from typing import Any

from app.grounding_field_types import (
    ALLOWED_AI_GROUNDING_FIELD_TYPES,
    ALLOWED_FIELD_SURFACES,
    TOGGLE_GROUNDING_TYPES,
)
from app.services.form_geometry import (
    BOUNDED_SURFACES,
    LINE_SURFACES,
    MAX_TOGGLE_AREA,
    bbox_center,
    bbox_intersects_line,
    bbox_subset_of,
    build_geometry_index,
    find_containing_cell,
    point_in_rect,
)

TEXT_LIKE_TYPES = frozenset(
    {"text", "multiline_text", "date", "signature", "character_boxes", "numeric", "unknown"}
)

MIN_TOGGLE_AREA = 16
MAX_TOGGLE_ASPECT = 4.0

CELL_SUBSET_TOLERANCE_PX = 1


class GroundingValidationError(Exception):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        self.errors = errors
        super().__init__(f"{len(errors)} validation error(s)")


def _parse_bbox(field: dict[str, Any], field_id: str) -> dict[str, int]:
    bbox = field.get("bbox")
    if not isinstance(bbox, dict):
        raise ValueError(f"fields[{field_id}].bbox must be an object.")
    try:
        x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"fields[{field_id}].bbox values must be integers.") from exc
    if w <= 0 or h <= 0:
        raise ValueError(f"fields[{field_id}].bbox requires w > 0 and h > 0.")
    if x < 0 or y < 0:
        raise ValueError(f"fields[{field_id}].bbox requires x >= 0 and y >= 0.")
    return {"x": x, "y": y, "w": w, "h": h}


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


def _supporting_line_ids(field: dict[str, Any]) -> set[str]:
    supporting = field.get("supporting_lines")
    if not isinstance(supporting, list):
        return set()
    return {lid for lid in supporting if isinstance(lid, str) and lid}


def _boundary_line_ids(cell: dict[str, Any] | None, field: dict[str, Any]) -> set[str]:
    ids = _supporting_line_ids(field)
    if cell is not None:
        for lid in cell.get("bounding_lines") or []:
            if isinstance(lid, str) and lid:
                ids.add(lid)
    return ids


def _nearest_horizontal_line_id(
    bbox: dict[str, int],
    raw_lines: list[Any],
    *,
    bottom_third: bool = True,
) -> str | None:
    """Horizontal line whose centerline is nearest the bbox bottom (underline anchor)."""
    target_y = bbox["y"] + bbox["h"] * (2 / 3 if bottom_third else 0.5)
    best_id: str | None = None
    best_dist = float("inf")
    for ln in raw_lines:
        if not isinstance(ln, dict) or ln.get("orientation") != "horizontal":
            continue
        lbb = ln.get("bbox")
        if not isinstance(lbb, dict):
            continue
        try:
            ly = int(lbb["y"]) + int(lbb["h"]) // 2
        except (KeyError, TypeError, ValueError):
            continue
        dist = abs(ly - target_y)
        if dist < best_dist:
            best_dist = dist
            lid = ln.get("line_id")
            best_id = lid if isinstance(lid, str) else None
    return best_id


def _field_line_violations(
    field: dict[str, Any],
    bbox: dict[str, int],
    *,
    geometry: dict[str, Any],
    raw_lines: list[Any],
    line_padding_px: int,
    ftype: str | None,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    cells = geometry.get("cells") or []
    cx, cy = bbox_center(bbox)
    surface = _effective_surface(field)
    cell = find_containing_cell(cx, cy, cells)
    boundary_ids = _boundary_line_ids(cell, field)

    is_bounded = surface in BOUNDED_SURFACES or (
        surface == "unknown"
        and ftype in TEXT_LIKE_TYPES
        and cell is not None
    )
    is_line_surface = surface in LINE_SURFACES
    anchor_id: str | None = None
    if is_line_surface:
        supporting = _supporting_line_ids(field)
        horizontals = [
            lid
            for lid in supporting
            if isinstance(lid, str) and lid.startswith("line_h_")
        ]
        if horizontals:
            anchor_id = horizontals[0]
        else:
            anchor_id = _nearest_horizontal_line_id(bbox, raw_lines)

    if is_bounded and cell is not None:
        cbb = cell.get("bbox")
        if isinstance(cbb, dict):
            try:
                cell_rect = {k: int(cbb[k]) for k in ("x", "y", "w", "h")}
            except (KeyError, TypeError, ValueError):
                cell_rect = None
            if cell_rect is not None and not bbox_subset_of(
                bbox, cell_rect, tolerance_px=CELL_SUBSET_TOLERANCE_PX
            ):
                violations.append(
                    {
                        "field_id": field.get("field_id"),
                        "code": "outside_cell",
                        "message": "bbox extends outside the containing table/box cell.",
                        "detail": {"bbox": bbox, "cell_bbox": cell_rect},
                    }
                )

    for ln in raw_lines:
        if not isinstance(ln, dict):
            continue
        line_id = ln.get("line_id")
        lbb = ln.get("bbox")
        if not isinstance(lbb, dict):
            continue
        try:
            line_bb = {k: int(lbb[k]) for k in ("x", "y", "w", "h")}
        except (KeyError, TypeError, ValueError):
            continue
        if not bbox_intersects_line(bbox, line_bb, padding=line_padding_px):
            continue

        lid_str = line_id if isinstance(line_id, str) else None

        if is_bounded and cell is not None and lid_str in boundary_ids:
            continue

        if is_line_surface and lid_str == anchor_id:
            continue

        if ftype in TOGGLE_GROUNDING_TYPES and cell is not None and lid_str in boundary_ids:
            continue

        violations.append(
            {
                "field_id": field.get("field_id"),
                "code": "crosses_line",
                "message": f"bbox intersects line {line_id!r}.",
                "detail": {"line_id": line_id, "bbox": bbox},
            }
        )

    return violations


def validate_page_grounding(
    payload: dict[str, Any],
    *,
    detected_lines: dict[str, Any],
    page_manifest: dict[str, Any],
    line_padding_px: int = 3,
    geometry: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Return a list of validation errors (empty if valid).
    Each error: ``{field_id, code, message, detail}``.
    """
    errors: list[dict[str, Any]] = []
    if geometry is None:
        geometry = build_geometry_index(detected_lines, line_padding_px=line_padding_px)

    image_node = page_manifest.get("image")
    if not isinstance(image_node, dict):
        errors.append(
            {
                "field_id": None,
                "code": "manifest_image",
                "message": "Page manifest missing image object.",
                "detail": None,
            }
        )
        return errors

    try:
        img_w = int(image_node["saved_image_width_px"])
        img_h = int(image_node["saved_image_height_px"])
    except (KeyError, TypeError, ValueError):
        errors.append(
            {
                "field_id": None,
                "code": "manifest_dims",
                "message": "Page manifest missing saved image dimensions.",
                "detail": None,
            }
        )
        return errors

    g_w = int(payload.get("width", -1))
    g_h = int(payload.get("height", -1))
    g_page = int(payload.get("page_index", -1))
    manifest_page = int(page_manifest.get("page_index", g_page))

    if g_w != img_w or g_h != img_h:
        errors.append(
            {
                "field_id": None,
                "code": "dimension_mismatch",
                "message": f"Grounding dimensions ({g_w}x{g_h}) != manifest ({img_w}x{img_h}).",
                "detail": None,
            }
        )
    if g_page != manifest_page:
        errors.append(
            {
                "field_id": None,
                "code": "page_index_mismatch",
                "message": f"Grounding page_index {g_page} != manifest {manifest_page}.",
                "detail": None,
            }
        )
    if payload.get("unit") != "px" or payload.get("origin") != "top-left":
        errors.append(
            {
                "field_id": None,
                "code": "coordinate_space",
                "message": "unit must be px and origin must be top-left.",
                "detail": None,
            }
        )

    fields = payload.get("fields")
    if not isinstance(fields, list):
        errors.append(
            {
                "field_id": None,
                "code": "fields_type",
                "message": "fields must be a list.",
                "detail": None,
            }
        )
        return errors

    seen_ids: set[str] = set()
    lines_by_id = geometry.get("lines_by_id") or {}
    raw_lines = detected_lines.get("lines") or []

    for field in fields:
        if not isinstance(field, dict):
            errors.append(
                {"field_id": None, "code": "field_shape", "message": "Each field must be an object.", "detail": None}
            )
            continue

        field_id = field.get("field_id")
        fid = field_id if isinstance(field_id, str) and field_id.strip() else "?"
        ftype = field.get("type")
        fsurface = field.get("field_surface")

        if not isinstance(field_id, str) or not field_id.strip():
            errors.append(
                {
                    "field_id": fid,
                    "code": "field_id",
                    "message": "field_id must be a non-empty string.",
                    "detail": None,
                }
            )
            continue
        if field_id in seen_ids:
            errors.append(
                {
                    "field_id": field_id,
                    "code": "duplicate_field_id",
                    "message": f"Duplicate field_id: {field_id}",
                    "detail": None,
                }
            )
        seen_ids.add(field_id)

        if ftype not in ALLOWED_AI_GROUNDING_FIELD_TYPES:
            errors.append(
                {
                    "field_id": field_id,
                    "code": "invalid_type",
                    "message": f"Invalid type {ftype!r}.",
                    "detail": {"allowed": sorted(ALLOWED_AI_GROUNDING_FIELD_TYPES)},
                }
            )
        if fsurface not in ALLOWED_FIELD_SURFACES:
            errors.append(
                {
                    "field_id": field_id,
                    "code": "invalid_surface",
                    "message": f"Invalid field_surface {fsurface!r}.",
                    "detail": {"allowed": sorted(ALLOWED_FIELD_SURFACES)},
                }
            )

        try:
            bbox = _parse_bbox(field, field_id)
        except ValueError as exc:
            errors.append(
                {"field_id": field_id, "code": "bbox", "message": str(exc), "detail": None}
            )
            continue

        if bbox["x"] + bbox["w"] > img_w or bbox["y"] + bbox["h"] > img_h:
            errors.append(
                {
                    "field_id": field_id,
                    "code": "out_of_bounds",
                    "message": f"bbox exceeds page bounds ({img_w}x{img_h}).",
                    "detail": bbox,
                }
            )

        for violation in _field_line_violations(
            field,
            bbox,
            geometry=geometry,
            raw_lines=raw_lines,
            line_padding_px=line_padding_px,
            ftype=ftype if isinstance(ftype, str) else None,
        ):
            violation["field_id"] = field_id
            errors.append(violation)

        cx, cy = bbox_center(bbox)
        forbidden = geometry.get("forbidden_zones") or []
        in_header = False
        for zone in forbidden:
            zbb = zone.get("bbox")
            if isinstance(zbb, dict) and point_in_rect(cx, cy, zbb):
                if zone.get("kind") == "header_margin":
                    in_header = True
                elif zone.get("source_line_id"):
                    pass

        if in_header and ftype in TEXT_LIKE_TYPES:
            errors.append(
                {
                    "field_id": field_id,
                    "code": "header_zone",
                    "message": "Text-like field center lies in header margin heuristic.",
                    "detail": None,
                }
            )

        if ftype in TEXT_LIKE_TYPES:
            bands = geometry.get("writable_bands") or []
            cells = geometry.get("cells") or []
            in_band = any(
                point_in_rect(cx, cy, {"x": b["x_min"], "y": b["y_min"], "w": b["x_max"] - b["x_min"], "h": b["y_max"] - b["y_min"]})
                for b in bands
                if isinstance(b, dict)
            )
            in_cell = any(
                isinstance(c, dict) and isinstance(c.get("bbox"), dict) and point_in_rect(cx, cy, c["bbox"])
                for c in cells
            )
            if bands and not in_band and not in_cell:
                errors.append(
                    {
                        "field_id": field_id,
                        "code": "writable_proximity",
                        "message": "Text-like field center is not inside a writable band or cell.",
                        "detail": None,
                    }
                )

        if ftype in TOGGLE_GROUNDING_TYPES:
            area = bbox["w"] * bbox["h"]
            aspect = max(bbox["w"], bbox["h"]) / max(1, min(bbox["w"], bbox["h"]))
            if area > MAX_TOGGLE_AREA or area < MIN_TOGGLE_AREA or aspect > MAX_TOGGLE_ASPECT:
                errors.append(
                    {
                        "field_id": field_id,
                        "code": "toggle_size",
                        "message": "Checkbox/radio bbox should be small and roughly square.",
                        "detail": {"area": area, "aspect": aspect},
                    }
                )

        supporting = field.get("supporting_lines")
        if isinstance(supporting, list):
            for lid in supporting:
                if not isinstance(lid, str) or lid not in lines_by_id:
                    errors.append(
                        {
                            "field_id": field_id,
                            "code": "unknown_line_id",
                            "message": f"supporting_lines references unknown line_id {lid!r}.",
                            "detail": None,
                        }
                    )

    return errors


def assert_valid_page_grounding(
    payload: dict[str, Any],
    *,
    detected_lines: dict[str, Any],
    page_manifest: dict[str, Any],
    line_padding_px: int = 3,
) -> None:
    errors = validate_page_grounding(
        payload,
        detected_lines=detected_lines,
        page_manifest=page_manifest,
        line_padding_px=line_padding_px,
    )
    if errors:
        raise GroundingValidationError(errors)
