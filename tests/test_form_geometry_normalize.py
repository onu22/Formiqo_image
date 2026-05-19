"""Tests for geometry normalization helpers."""

from __future__ import annotations

from app.services.form_geometry import (
    bbox_subset_of,
    build_geometry_index,
    find_containing_cell,
    inset_bbox,
    normalize_page_grounding,
)


def _single_cell_detected_lines() -> dict:
    return {
        "image": {"width": 500, "height": 300},
        "lines": [
            {
                "line_id": "line_h_001",
                "orientation": "horizontal",
                "bbox": {"x": 0, "y": 100, "w": 500, "h": 2},
            },
            {
                "line_id": "line_h_002",
                "orientation": "horizontal",
                "bbox": {"x": 0, "y": 200, "w": 500, "h": 2},
            },
            {
                "line_id": "line_v_001",
                "orientation": "vertical",
                "bbox": {"x": 100, "y": 0, "w": 2, "h": 300},
            },
            {
                "line_id": "line_v_002",
                "orientation": "vertical",
                "bbox": {"x": 400, "y": 0, "w": 2, "h": 300},
            },
        ],
    }


def test_find_containing_cell_smallest_wins() -> None:
    geometry = build_geometry_index(_single_cell_detected_lines())
    cells = geometry["cells"]
    assert len(cells) == 1
    cell = find_containing_cell(250, 150, cells)
    assert cell is not None
    assert cell["cell_index"] == 0


def test_normalize_snaps_overlapping_bbox_inside_cell() -> None:
    detected = _single_cell_detected_lines()
    geometry = build_geometry_index(detected)
    payload = {
        "page_index": 0,
        "width": 500,
        "height": 300,
        "unit": "px",
        "origin": "top-left",
        "fields": [
            {
                "field_id": "first_name",
                "type": "text",
                "field_surface": "solid_box",
                "bbox": {"x": 150, "y": 100, "w": 200, "h": 50},
            }
        ],
    }
    out = normalize_page_grounding(
        payload,
        geometry,
        stamp_inset_px=2,
        page_w=500,
        page_h=300,
    )
    field = out["fields"][0]
    snapped = field["bbox"]
    cell_bbox = geometry["cells"][0]["bbox"]
    assert bbox_subset_of(snapped, cell_bbox, tolerance_px=1)
    assert field.get("supporting_lines")
    assert "line_h_001" in field["supporting_lines"]


def test_inset_bbox_clamps_to_page() -> None:
    bbox = {"x": 0, "y": 0, "w": 100, "h": 100}
    shrunk = inset_bbox(bbox, 10, page_w=500, page_h=300)
    assert shrunk["w"] >= 1 and shrunk["h"] >= 1
    assert shrunk["x"] + shrunk["w"] <= 500
    assert shrunk["y"] + shrunk["h"] <= 300


def test_aggressive_normalize_for_unknown_surface() -> None:
    detected = _single_cell_detected_lines()
    geometry = build_geometry_index(detected)
    payload = {
        "page_index": 0,
        "width": 500,
        "height": 300,
        "unit": "px",
        "origin": "top-left",
        "fields": [
            {
                "field_id": "notes",
                "type": "text",
                "field_surface": "unknown",
                "bbox": {"x": 0, "y": 0, "w": 400, "h": 250},
            }
        ],
    }
    out = normalize_page_grounding(
        payload,
        geometry,
        stamp_inset_px=2,
        page_w=500,
        page_h=300,
        aggressive=True,
    )
    snapped = out["fields"][0]["bbox"]
    cell_bbox = geometry["cells"][0]["bbox"]
    assert bbox_subset_of(snapped, cell_bbox, tolerance_px=1)
