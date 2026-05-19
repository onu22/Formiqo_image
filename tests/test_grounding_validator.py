"""Tests for surface-aware grounding validation."""

from __future__ import annotations

from app.services.form_geometry import build_geometry_index, normalize_page_grounding
from app.services.grounding_validator import validate_page_grounding

from tests.test_form_geometry_normalize import _single_cell_detected_lines


def _page_manifest() -> dict:
    return {
        "page_index": 0,
        "image": {"saved_image_width_px": 500, "saved_image_height_px": 300},
    }


def _base_payload(field: dict) -> dict:
    return {
        "page_index": 0,
        "width": 500,
        "height": 300,
        "unit": "px",
        "origin": "top-left",
        "fields": [field],
    }


def test_normalized_solid_box_passes_crosses_line() -> None:
    detected = _single_cell_detected_lines()
    geometry = build_geometry_index(detected)
    payload = _base_payload(
        {
            "field_id": "first_name",
            "type": "text",
            "field_surface": "solid_box",
            "bbox": {"x": 150, "y": 100, "w": 200, "h": 50},
        }
    )
    normalized = normalize_page_grounding(
        payload,
        geometry,
        stamp_inset_px=2,
        page_w=500,
        page_h=300,
    )
    errors = validate_page_grounding(
        normalized,
        detected_lines=detected,
        page_manifest=_page_manifest(),
        geometry=geometry,
    )
    codes = {e["code"] for e in errors}
    assert "crosses_line" not in codes


def test_unnormalized_bbox_extending_past_cell_fails_outside_cell() -> None:
    detected = _single_cell_detected_lines()
    geometry = build_geometry_index(detected)
    payload = _base_payload(
        {
            "field_id": "first_name",
            "type": "text",
            "field_surface": "solid_box",
            "bbox": {"x": 150, "y": 100, "w": 200, "h": 50},
            "supporting_lines": [],
        }
    )
    errors = validate_page_grounding(
        payload,
        detected_lines=detected,
        page_manifest=_page_manifest(),
        geometry=geometry,
    )
    assert any(e["code"] == "outside_cell" for e in errors)


def test_open_area_crosses_unrelated_line() -> None:
    detected = _single_cell_detected_lines()
    geometry = build_geometry_index(detected)
    payload = _base_payload(
        {
            "field_id": "free_text",
            "type": "text",
            "field_surface": "open_area",
            "bbox": {"x": 50, "y": 95, "w": 400, "h": 120},
        }
    )
    errors = validate_page_grounding(
        payload,
        detected_lines=detected,
        page_manifest=_page_manifest(),
        geometry=geometry,
    )
    assert any(e["code"] == "crosses_line" for e in errors)


def test_bbox_spilling_into_neighbor_fails_outside_cell() -> None:
    detected = {
        "image": {"width": 600, "height": 300},
        "lines": [
            {"line_id": "line_h_001", "orientation": "horizontal", "bbox": {"x": 0, "y": 100, "w": 600, "h": 2}},
            {"line_id": "line_h_002", "orientation": "horizontal", "bbox": {"x": 0, "y": 200, "w": 600, "h": 2}},
            {"line_id": "line_v_001", "orientation": "vertical", "bbox": {"x": 100, "y": 0, "w": 2, "h": 300}},
            {"line_id": "line_v_002", "orientation": "vertical", "bbox": {"x": 300, "y": 0, "w": 2, "h": 300}},
            {"line_id": "line_v_003", "orientation": "vertical", "bbox": {"x": 500, "y": 0, "w": 2, "h": 300}},
        ],
    }
    geometry = build_geometry_index(detected)
    # Center in left cell but bbox extends into right cell.
    payload = _base_payload(
        {
            "field_id": "left_only",
            "type": "text",
            "field_surface": "solid_box",
            "bbox": {"x": 110, "y": 110, "w": 250, "h": 80},
            "supporting_lines": ["line_h_001", "line_h_002", "line_v_001", "line_v_002"],
        }
    )
    errors = validate_page_grounding(
        payload,
        detected_lines=detected,
        page_manifest={"page_index": 0, "image": {"saved_image_width_px": 600, "saved_image_height_px": 300}},
        geometry=geometry,
    )
    assert any(e["code"] == "outside_cell" for e in errors)


def test_underline_allows_anchor_horizontal() -> None:
    detected = _single_cell_detected_lines()
    geometry = build_geometry_index(detected)
    band = geometry["writable_bands"][0]
    payload = _base_payload(
        {
            "field_id": "signature",
            "type": "signature",
            "field_surface": "signature_line",
            "bbox": {
                "x": band["x_min"] + 10,
                "y": band["y_min"] + 5,
                "w": 200,
                "h": band["y_max"] - band["y_min"] - 10,
            },
            "supporting_lines": ["line_h_001"],
        }
    )
    errors = validate_page_grounding(
        payload,
        detected_lines=detected,
        page_manifest=_page_manifest(),
        geometry=geometry,
    )
    assert not any(e["code"] == "crosses_line" and e["detail"]["line_id"] == "line_h_001" for e in errors)
