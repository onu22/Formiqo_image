"""Unit tests for image -> PDF coordinate mapping used by the stamp plan."""

from __future__ import annotations

import math

import pytest

from app.services.stamping import image_bbox_to_pdf_rects


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def test_uniform_scale_top_left_corner() -> None:
    bbox = {"x": 0, "y": 0, "w": 100, "h": 100}
    bl, tl = image_bbox_to_pdf_rects(
        bbox,
        pdf_width_pt=612.0,
        pdf_height_pt=792.0,
        image_width_px=1700,
        image_height_px=2200,
    )
    sx = 612.0 / 1700
    sy = 792.0 / 2200

    assert _close(bl["pdf_x"], 0.0)
    assert _close(bl["pdf_y"], 792.0 - 100 * sy)
    assert _close(bl["pdf_w"], 100 * sx)
    assert _close(bl["pdf_h"], 100 * sy)

    assert _close(tl["x0"], 0.0)
    assert _close(tl["y0"], 0.0)
    assert _close(tl["x1"], 100 * sx)
    assert _close(tl["y1"], 100 * sy)


def test_full_page_bbox_maps_to_full_page() -> None:
    bbox = {"x": 0, "y": 0, "w": 1700, "h": 2200}
    bl, tl = image_bbox_to_pdf_rects(
        bbox,
        pdf_width_pt=612.0,
        pdf_height_pt=792.0,
        image_width_px=1700,
        image_height_px=2200,
    )
    assert _close(bl["pdf_x"], 0.0)
    assert _close(bl["pdf_y"], 0.0)
    assert _close(bl["pdf_w"], 612.0)
    assert _close(bl["pdf_h"], 792.0)

    assert _close(tl["x0"], 0.0)
    assert _close(tl["y0"], 0.0)
    assert _close(tl["x1"], 612.0)
    assert _close(tl["y1"], 792.0)


def test_known_field_mapping_matches_expected_example() -> None:
    """Mirrors the real page_0001 mapping for a 1700x2200 -> 612x792 page."""
    bbox = {"x": 395, "y": 269, "w": 365, "h": 36}
    bl, tl = image_bbox_to_pdf_rects(
        bbox,
        pdf_width_pt=612.0,
        pdf_height_pt=792.0,
        image_width_px=1700,
        image_height_px=2200,
    )
    sx = 612.0 / 1700
    sy = 792.0 / 2200

    assert _close(bl["pdf_x"], 395 * sx)
    assert _close(bl["pdf_y"], 792.0 - (269 + 36) * sy)
    assert _close(bl["pdf_w"], 365 * sx)
    assert _close(bl["pdf_h"], 36 * sy)

    assert _close(tl["x0"], 395 * sx)
    assert _close(tl["y0"], 269 * sy)
    assert _close(tl["x1"], (395 + 365) * sx)
    assert _close(tl["y1"], (269 + 36) * sy)


def test_tl_and_bl_consistency_y_axis() -> None:
    bbox = {"x": 10, "y": 20, "w": 30, "h": 40}
    bl, tl = image_bbox_to_pdf_rects(
        bbox,
        pdf_width_pt=500.0,
        pdf_height_pt=700.0,
        image_width_px=1000,
        image_height_px=1400,
    )
    assert _close(tl["y0"] + bl["pdf_h"], tl["y1"])
    assert _close(bl["pdf_y"], 700.0 - tl["y1"])


def test_rejects_bbox_outside_image() -> None:
    with pytest.raises(ValueError):
        image_bbox_to_pdf_rects(
            {"x": 900, "y": 10, "w": 200, "h": 10},
            pdf_width_pt=612.0,
            pdf_height_pt=792.0,
            image_width_px=1000,
            image_height_px=1400,
        )


def test_rejects_non_positive_dims() -> None:
    with pytest.raises(ValueError):
        image_bbox_to_pdf_rects(
            {"x": 0, "y": 0, "w": 10, "h": 10},
            pdf_width_pt=0.0,
            pdf_height_pt=100.0,
            image_width_px=100,
            image_height_px=100,
        )
    with pytest.raises(ValueError):
        image_bbox_to_pdf_rects(
            {"x": 0, "y": 0, "w": 10, "h": 10},
            pdf_width_pt=100.0,
            pdf_height_pt=100.0,
            image_width_px=0,
            image_height_px=100,
        )


def test_rejects_non_positive_wh() -> None:
    with pytest.raises(ValueError):
        image_bbox_to_pdf_rects(
            {"x": 0, "y": 0, "w": 0, "h": 10},
            pdf_width_pt=100.0,
            pdf_height_pt=100.0,
            image_width_px=100,
            image_height_px=100,
        )
