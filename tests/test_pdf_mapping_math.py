"""Unit tests for scale helpers and bbox mapping (no PDF I/O)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from convert_pdf_pages_for_grounding import (  # noqa: E402
    map_image_bbox_to_pdf,
    scales_from_dimensions,
)


def test_scales_from_dimensions_basic() -> None:
    sx, sy = scales_from_dimensions(612.0, 792.0, 612, 792)
    assert sx == 1.0 and sy == 1.0


def test_scales_from_dimensions_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        scales_from_dimensions(0, 100, 10, 10)
    with pytest.raises(ValueError):
        scales_from_dimensions(100, 100, 0, 10)


def test_map_image_bbox_to_pdf_full_page() -> None:
    manifest = {
        "pdf": {"width_pt": 612.0, "height_pt": 792.0, "origin": "bottom-left"},
        "image": {
            "saved_image_width_px": 612,
            "saved_image_height_px": 792,
            "origin": "top-left",
        },
        "mapping": {
            "image_to_pdf_scale_x": 612.0 / 612,
            "image_to_pdf_scale_y": 792.0 / 792,
            "formula": {},
        },
    }
    r = map_image_bbox_to_pdf((0.0, 0.0, 612.0, 792.0), manifest)
    assert math.isclose(r["pdf_x"], 0.0)
    assert math.isclose(r["pdf_y"], 0.0)
    assert math.isclose(r["pdf_w"], 612.0)
    assert math.isclose(r["pdf_h"], 792.0)


def test_map_image_bbox_to_pdf_corner_strip() -> None:
    """Top-left 1x1 px at image origin should map to a thin strip at the PDF top."""
    manifest = {
        "pdf": {"width_pt": 100.0, "height_pt": 200.0, "origin": "bottom-left"},
        "image": {
            "saved_image_width_px": 1000,
            "saved_image_height_px": 2000,
            "origin": "top-left",
        },
        "mapping": {
            "image_to_pdf_scale_x": 0.1,
            "image_to_pdf_scale_y": 0.1,
            "formula": {},
        },
    }
    r = map_image_bbox_to_pdf((0.0, 0.0, 1.0, 1.0), manifest)
    assert math.isclose(r["pdf_x"], 0.0)
    assert math.isclose(r["pdf_w"], 0.1)
    assert math.isclose(r["pdf_h"], 0.1)
    # Lower-left y of a 1px-high strip from the top row: pdf_y = 200 - 0.1 = 199.9
    assert math.isclose(r["pdf_y"], 199.9)


def test_map_image_bbox_mismatch_raises() -> None:
    manifest = {
        "pdf": {"width_pt": 100.0, "height_pt": 100.0, "origin": "bottom-left"},
        "image": {
            "saved_image_width_px": 10,
            "saved_image_height_px": 10,
            "origin": "top-left",
        },
        "mapping": {
            "image_to_pdf_scale_x": 9.0,  # wrong on purpose
            "image_to_pdf_scale_y": 10.0,
            "formula": {},
        },
    }
    with pytest.raises(ValueError):
        map_image_bbox_to_pdf((0.0, 0.0, 1.0, 1.0), manifest)
