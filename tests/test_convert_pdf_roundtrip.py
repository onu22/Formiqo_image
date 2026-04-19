"""Integration test: render a synthetic PDF and verify mapping."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import fitz
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from convert_pdf_pages_for_grounding import (  # noqa: E402
    convert_pdf_to_images,
    map_image_bbox_to_pdf,
)


@pytest.fixture()
def one_page_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    try:
        page = doc.new_page(width=612, height=792)
        page.draw_rect(fitz.Rect(50, 50, 150, 150), color=(0, 0, 0), fill=(0.1, 0.2, 0.3))
        doc.save(str(pdf_path))
    finally:
        doc.close()
    return pdf_path


def test_convert_pdf_to_images_roundtrip(one_page_pdf: Path, tmp_path: Path) -> None:
    out = tmp_path / "grounding_out"
    result = convert_pdf_to_images(str(one_page_pdf), str(out), dpi=144.0, overwrite=True)

    assert Path(result["document_manifest"]).is_file()
    assert Path(result["images_manifest"]).is_file()
    assert len(result["pages"]) == 1

    page_manifest_path = out / result["pages"][0]["page_manifest_path"]
    manifest = json.loads(page_manifest_path.read_text(encoding="utf-8"))

    iw = manifest["image"]["saved_image_width_px"]
    ih = manifest["image"]["saved_image_height_px"]
    full = map_image_bbox_to_pdf((0.0, 0.0, float(iw), float(ih)), manifest)
    assert math.isclose(full["pdf_x"], 0.0, abs_tol=1e-3)
    assert math.isclose(full["pdf_y"], 0.0, abs_tol=1e-3)
    assert math.isclose(full["pdf_w"], 612.0, rel_tol=1e-4, abs_tol=0.1)
    assert math.isclose(full["pdf_h"], 792.0, rel_tol=1e-4, abs_tol=0.1)

    png_path = out / result["pages"][0]["image_path"]
    assert png_path.is_file()
