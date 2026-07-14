"""E2 golden parity tests: stamped PNG preview vs rasterized stamped PDF page (Gate G2).

Validates the PRD E2 acceptance criteria:
  - no duplicated validation/discovery/manifest logic between the two stampers
    (both call into ``stamping_common``; exercised implicitly by every test here)
  - preview PNG and rasterized stamped PDF match within a small pixel tolerance
  - identical wrapping/truncation behavior for ``multiline_text``
  - a per-field ``font_size_pt`` override set in ``stamping.json`` round-trips through
    both stamper paths
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import fitz
import numpy as np
import pytest
from PIL import Image

from app.services.image_stamping import run_image_stamping_for_job
from app.services.pdf_stamping import run_pdf_stamping_for_job
from app.services.stamping_config import load_stamping_json_parsed, stamping_overrides, stamping_style
from tests.fixtures.parity.builder import DPI, build_parity_job

PIXEL_TOLERANCE = 8


def _rasterize_pdf_page(pdf_path: Path, *, dpi: float = DPI) -> Image.Image:
    doc = fitz.open(pdf_path)
    try:
        pix = doc[0].get_pixmap(dpi=int(dpi))
        mode = "RGBA" if pix.alpha else "RGB"
        return Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("RGB")
    finally:
        doc.close()


def _ink_bbox(image: Image.Image, *, bbox: dict[str, float], margin: int = 6) -> tuple[int, int, int, int] | None:
    """Bounding box (x0, y0, x1, y1) of non-white pixels within *bbox* (+margin), or None."""
    x0 = max(0, int(bbox["x"]) - margin)
    y0 = max(0, int(bbox["y"]) - margin)
    x1 = min(image.width, int(bbox["x"] + bbox["w"]) + margin)
    y1 = min(image.height, int(bbox["y"] + bbox["h"]) + margin)
    crop = np.asarray(image.crop((x0, y0, x1, y1)).convert("L"))
    dark = crop < 200
    if not dark.any():
        return None
    ys, xs = np.where(dark)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _line_band_count(image: Image.Image, *, bbox: dict[str, float], margin: int = 4) -> int:
    """Count distinct horizontal ink bands (rendered text lines) within *bbox*."""
    x0 = max(0, int(bbox["x"]) - margin)
    y0 = max(0, int(bbox["y"]) - margin)
    x1 = min(image.width, int(bbox["x"] + bbox["w"]) + margin)
    y1 = min(image.height, int(bbox["y"] + bbox["h"]) + margin)
    crop = np.asarray(image.crop((x0, y0, x1, y1)).convert("L"))
    dark_rows = (crop < 200).any(axis=1)
    bands = 0
    in_band = False
    for is_dark in dark_rows:
        if is_dark and not in_band:
            bands += 1
            in_band = True
        elif not is_dark:
            in_band = False
    return bands


def _fields_by_id(output_dir: Path) -> dict[str, dict]:
    payload = json.loads((output_dir / "field_grounding" / "page_0001.fields.json").read_text(encoding="utf-8"))
    return {f["field_id"]: f for f in payload["fields"]}


def _stamp_both(form) -> tuple[dict, dict]:
    stamping = load_stamping_json_parsed(form.output_dir)
    style = stamping_style(stamping)
    overrides = stamping_overrides(stamping)

    image_result = run_image_stamping_for_job(
        job_id=form.job_id,
        output_dir=form.output_dir,
        provider="openai",
        model="gpt-parity-test",
        values=stamping.values,
        style=style,
        overrides=overrides,
        require_all_values=False,
    )
    pdf_result = run_pdf_stamping_for_job(
        job_id=form.job_id,
        input_pdf=form.input_pdf,
        output_dir=form.output_dir,
        provider="openai",
        model="gpt-parity-test",
        values=stamping.values,
        style=style,
        overrides=overrides,
        require_all_values=False,
    )
    return image_result, pdf_result


@pytest.mark.parametrize("variant", ["form_a", "form_b"])
def test_preview_pdf_parity(tmp_path: Path, variant: str) -> None:
    job_id = str(uuid.uuid4())
    form = build_parity_job(tmp_path, variant=variant, job_id=job_id)

    image_result, pdf_result = _stamp_both(form)
    assert image_result["succeeded_count"] == 1
    assert pdf_result["succeeded_count"] == 1

    preview = Image.open(form.output_dir / image_result["pages"][0]["output_image"]).convert("RGB")
    rasterized = _rasterize_pdf_page(form.output_dir / pdf_result["output_pdf"])
    assert preview.size == rasterized.size

    fields = _fields_by_id(form.output_dir)

    for kind in ("text", "multiline", "checkbox"):
        field_id = form.field_ids[kind]
        bbox = fields[field_id]["bbox"]

        preview_box = _ink_bbox(preview, bbox=bbox)
        pdf_box = _ink_bbox(rasterized, bbox=bbox)
        assert preview_box is not None, f"{kind} field did not render in preview"
        assert pdf_box is not None, f"{kind} field did not render in PDF"

        for a, b in zip(preview_box, pdf_box):
            assert abs(a - b) <= PIXEL_TOLERANCE, (
                f"{variant}/{kind} ink bbox mismatch: preview={preview_box} pdf={pdf_box}"
            )

        if kind == "multiline":
            preview_lines = _line_band_count(preview, bbox=bbox)
            pdf_lines = _line_band_count(rasterized, bbox=bbox)
            assert preview_lines == pdf_lines, (
                f"{variant} multiline wrap mismatch: preview_lines={preview_lines} pdf_lines={pdf_lines}"
            )
            assert preview_lines >= 2  # confirms word-wrapping actually happened


def test_multiline_overflow_truncates_identically(tmp_path: Path) -> None:
    """A field too small for its text shrinks then truncates identically on both paths."""
    job_id = str(uuid.uuid4())
    form = build_parity_job(tmp_path, variant="form_a", job_id=job_id)

    # Shrink the multiline bbox so the long fixture value must overflow even at min font size.
    fg_dir = form.output_dir / "field_grounding"
    fields_path = fg_dir / "page_0001.fields.json"
    payload = json.loads(fields_path.read_text(encoding="utf-8"))
    for field in payload["fields"]:
        if field["field_id"] == form.field_ids["multiline"]:
            field["bbox"] = {"x": 200, "y": 400, "w": 400, "h": 24}
    fields_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    image_result, pdf_result = _stamp_both(form)
    assert image_result["succeeded_count"] == 1
    assert pdf_result["succeeded_count"] == 1

    preview = Image.open(form.output_dir / image_result["pages"][0]["output_image"]).convert("RGB")
    rasterized = _rasterize_pdf_page(form.output_dir / pdf_result["output_pdf"])

    bbox = _fields_by_id(form.output_dir)[form.field_ids["multiline"]]["bbox"]
    preview_lines = _line_band_count(preview, bbox=bbox)
    pdf_lines = _line_band_count(rasterized, bbox=bbox)
    assert preview_lines == pdf_lines
    assert preview_lines == 1  # the tiny box only fits a single truncated line


def test_font_size_override_round_trips_from_stamping_json(tmp_path: Path) -> None:
    """A per-field ``font_size_pt`` override set in stamping.json wins on both paths."""
    job_id = str(uuid.uuid4())
    form = build_parity_job(tmp_path, variant="form_a", job_id=job_id, font_size_pt=9.0)

    stamping = load_stamping_json_parsed(form.output_dir)
    overrides = stamping_overrides(stamping)
    text_field_id = form.field_ids["text"]
    assert text_field_id in overrides
    assert overrides[text_field_id]["font_size_pt"] == pytest.approx(12.0)
    assert overrides[text_field_id]["font_size_pt"] != stamping.style.font_size_pt

    image_result, pdf_result = _stamp_both(form)

    preview = Image.open(form.output_dir / image_result["pages"][0]["output_image"]).convert("RGB")
    rasterized = _rasterize_pdf_page(form.output_dir / pdf_result["output_pdf"])

    bbox = _fields_by_id(form.output_dir)[text_field_id]["bbox"]
    preview_box = _ink_bbox(preview, bbox=bbox)
    pdf_box = _ink_bbox(rasterized, bbox=bbox)
    assert preview_box is not None and pdf_box is not None

    # Glyph-band height scales with the overridden font size; both paths must agree closely.
    preview_h = preview_box[3] - preview_box[1]
    pdf_h = pdf_box[3] - pdf_box[1]
    assert abs(preview_h - pdf_h) <= PIXEL_TOLERANCE
