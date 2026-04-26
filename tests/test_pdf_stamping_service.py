"""Service tests for PDF stamping."""

from __future__ import annotations

import json
from pathlib import Path

import fitz

from app.services.pdf_stamping import (
    StampPdfStyle,
    _map_bbox_to_pdf_points,
    _pdf_bl_rect_to_pymupdf_rect,
    run_pdf_stamping_for_job,
)

JOB_ID = "90082174-a5ac-466a-8798-3cb0204bffb3"


def _write_pdf_fixture(
    output_dir: Path,
    *,
    provider: str = "openai",
    model: str = "gpt-5.5",
    width: int = 200,
    height: int = 100,
    grounding_width: int | None = None,
    grounding_height: int | None = None,
) -> Path:
    pages_dir = output_dir / "converted_images" / "pages"
    images_dir = output_dir / "converted_images"
    grounding_dir = output_dir / "field_grounding" / f"{provider}_{model}"
    pages_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    grounding_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir.parent / "input.pdf"
    doc = fitz.open()
    doc.new_page(width=width, height=height)
    doc.save(pdf_path)
    doc.close()

    (images_dir / "page_0001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (pages_dir / "page_0001.json").write_text(
        json.dumps(
            {
                "manifest_version": "1.0",
                "page_index": 0,
                "pdf": {"width_pt": float(width), "height_pt": float(height), "origin": "bottom-left"},
                "image": {
                    "path": "converted_images/page_0001.png",
                    "saved_image_width_px": width,
                    "saved_image_height_px": height,
                },
                "mapping": {"formula": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (grounding_dir / "page_0001.fields.json").write_text(
        json.dumps(
            {
                "page_index": 0,
                "width": grounding_width if grounding_width is not None else width,
                "height": grounding_height if grounding_height is not None else height,
                "unit": "px",
                "origin": "top-left",
                "fields": [
                    {
                        "field_id": "first_name",
                        "type": "text",
                        "bbox": {"x": 10, "y": 10, "w": 100, "h": 20},
                    },
                    {
                        "field_id": "last_name",
                        "type": "text",
                        "bbox": {"x": 10, "y": 40, "w": 100, "h": 20},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return pdf_path


def test_run_pdf_stamping_writes_output_and_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "job" / "output"
    pdf_path = _write_pdf_fixture(output_dir)

    result = run_pdf_stamping_for_job(
        job_id=JOB_ID,
        input_pdf=pdf_path,
        output_dir=output_dir,
        provider="openai",
        model="gpt-5.5",
        values={"first_name": "Jane", "last_name": "Doe"},
        style=StampPdfStyle(font_size_pt=12.0),
        require_all_values=False,
    )

    assert result["succeeded_count"] == 1
    assert result["failed_count"] == 0
    page = result["pages"][0]
    assert page["status"] == "succeeded"
    assert page["field_count"] == 2
    assert page["stamped_count"] == 2
    assert page["missing_value_count"] == 0
    assert (output_dir / result["output_pdf"]).is_file()
    assert (output_dir / result["manifest_path"]).is_file()
    with fitz.open(output_dir / result["output_pdf"]) as stamped:
        text = stamped[0].get_text("text")
    assert "Jane" in text
    assert "Doe" in text


def test_mapping_math_is_applied(tmp_path: Path) -> None:
    output_dir = tmp_path / "job" / "output"
    pdf_path = _write_pdf_fixture(output_dir, width=400, height=200)
    page_manifest_path = output_dir / "converted_images" / "pages" / "page_0001.json"
    page_manifest = json.loads(page_manifest_path.read_text(encoding="utf-8"))
    page_manifest["image"]["saved_image_width_px"] = 200
    page_manifest["image"]["saved_image_height_px"] = 100
    page_manifest_path.write_text(json.dumps(page_manifest) + "\n", encoding="utf-8")
    (output_dir / "field_grounding" / "openai_gpt-5.5" / "page_0001.fields.json").write_text(
        json.dumps(
            {
                "page_index": 0,
                "width": 200,
                "height": 100,
                "unit": "px",
                "origin": "top-left",
                "fields": [
                    {
                        "field_id": "first_name",
                        "type": "text",
                        "bbox": {"x": 10, "y": 20, "w": 50, "h": 10},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    pdf_x, pdf_y, pdf_w, pdf_h = _map_bbox_to_pdf_points(
        bbox={"x": 10, "y": 20, "w": 50, "h": 10},
        pdf_w_pt=400.0,
        pdf_h_pt=200.0,
        image_w_px=200,
        image_h_px=100,
    )
    assert pdf_x == 20.0
    assert pdf_y == 140.0
    assert pdf_w == 100.0
    assert pdf_h == 20.0


def test_bottom_left_pdf_rect_converts_to_pymupdf_top_left_rect() -> None:
    pdf_x, pdf_y, pdf_w, pdf_h = _map_bbox_to_pdf_points(
        bbox={"x": 478, "y": 270, "w": 644, "h": 280},
        pdf_w_pt=603.9500122070312,
        pdf_h_pt=781.5800170898438,
        image_w_px=1678,
        image_h_px=2172,
    )

    assert round(pdf_y, 2) == 583.67
    rect = _pdf_bl_rect_to_pymupdf_rect(
        pdf_x=pdf_x,
        pdf_y=pdf_y,
        pdf_w=pdf_w,
        pdf_h=pdf_h,
        pdf_h_pt=781.5800170898438,
    )
    assert round(rect.y0, 2) == 97.16
    assert round(rect.y1, 2) == 197.91


def test_dimension_mismatch_fails_page(tmp_path: Path) -> None:
    output_dir = tmp_path / "job" / "output"
    pdf_path = _write_pdf_fixture(output_dir, grounding_width=9999, grounding_height=9999)

    result = run_pdf_stamping_for_job(
        job_id=JOB_ID,
        input_pdf=pdf_path,
        output_dir=output_dir,
        provider="openai",
        model="gpt-5.5",
        values={"first_name": "Jane", "last_name": "Doe"},
        style=StampPdfStyle(),
        require_all_values=False,
    )

    assert result["succeeded_count"] == 0
    assert result["failed_count"] == 1
    assert "do not match converted image" in result["pages"][0]["error"]


def test_missing_grounding_run_raises(tmp_path: Path) -> None:
    output_dir = tmp_path / "job" / "output"
    pdf_path = output_dir.parent / "input.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    doc.new_page(width=100, height=100)
    doc.save(pdf_path)
    doc.close()
    (output_dir / "converted_images" / "pages").mkdir(parents=True)

    try:
        run_pdf_stamping_for_job(
            job_id=JOB_ID,
            input_pdf=pdf_path,
            output_dir=output_dir,
            provider="openai",
            model="gpt-5.5",
            values={},
            style=StampPdfStyle(),
            require_all_values=False,
        )
    except FileNotFoundError as exc:
        assert "Field grounding run not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
