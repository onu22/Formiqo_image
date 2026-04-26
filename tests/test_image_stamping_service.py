"""Service tests for PNG image stamping."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageChops

from app.services.image_stamping import StampImageStyle, run_image_stamping_for_job

JOB_ID = "90082174-a5ac-466a-8798-3cb0204bffb3"


def _write_stamp_fixture(
    output_dir: Path,
    *,
    provider: str = "openai",
    model: str = "gpt-4o",
    width: int = 200,
    height: int = 100,
    grounding_width: int | None = None,
    grounding_height: int | None = None,
) -> None:
    pages_dir = output_dir / "converted_images" / "pages"
    images_dir = output_dir / "converted_images"
    grounding_dir = output_dir / "field_grounding" / f"{provider}_{model}"
    pages_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    grounding_dir.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (width, height), "white").save(images_dir / "page_0001.png")
    (pages_dir / "page_0001.json").write_text(
        json.dumps(
            {
                "manifest_version": "1.0",
                "page_index": 0,
                "pdf": {"width_pt": 200.0, "height_pt": 100.0, "origin": "bottom-left"},
                "image": {
                    "path": "converted_images/page_0001.png",
                    "format": "png",
                    "width_px": width,
                    "height_px": height,
                    "origin": "top-left",
                    "rendered_image_width_px": width,
                    "rendered_image_height_px": height,
                    "saved_image_width_px": width,
                    "saved_image_height_px": height,
                },
                "rendering": {"dpi": 72.0, "zoom": 1.0, "library": "pymupdf"},
                "mapping": {
                    "image_to_pdf_scale_x": 1.0,
                    "image_to_pdf_scale_y": 1.0,
                    "formula": {},
                },
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
                        "bbox": {"x": 10, "y": 10, "w": 110, "h": 24},
                    },
                    {
                        "field_id": "last_name",
                        "type": "text",
                        "bbox": {"x": 10, "y": 45, "w": 110, "h": 24},
                    },
                ],
                "provider": provider,
                "model": model,
                "run_id": "run_test",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _images_differ(a: Path, b: Path) -> bool:
    with Image.open(a) as left, Image.open(b) as right:
        diff = ImageChops.difference(left.convert("RGB"), right.convert("RGB"))
        return diff.getbbox() is not None


def test_run_image_stamping_writes_output_and_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write_stamp_fixture(output_dir)

    result = run_image_stamping_for_job(
        job_id=JOB_ID,
        output_dir=output_dir,
        provider="openai",
        model="gpt-4o",
        values={"first_name": "Jane", "last_name": "Doe"},
        style=StampImageStyle(font_size_px=18),
        require_all_values=False,
    )

    assert result["succeeded_count"] == 1
    assert result["failed_count"] == 0
    page = result["pages"][0]
    assert page["status"] == "succeeded"
    assert page["field_count"] == 2
    assert page["stamped_count"] == 2
    assert page["missing_value_count"] == 0

    out_image = output_dir / page["output_image"]
    assert out_image.is_file()
    assert (output_dir / result["manifest_path"]).is_file()
    assert _images_differ(output_dir / "converted_images" / "page_0001.png", out_image)


def test_missing_values_are_skipped_by_default(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write_stamp_fixture(output_dir)

    result = run_image_stamping_for_job(
        job_id=JOB_ID,
        output_dir=output_dir,
        provider="openai",
        model="gpt-4o",
        values={"first_name": "Jane"},
        style=StampImageStyle(font_size_px=18),
        require_all_values=False,
    )

    page = result["pages"][0]
    assert page["status"] == "succeeded"
    assert page["stamped_count"] == 1
    assert page["missing_value_count"] == 1


def test_require_all_values_fails_page(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write_stamp_fixture(output_dir)

    result = run_image_stamping_for_job(
        job_id=JOB_ID,
        output_dir=output_dir,
        provider="openai",
        model="gpt-4o",
        values={"first_name": "Jane"},
        style=StampImageStyle(),
        require_all_values=True,
    )

    assert result["succeeded_count"] == 0
    assert result["failed_count"] == 1
    assert "Missing values" in result["pages"][0]["error"]


def test_dimension_mismatch_fails_page(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write_stamp_fixture(output_dir, grounding_width=999, grounding_height=100)

    result = run_image_stamping_for_job(
        job_id=JOB_ID,
        output_dir=output_dir,
        provider="openai",
        model="gpt-4o",
        values={"first_name": "Jane", "last_name": "Doe"},
        style=StampImageStyle(),
        require_all_values=False,
    )

    assert result["succeeded_count"] == 0
    assert result["failed_count"] == 1
    assert "do not match converted image" in result["pages"][0]["error"]


def test_missing_grounding_run_raises(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    (output_dir / "converted_images" / "pages").mkdir(parents=True)

    try:
        run_image_stamping_for_job(
            job_id=JOB_ID,
            output_dir=output_dir,
            provider="openai",
            model="gpt-4o",
            values={},
            style=StampImageStyle(),
            require_all_values=False,
        )
    except FileNotFoundError as exc:
        assert "Field grounding run not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
