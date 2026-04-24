"""Service-level tests for the stamp plan generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.stamping import (
    STAMP_PLAN_SCHEMA_VERSION,
    build_stamp_plan_for_page,
    run_prepare_stamp_for_job,
)

JOB_ID = "90082174-a5ac-466a-8798-3cb0204bffb3"


def _conv_manifest(
    *,
    page_index: int = 0,
    pdf_w: float = 612.0,
    pdf_h: float = 792.0,
    img_w: int = 1700,
    img_h: int = 2200,
) -> dict:
    sx = pdf_w / img_w
    sy = pdf_h / img_h
    return {
        "manifest_version": "1.0",
        "page_index": page_index,
        "pdf": {"width_pt": pdf_w, "height_pt": pdf_h, "origin": "bottom-left"},
        "image": {
            "path": f"converted_images/page_{page_index + 1:04d}.png",
            "format": "png",
            "width_px": img_w,
            "height_px": img_h,
            "origin": "top-left",
            "rendered_image_width_px": img_w,
            "rendered_image_height_px": img_h,
            "saved_image_width_px": img_w,
            "saved_image_height_px": img_h,
        },
        "rendering": {"dpi": 200.0, "zoom": 2.7777777777777777, "library": "pymupdf"},
        "mapping": {
            "image_to_pdf_scale_x": sx,
            "image_to_pdf_scale_y": sy,
            "formula": {
                "pdf_x": "x * image_to_pdf_scale_x",
                "pdf_y": "pdf_height_pt - ((y + h) * image_to_pdf_scale_y)",
                "pdf_w": "w * image_to_pdf_scale_x",
                "pdf_h": "h * image_to_pdf_scale_y",
            },
        },
    }


def _grounding_json(
    *,
    page_index: int = 0,
    img_w: int = 1700,
    img_h: int = 2200,
    provider: str = "anthropic",
    model: str = "claude-opus-4-7",
    run_id: str = "run_test_abc",
) -> dict:
    return {
        "page_index": page_index,
        "width": img_w,
        "height": img_h,
        "unit": "px",
        "origin": "top-left",
        "fields": [
            {"field_id": "first_name", "type": "text", "bbox": {"x": 395, "y": 269, "w": 365, "h": 36}},
            {"field_id": "last_name", "type": "text", "bbox": {"x": 1096, "y": 884, "w": 390, "h": 36}},
        ],
        "provider": provider,
        "model": model,
        "run_id": run_id,
    }


def test_build_stamp_plan_for_page_shape_and_dual_coords() -> None:
    conv = _conv_manifest()
    g = _grounding_json()
    plan = build_stamp_plan_for_page(
        job_id=JOB_ID,
        source_pdf="/abs/input.pdf",
        page_index=0,
        conversion_page_manifest=conv,
        grounding_page_json=g,
        provider="anthropic",
        model="claude-opus-4-7",
        run_id="run_test_abc",
    )
    assert plan["schema_version"] == STAMP_PLAN_SCHEMA_VERSION
    assert plan["job_id"] == JOB_ID
    assert plan["source_pdf"] == "/abs/input.pdf"
    assert plan["page_index"] == 0
    assert plan["unit"] == "pt"
    assert plan["pdf"] == {"width_pt": 612.0, "height_pt": 792.0, "origin": "bottom-left"}
    assert plan["image_reference"]["path"] == "converted_images/page_0001.png"
    assert plan["image_reference"]["width_px"] == 1700
    assert plan["image_reference"]["height_px"] == 2200
    assert plan["grounding"] == {
        "provider": "anthropic",
        "model": "claude-opus-4-7",
        "run_id": "run_test_abc",
    }

    assert len(plan["fields"]) == 2
    f0 = plan["fields"][0]
    assert f0["field_id"] == "first_name"
    assert f0["type"] == "text"
    assert f0["image_bbox"] == {"x": 395, "y": 269, "w": 365, "h": 36}
    for key in ("pdf_x", "pdf_y", "pdf_w", "pdf_h"):
        assert key in f0["pdf_bbox_bl"]
    for key in ("x0", "y0", "x1", "y1"):
        assert key in f0["pdf_rect_tl"]


def test_build_stamp_plan_rejects_dim_mismatch() -> None:
    conv = _conv_manifest(img_w=1700, img_h=2200)
    g = _grounding_json(img_w=1600, img_h=2200)
    with pytest.raises(ValueError, match="Grounding image dimensions"):
        build_stamp_plan_for_page(
            job_id=JOB_ID,
            source_pdf="/abs/input.pdf",
            page_index=0,
            conversion_page_manifest=conv,
            grounding_page_json=g,
            provider="openai",
            model="gpt-4o",
            run_id="x",
        )


def test_build_stamp_plan_rejects_page_index_mismatch() -> None:
    conv = _conv_manifest(page_index=0)
    g = _grounding_json(page_index=3)
    with pytest.raises(ValueError, match="page_index"):
        build_stamp_plan_for_page(
            job_id=JOB_ID,
            source_pdf="/abs/input.pdf",
            page_index=0,
            conversion_page_manifest=conv,
            grounding_page_json=g,
            provider="openai",
            model="gpt-4o",
            run_id="x",
        )


def _write_job_fixture(
    output_dir: Path,
    *,
    provider: str = "anthropic",
    model: str = "claude-opus-4-7",
    page_count: int = 1,
    img_w: int = 1700,
    img_h: int = 2200,
) -> Path:
    pages_dir = output_dir / "converted_images" / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    run_dir_name = f"{provider}_{model}"
    grounding_dir = output_dir / "field_grounding" / run_dir_name
    grounding_dir.mkdir(parents=True, exist_ok=True)

    for i in range(page_count):
        (pages_dir / f"page_{i + 1:04d}.json").write_text(
            json.dumps(_conv_manifest(page_index=i, img_w=img_w, img_h=img_h)) + "\n",
            encoding="utf-8",
        )
        (grounding_dir / f"page_{i + 1:04d}.fields.json").write_text(
            json.dumps(_grounding_json(page_index=i, img_w=img_w, img_h=img_h, provider=provider, model=model)) + "\n",
            encoding="utf-8",
        )

    (grounding_dir / "manifest.json").write_text(
        json.dumps(
            {
                "job_id": JOB_ID,
                "provider": provider,
                "model": model,
                "run_id": "run_manifest_xyz",
                "run_dir": f"field_grounding/{run_dir_name}",
                "created_at": "2026-01-01T00:00:00Z",
                "page_count": page_count,
                "output_dir": f"field_grounding/{run_dir_name}",
                "files": [f"field_grounding/{run_dir_name}/page_{i + 1:04d}.fields.json" for i in range(page_count)],
                "succeeded_count": page_count,
                "failed_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return grounding_dir


def test_run_prepare_stamp_writes_per_page_and_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_job_fixture(output_dir, page_count=2)

    result = run_prepare_stamp_for_job(
        job_id=JOB_ID,
        output_dir=output_dir,
        source_pdf=str(tmp_path / "input.pdf"),
        provider="anthropic",
        model="claude-opus-4-7",
    )
    assert result["provider"] == "anthropic"
    assert result["model"] == "claude-opus-4-7"
    assert result["run_dir"] == "stamp_plans/anthropic_claude-opus-4-7"
    assert result["page_count"] == 2
    assert result["succeeded_count"] == 2
    assert result["failed_count"] == 0

    stamp_dir = output_dir / "stamp_plans" / "anthropic_claude-opus-4-7"
    assert (stamp_dir / "manifest.json").is_file()
    assert (stamp_dir / "page_0001.stamp.json").is_file()
    assert (stamp_dir / "page_0002.stamp.json").is_file()

    page1 = json.loads((stamp_dir / "page_0001.stamp.json").read_text(encoding="utf-8"))
    assert page1["schema_version"] == STAMP_PLAN_SCHEMA_VERSION
    assert page1["job_id"] == JOB_ID
    assert page1["grounding"]["provider"] == "anthropic"
    assert page1["grounding"]["model"] == "claude-opus-4-7"
    assert page1["grounding"]["run_id"] == "run_test_abc"
    assert len(page1["fields"]) == 2
    assert "pdf_bbox_bl" in page1["fields"][0]
    assert "pdf_rect_tl" in page1["fields"][0]

    manifest = json.loads((stamp_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider"] == "anthropic"
    assert manifest["model"] == "claude-opus-4-7"
    assert manifest["page_count"] == 2
    assert len(manifest["files"]) == 2
    assert manifest["files"][0] == "stamp_plans/anthropic_claude-opus-4-7/page_0001.stamp.json"


def test_run_prepare_stamp_overwrites_in_place(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_job_fixture(output_dir, page_count=1)

    first = run_prepare_stamp_for_job(
        job_id=JOB_ID,
        output_dir=output_dir,
        source_pdf=str(tmp_path / "input.pdf"),
        provider="anthropic",
        model="claude-opus-4-7",
    )
    stamp_file = output_dir / "stamp_plans" / "anthropic_claude-opus-4-7" / "page_0001.stamp.json"
    first_created = json.loads(stamp_file.read_text(encoding="utf-8"))["created_at"]
    assert first["succeeded_count"] == 1

    second = run_prepare_stamp_for_job(
        job_id=JOB_ID,
        output_dir=output_dir,
        source_pdf=str(tmp_path / "input.pdf"),
        provider="anthropic",
        model="claude-opus-4-7",
    )
    second_created = json.loads(stamp_file.read_text(encoding="utf-8"))["created_at"]
    assert second["succeeded_count"] == 1
    assert second_created >= first_created


def test_run_prepare_stamp_missing_grounding_run_raises(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    (output_dir / "converted_images" / "pages").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Field grounding run not found"):
        run_prepare_stamp_for_job(
            job_id=JOB_ID,
            output_dir=output_dir,
            source_pdf=str(tmp_path / "input.pdf"),
            provider="openai",
            model="gpt-4o",
        )


def test_run_prepare_stamp_mixed_pages(tmp_path: Path) -> None:
    """If one grounding page has a dimension mismatch, only that page fails."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_job_fixture(output_dir, page_count=2)

    grounding_dir = output_dir / "field_grounding" / "anthropic_claude-opus-4-7"
    bad = _grounding_json(page_index=1, img_w=9999, img_h=9999)
    (grounding_dir / "page_0002.fields.json").write_text(json.dumps(bad) + "\n", encoding="utf-8")

    result = run_prepare_stamp_for_job(
        job_id=JOB_ID,
        output_dir=output_dir,
        source_pdf=str(tmp_path / "input.pdf"),
        provider="anthropic",
        model="claude-opus-4-7",
    )
    assert result["page_count"] == 2
    assert result["succeeded_count"] == 1
    assert result["failed_count"] == 1
    statuses = [p["status"] for p in result["pages"]]
    assert statuses == ["succeeded", "failed"]
