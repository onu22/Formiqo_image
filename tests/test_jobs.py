"""Tests for job.json manifest read/write and status transitions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.jobs import (
    new_job_manifest,
    read_job_manifest,
    set_job_status,
    update_job_after_convert,
    update_job_after_grounding,
    update_job_after_line_detect,
    update_job_stage,
    write_job_manifest,
)


def test_new_job_manifest_shape() -> None:
    manifest = new_job_manifest(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        source_filename="form.pdf",
        dpi=200,
        detected_pdf_type="flat",
    )
    assert manifest["status"] == "converting"
    assert manifest["stages"]["convert"]["status"] == "pending"
    assert manifest["stages"]["qa_refine"]["status"] == "skipped"
    assert manifest["grounding"]["provider"] is None
    assert manifest["retention"]["max_stamp_runs"] == 3


def test_job_manifest_lifecycle(tmp_path: Path) -> None:
    job_id = "550e8400-e29b-41d4-a716-446655440000"
    root = tmp_path / job_id
    root.mkdir()

    write_job_manifest(
        root,
        new_job_manifest(job_id=job_id, source_filename="demo.pdf"),
    )

    update_job_after_convert(root, page_count=2, dpi=200)
    manifest = read_job_manifest(root)
    assert manifest["page_count"] == 2
    assert manifest["stages"]["convert"]["status"] == "done"

    update_job_after_line_detect(root)
    manifest = read_job_manifest(root)
    assert manifest["status"] == "grounding"
    assert manifest["stages"]["line_detect"]["status"] == "done"
    assert manifest["stages"]["grounding"]["status"] == "pending"
    assert manifest["stages"]["grounding"]["total_pages"] == 2

    update_job_after_grounding(
        root,
        provider="openai",
        model="gpt-4.1",
        grounded_pages=2,
        total_pages=2,
        failed_pages=[],
    )
    manifest = read_job_manifest(root)
    assert manifest["status"] == "ready"
    assert manifest["grounding"]["provider"] == "openai"
    assert manifest["grounding"]["model"] == "gpt-4.1"
    assert manifest["stages"]["grounding"]["grounded_pages"] == 2


def test_set_job_status_updates_timestamp(tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    write_job_manifest(
        root,
        new_job_manifest(
            job_id="550e8400-e29b-41d4-a716-446655440001",
            source_filename="x.pdf",
        ),
    )
    before = read_job_manifest(root)["updated_at"]
    set_job_status(root, "failed")
    after = read_job_manifest(root)
    assert after["status"] == "failed"
    assert after["updated_at"] >= before


def test_update_job_stage_merges_fields(tmp_path: Path) -> None:
    root = tmp_path / "job2"
    root.mkdir()
    write_job_manifest(
        root,
        new_job_manifest(
            job_id="550e8400-e29b-41d4-a716-446655440002",
            source_filename="y.pdf",
        ),
    )
    update_job_stage(root, "convert", status="failed", error="render error", failed_pages=[1])
    manifest = read_job_manifest(root)
    assert manifest["stages"]["convert"]["status"] == "failed"
    assert manifest["stages"]["convert"]["error"] == "render error"
    assert manifest["stages"]["convert"]["failed_pages"] == [1]


def test_read_job_manifest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_job_manifest(tmp_path / "missing")


def test_golden_job_fixture_layout() -> None:
    fixture_root = Path(__file__).resolve().parent / "fixtures" / "golden-job"
    job_json = fixture_root / "job.json"
    assert job_json.is_file(), "golden-job fixture missing job.json"
    manifest = json.loads(job_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "ready"
    assert (fixture_root / "output" / "field_grounding" / "page_0001.fields.json").is_file()
    page_manifest = json.loads(
        (fixture_root / "output" / "converted_images" / "pages" / "page_0001.json").read_text(
            encoding="utf-8"
        )
    )
    image = page_manifest["image"]
    assert "width_px" in image
    assert "saved_image_width_px" not in image
    fields_payload = json.loads(
        (fixture_root / "output" / "field_grounding" / "page_0001.fields.json").read_text(
            encoding="utf-8"
        )
    )
    assert "width_px" in fields_payload
    field = fields_payload["fields"][0]
    assert "nearby_label_text" not in field
    assert "supporting_lines" not in field
    assert "field_surface" not in field
