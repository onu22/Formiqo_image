#!/usr/bin/env python3
"""Seed deterministic jobs for frontend/e2e/smoke.mjs (G4 ship review)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.jobs import new_job_manifest, set_job_grounding, write_job_manifest

JOBS_DIR = ROOT / "data" / "jobs"

READY_ID = "11111111-1111-4111-8111-111111111101"
FAILED_ID = "22222222-2222-4222-8222-222222222202"
PROCESSING_ID = "33333333-3333-4333-8333-333333333303"

PDF_WIDTH_PT = 612.0
PDF_HEIGHT_PT = 792.0
DPI = 200.0
SCALE = 72.0 / DPI
IMAGE_WIDTH_PX = round(PDF_WIDTH_PT / SCALE)
IMAGE_HEIGHT_PX = round(PDF_HEIGHT_PT / SCALE)


def _write_page_assets(root: Path) -> None:
    output_dir = root / "output"
    pages_dir = output_dir / "converted_images" / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    input_pdf = root / "input.pdf"
    doc = fitz.open()
    try:
        page = doc.new_page(width=PDF_WIDTH_PT, height=PDF_HEIGHT_PT)
        page.insert_text((72, 120), "Patient Intake Form")
        page.insert_text((72, 200), "Date of Birth:")
        doc.save(input_pdf)
    finally:
        doc.close()

    Image.new("RGB", (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX), "white").save(
        output_dir / "converted_images" / "page_0001.png", format="PNG"
    )

    page_manifest = {
        "manifest_version": "1.0",
        "page_index": 0,
        "pdf": {"width_pt": PDF_WIDTH_PT, "height_pt": PDF_HEIGHT_PT, "origin": "bottom-left"},
        "image": {
            "path": "converted_images/page_0001.png",
            "format": "png",
            "width_px": IMAGE_WIDTH_PX,
            "height_px": IMAGE_HEIGHT_PX,
            "origin": "top-left",
        },
        "mapping": {"image_to_pdf_scale_x": SCALE, "image_to_pdf_scale_y": SCALE},
    }
    (pages_dir / "page_0001.json").write_text(json.dumps(page_manifest, indent=2) + "\n", encoding="utf-8")


def _write_fields(root: Path, *, include_date_of_birth: bool = True) -> None:
    fg_dir = root / "output" / "field_grounding"
    fg_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        {
            "field_id": "full_name",
            "type": "text",
            "bbox": {"x": 200, "y": 280, "w": 400, "h": 32},
            "confidence": 0.92,
            "label": "Full Name",
            "evidence": {"line_ids": []},
            "grounding_type": "text",
            "grounding_source": "label_anchor",
            "qa_status": None,
            "reviewed": False,
            "font_size_pt": None,
        },
    ]
    if include_date_of_birth:
        fields.append(
            {
                "field_id": "date_of_birth",
                "type": "text",
                "bbox": {"x": 200, "y": 360, "w": 220, "h": 28},
                "confidence": 0.88,
                "label": "Date of Birth",
                "evidence": {"line_ids": []},
                "grounding_type": "text",
                "grounding_source": "line_anchor",
                "qa_status": "flagged",
                "reviewed": False,
                "font_size_pt": 11.0,
            }
        )

    page_fields = {
        "page_index": 0,
        "width_px": IMAGE_WIDTH_PX,
        "height_px": IMAGE_HEIGHT_PX,
        "unit": "px",
        "origin": "top-left",
        "fields": fields,
    }
    (fg_dir / "page_0001.fields.json").write_text(json.dumps(page_fields, indent=2) + "\n", encoding="utf-8")

    stamping = {
        "values": {f["field_id"]: "" for f in fields},
        "require_all_values": False,
        "style": {"font_size_pt": 11, "text_color": "#111111"},
        "overrides": {},
    }
    (fg_dir / "stamping.json").write_text(json.dumps(stamping, indent=2) + "\n", encoding="utf-8")


def _write_line_detection(root: Path) -> None:
    src = ROOT / "tests" / "fixtures" / "golden-job" / "output" / "line_detection" / "page_0001"
    dst = root / "output" / "line_detection" / "page_0001"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def seed_ready_job() -> None:
    root = JOBS_DIR / READY_ID
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    manifest = new_job_manifest(job_id=READY_ID, source_filename="patient-intake.pdf")
    manifest["status"] = "ready"
    manifest["page_count"] = 1
    manifest["stages"]["convert"] = {"status": "done", "error": None, "failed_pages": []}
    manifest["stages"]["line_detect"] = {"status": "done", "error": None, "failed_pages": []}
    manifest["stages"]["grounding"] = {
        "status": "done",
        "error": None,
        "grounded_pages": 1,
        "total_pages": 1,
        "failed_pages": [],
    }
    write_job_manifest(root, manifest)
    set_job_grounding(root, provider="openai", model="gpt-4.1")

    _write_page_assets(root)
    _write_line_detection(root)
    _write_fields(root)


def seed_failed_job() -> None:
    root = JOBS_DIR / FAILED_ID
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    manifest = new_job_manifest(job_id=FAILED_ID, source_filename="corrupt-scan.pdf")
    manifest["status"] = "failed"
    manifest["page_count"] = 2
    manifest["stages"]["convert"] = {"status": "done", "error": None, "failed_pages": []}
    manifest["stages"]["line_detect"] = {"status": "done", "error": None, "failed_pages": []}
    manifest["stages"]["grounding"] = {
        "status": "failed",
        "error": "Grounding failed on page 2",
        "grounded_pages": 1,
        "total_pages": 2,
        "failed_pages": [{"page_index": 1, "detail": "vision model returned invalid field data."}],
    }
    write_job_manifest(root, manifest)
    set_job_grounding(root, provider="openai", model="gpt-4.1")

    _write_page_assets(root)
    _write_line_detection(root)
    _write_fields(root)


def seed_processing_job() -> None:
    root = JOBS_DIR / PROCESSING_ID
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    manifest = new_job_manifest(job_id=PROCESSING_ID, source_filename="w4-tax-form.pdf")
    manifest["status"] = "grounding"
    manifest["page_count"] = 3
    manifest["stages"]["convert"] = {"status": "done", "error": None, "failed_pages": []}
    manifest["stages"]["line_detect"] = {"status": "done", "error": None, "failed_pages": []}
    manifest["stages"]["grounding"] = {
        "status": "running",
        "error": None,
        "grounded_pages": 1,
        "total_pages": 3,
        "failed_pages": [],
    }
    write_job_manifest(root, manifest)

    _write_page_assets(root)


def main() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    seed_ready_job()
    seed_failed_job()
    seed_processing_job()
    print(f"READY_JOB={READY_ID}")
    print(f"FAILED_JOB={FAILED_ID}")
    print(f"PROCESSING_JOB={PROCESSING_ID}")


if __name__ == "__main__":
    main()
