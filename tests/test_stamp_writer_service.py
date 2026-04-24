"""Service-level tests for the stamp_writer (real PyMuPDF round-trip)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import fitz
import pytest

from app.services.stamp_writer import (
    StampOptions,
    resolve_stamped_pdf_path,
    run_stamp_pdf_for_job,
)

JOB_ID = "90082174-a5ac-466a-8798-3cb0204bffb3"
PROVIDER = "anthropic"
MODEL = "claude-opus-4-7"
RUN_DIR_NAME = f"{PROVIDER}_{MODEL}"


def _write_input_pdf(path: Path, *, width: float = 612.0, height: float = 792.0) -> None:
    doc = fitz.open()
    try:
        doc.new_page(width=width, height=height)
        buf = io.BytesIO()
        doc.save(buf)
    finally:
        doc.close()
    path.write_bytes(buf.getvalue())


def _write_stamp_plan(
    output_dir: Path,
    *,
    source_pdf: Path,
    provider: str = PROVIDER,
    model: str = MODEL,
) -> Path:
    plan_dir = output_dir / "stamp_plans" / f"{provider}_{model}"
    plan_dir.mkdir(parents=True, exist_ok=True)
    page_plan = {
        "schema_version": "1.0",
        "job_id": JOB_ID,
        "source_pdf": str(source_pdf),
        "page_index": 0,
        "pdf": {"width_pt": 612.0, "height_pt": 792.0, "origin": "bottom-left"},
        "image_reference": {
            "path": "converted_images/page_0001.png",
            "width_px": 1700,
            "height_px": 2200,
            "origin": "top-left",
        },
        "scales": {"image_to_pdf_scale_x": 0.36, "image_to_pdf_scale_y": 0.36},
        "grounding": {"provider": provider, "model": model, "run_id": "r1"},
        "unit": "pt",
        "created_at": "2026-01-01T00:00:00Z",
        "fields": [
            {
                "field_id": "first_name",
                "type": "text",
                "image_bbox": {"x": 100, "y": 100, "w": 400, "h": 60},
                "pdf_bbox_bl": {"pdf_x": 36.0, "pdf_y": 735.0, "pdf_w": 144.0, "pdf_h": 21.0},
                "pdf_rect_tl": {"x0": 36.0, "y0": 36.0, "x1": 180.0, "y1": 57.0},
            },
            {
                "field_id": "last_name",
                "type": "text",
                "image_bbox": {"x": 100, "y": 200, "w": 400, "h": 60},
                "pdf_bbox_bl": {"pdf_x": 36.0, "pdf_y": 700.0, "pdf_w": 144.0, "pdf_h": 21.0},
                "pdf_rect_tl": {"x0": 36.0, "y0": 71.0, "x1": 180.0, "y1": 92.0},
            },
            {
                "field_id": "personality_trait",
                "type": "text",
                "image_bbox": {"x": 100, "y": 300, "w": 400, "h": 60},
                "pdf_bbox_bl": {"pdf_x": 36.0, "pdf_y": 665.0, "pdf_w": 144.0, "pdf_h": 21.0},
                "pdf_rect_tl": {"x0": 36.0, "y0": 106.0, "x1": 180.0, "y1": 127.0},
            },
        ],
    }
    (plan_dir / "page_0001.stamp.json").write_text(json.dumps(page_plan) + "\n", encoding="utf-8")
    (plan_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "job_id": JOB_ID,
                "source_pdf": str(source_pdf),
                "provider": provider,
                "model": model,
                "run_id": "r1",
                "run_dir": f"stamp_plans/{provider}_{model}",
                "created_at": "2026-01-01T00:00:00Z",
                "page_count": 1,
                "succeeded_count": 1,
                "failed_count": 0,
                "files": [f"stamp_plans/{provider}_{model}/page_0001.stamp.json"],
                "pages": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return plan_dir


def _read_stamped_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    try:
        return " ".join(page.get_text() for page in doc)
    finally:
        doc.close()


def _setup_fixture(tmp_path: Path) -> tuple[Path, Path]:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    input_pdf = tmp_path / "input.pdf"
    _write_input_pdf(input_pdf)
    _write_stamp_plan(output_dir, source_pdf=input_pdf)
    return input_pdf, output_dir


def test_stamp_pdf_happy_path_renders_text_in_output(tmp_path: Path) -> None:
    input_pdf, output_dir = _setup_fixture(tmp_path)

    result = run_stamp_pdf_for_job(
        job_id=JOB_ID,
        input_pdf=input_pdf,
        output_dir=output_dir,
        provider=PROVIDER,
        model=MODEL,
        values={"first_name": "Jane", "last_name": "Doe"},
        strict=False,
    )
    assert result["provider"] == PROVIDER
    assert result["model"] == MODEL
    assert result["stamped_field_count"] == 2
    assert "personality_trait" in result["skipped_missing_values"]
    assert result["unknown_field_ids"] == []
    assert result["run_dir"] == f"stamped_pdfs/{RUN_DIR_NAME}"

    stamped_path = output_dir / "stamped_pdfs" / RUN_DIR_NAME / "stamped.pdf"
    result_path = output_dir / "stamped_pdfs" / RUN_DIR_NAME / "stamp_result.json"
    assert stamped_path.is_file()
    assert result_path.is_file()

    text = _read_stamped_text(stamped_path)
    assert "Jane" in text
    assert "Doe" in text

    audit = json.loads(result_path.read_text(encoding="utf-8"))
    assert audit["schema_version"] == "1.0"
    assert audit["summary"]["stamped_fields"] == 2
    assert audit["summary"]["skipped_missing_values"] == 1


def test_stamp_pdf_missing_stamp_plan_raises(tmp_path: Path) -> None:
    input_pdf = tmp_path / "input.pdf"
    _write_input_pdf(input_pdf)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Stamp plan run not found"):
        run_stamp_pdf_for_job(
            job_id=JOB_ID,
            input_pdf=input_pdf,
            output_dir=output_dir,
            provider=PROVIDER,
            model=MODEL,
            values={"first_name": "Jane"},
        )


def test_stamp_pdf_strict_rejects_unknown_id(tmp_path: Path) -> None:
    input_pdf, output_dir = _setup_fixture(tmp_path)

    with pytest.raises(ValueError, match="unknown field_ids"):
        run_stamp_pdf_for_job(
            job_id=JOB_ID,
            input_pdf=input_pdf,
            output_dir=output_dir,
            provider=PROVIDER,
            model=MODEL,
            values={
                "first_name": "Jane",
                "last_name": "Doe",
                "personality_trait": "calm",
                "bogus_field": "x",
            },
            strict=True,
        )


def test_stamp_pdf_strict_rejects_missing_values(tmp_path: Path) -> None:
    input_pdf, output_dir = _setup_fixture(tmp_path)

    with pytest.raises(ValueError, match="missing values"):
        run_stamp_pdf_for_job(
            job_id=JOB_ID,
            input_pdf=input_pdf,
            output_dir=output_dir,
            provider=PROVIDER,
            model=MODEL,
            values={"first_name": "Jane"},
            strict=True,
        )


def test_stamp_pdf_permissive_records_unknown_ids(tmp_path: Path) -> None:
    input_pdf, output_dir = _setup_fixture(tmp_path)

    result = run_stamp_pdf_for_job(
        job_id=JOB_ID,
        input_pdf=input_pdf,
        output_dir=output_dir,
        provider=PROVIDER,
        model=MODEL,
        values={"first_name": "Jane", "bogus_field": "ignored"},
        strict=False,
    )
    assert result["unknown_field_ids"] == ["bogus_field"]
    assert result["stamped_field_count"] == 1


def test_stamp_pdf_autoshrink_fits_long_text(tmp_path: Path) -> None:
    input_pdf, output_dir = _setup_fixture(tmp_path)
    long_text = "supercalifragilisticexpialidocious"

    result = run_stamp_pdf_for_job(
        job_id=JOB_ID,
        input_pdf=input_pdf,
        output_dir=output_dir,
        provider=PROVIDER,
        model=MODEL,
        values={"first_name": long_text},
        strict=False,
        options=StampOptions(fontsize=20.0, min_fontsize=6.0, autoshrink=True),
    )
    assert result["stamped_field_count"] == 1
    first_page = result["pages"][0]
    first_field = [f for f in first_page["fields"] if f["field_id"] == "first_name"][0]
    assert first_field["status"] == "stamped"
    assert first_field["final_fontsize"] <= 20.0


def test_stamp_pdf_overflow_is_recorded_when_autoshrink_disabled(tmp_path: Path) -> None:
    input_pdf, output_dir = _setup_fixture(tmp_path)
    result = run_stamp_pdf_for_job(
        job_id=JOB_ID,
        input_pdf=input_pdf,
        output_dir=output_dir,
        provider=PROVIDER,
        model=MODEL,
        values={"first_name": "supercalifragilisticexpialidocious"},
        strict=False,
        options=StampOptions(fontsize=40.0, min_fontsize=40.0, autoshrink=False),
    )
    first_page = result["pages"][0]
    first_field = [f for f in first_page["fields"] if f["field_id"] == "first_name"][0]
    assert first_field["status"] == "overflow"


def test_resolve_stamped_pdf_path_uses_run_dir(tmp_path: Path) -> None:
    path = resolve_stamped_pdf_path(tmp_path / "output", PROVIDER, MODEL)
    assert path == tmp_path / "output" / "stamped_pdfs" / RUN_DIR_NAME / "stamped.pdf"


def test_stamp_options_validates_inputs() -> None:
    with pytest.raises(ValueError):
        StampOptions(fontsize=0)
    with pytest.raises(ValueError):
        StampOptions(fontsize=10, min_fontsize=20)
    with pytest.raises(ValueError):
        StampOptions(align="justified")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        StampOptions(color_rgb=(1.5, 0, 0))
    with pytest.raises(ValueError):
        StampOptions(fontname="  ")
