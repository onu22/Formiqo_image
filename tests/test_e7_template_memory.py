"""E7 template memory: fingerprint, store, reuse, capture, and pipeline integration.

Proves the PRD acceptance criteria:
- Re-uploading a previously corrected form reuses the corrected layout with **zero LLM calls**
  on matched pages (``grounding_source: template``), reaching ``ready`` without an API key.
- A near-miss form (different layout) does **not** false-positive match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import app.services.semantic_grounding as sg
from app.config import Settings
from app.services.jobs import new_job_manifest, read_job_manifest, write_job_manifest
from app.services.semantic_grounding import run_semantic_grounding_for_job
from app.services.template_memory import (
    TemplateStore,
    build_template_grounding_for_page,
    capture_corrected_page,
    match_templates_for_job,
    normalized_line_tokens,
    page_fingerprint,
)

_JOB_ID = "33333333-3333-4333-8333-333333333333"
_JOB_ID_2 = "44444444-4444-4444-8444-444444444444"

_BINS = 200


# --------------------------------------------------------------------------- layouts


def _line(orientation: str, x: int, y: int, w: int, h: int) -> dict[str, Any]:
    return {"orientation": orientation, "bbox": {"x": x, "y": y, "w": w, "h": h}, "thickness": h}


def _detected_lines(lines: list[dict[str, Any]], *, width: int = 1700, height: int = 2200) -> dict[str, Any]:
    return {
        "image": {"width": width, "height": height, "unit": "px", "origin": "top-left"},
        "detector": {"method": "opencv_morphology"},
        "counts": {"total": len(lines)},
        "lines": lines,
    }


def _reference_layout() -> list[dict[str, Any]]:
    return [
        _line("horizontal", 100, 300, 800, 4),
        _line("horizontal", 100, 500, 800, 4),
        _line("horizontal", 100, 700, 800, 4),
        _line("vertical", 100, 300, 4, 400),
        _line("vertical", 900, 300, 4, 400),
    ]


# --------------------------------------------------------------------------- fingerprint


def test_fingerprint_is_deterministic() -> None:
    dl = _detected_lines(_reference_layout())
    fp1 = page_fingerprint(dl, bins=_BINS)
    fp2 = page_fingerprint(_detected_lines(_reference_layout()), bins=_BINS)
    assert fp1 is not None
    assert fp1 == fp2
    assert len(fp1) == 64  # sha256 hex


def test_fingerprint_is_scale_invariant() -> None:
    """Same layout rendered at 2x DPI (coords + dims doubled) yields the same fingerprint."""
    base = _reference_layout()
    scaled = [
        _line(ln["orientation"], ln["bbox"]["x"] * 2, ln["bbox"]["y"] * 2, ln["bbox"]["w"] * 2, ln["bbox"]["h"] * 2)
        for ln in base
    ]
    fp_base = page_fingerprint(_detected_lines(base, width=1700, height=2200), bins=_BINS)
    fp_scaled = page_fingerprint(_detected_lines(scaled, width=3400, height=4400), bins=_BINS)
    assert fp_base == fp_scaled


def test_fingerprint_none_for_empty_layout() -> None:
    assert page_fingerprint(_detected_lines([]), bins=_BINS) is None
    assert page_fingerprint({"image": {"width": 0, "height": 0}, "lines": []}, bins=_BINS) is None
    assert normalized_line_tokens({"image": {}, "lines": []}, bins=_BINS) == []


def test_near_miss_layout_does_not_match_fingerprint() -> None:
    reference = page_fingerprint(_detected_lines(_reference_layout()), bins=_BINS)

    # Move one horizontal line by ~80px (well beyond one 200-bin bucket on a 2200px page).
    moved = _reference_layout()
    moved[1] = _line("horizontal", 100, 580, 800, 4)
    assert page_fingerprint(_detected_lines(moved), bins=_BINS) != reference

    # A different number of lines is also a clear miss.
    fewer = _reference_layout()[:-1]
    assert page_fingerprint(_detected_lines(fewer), bins=_BINS) != reference


def test_sub_bucket_jitter_still_matches() -> None:
    """A 1px shift (below one bucket at 200 bins on a 2200px page ~ 11px) stays a match."""
    reference = page_fingerprint(_detected_lines(_reference_layout()), bins=_BINS)
    jittered = _reference_layout()
    jittered[0] = _line("horizontal", 101, 301, 800, 4)
    assert page_fingerprint(_detected_lines(jittered), bins=_BINS) == reference


# --------------------------------------------------------------------------- store


def _corrected_fields() -> list[dict[str, Any]]:
    return [
        {
            "field_id": "field_001",
            "type": "text",
            "bbox": {"x": 120, "y": 292, "w": 300, "h": 26},
            "confidence": 0.9,
            "label": "Name",
            "evidence": {"line_ids": ["line_h_001"]},
            "grounding_source": "pixel",
            "qa_status": None,
            "reviewed": True,
            "font_size_pt": 11,
        },
        {
            "field_id": "field_002",
            "type": "text",
            "bbox": {"x": 120, "y": 492, "w": 300, "h": 26},
            "confidence": 0.8,
            "label": "Date",
            "evidence": {"line_ids": ["line_h_002"]},
            "grounding_source": "cell",
            "qa_status": None,
            "reviewed": True,
            "font_size_pt": None,
        },
    ]


def test_store_roundtrip_and_small_index(tmp_path: Path) -> None:
    store = TemplateStore(tmp_path / "templates")
    fp = page_fingerprint(_detected_lines(_reference_layout()), bins=_BINS)
    assert fp is not None

    store.upsert(
        fp,
        source_job_id=_JOB_ID,
        source_page_number=1,
        width_px=1700,
        height_px=2200,
        line_count=5,
        fields=_corrected_fields(),
    )

    found = store.lookup(fp)
    assert found is not None
    assert found["record"]["source_job_id"] == _JOB_ID
    assert found["record"]["field_count"] == 2
    assert found["payload"]["fields"][0]["field_id"] == "field_001"

    # Index stays small: metadata only, field payloads live in a separate file.
    index = json.loads((tmp_path / "templates" / "index.json").read_text())
    assert list(index["templates"]) == [fp]
    assert "fields" not in index["templates"][fp]
    assert (tmp_path / "templates" / f"{fp}.fields.json").is_file()


def test_store_lookup_miss_returns_none(tmp_path: Path) -> None:
    store = TemplateStore(tmp_path / "templates")
    assert store.lookup("0" * 64) is None
    assert store.lookup("not-a-valid-fingerprint") is None


def test_build_template_grounding_scales_and_marks_source() -> None:
    payload = {
        "width_px": 1700,
        "height_px": 2200,
        "fields": _corrected_fields(),
    }
    # Current page is 2x the template page: bboxes should scale accordingly.
    grounding = build_template_grounding_for_page(payload, page_index=0, cur_w=3400, cur_h=4400)
    assert grounding["width_px"] == 3400
    f0 = grounding["fields"][0]
    assert f0["grounding_source"] == "template"
    assert f0["bbox"] == {"x": 240, "y": 584, "w": 600, "h": 52}


# --------------------------------------------------------------------------- job fixture


def _build_job(
    root: Path,
    *,
    page_layouts: list[list[dict[str, Any]]],
    width: int = 1700,
    height: int = 2200,
    job_id: str = _JOB_ID,
    with_fields: dict[int, list[dict[str, Any]]] | None = None,
) -> Path:
    """Create converted PNGs, per-page detected_lines.json, page manifests, and job.json."""
    output = root / "output"
    conv = output / "converted_images"
    (conv / "pages").mkdir(parents=True)
    for i, layout in enumerate(page_layouts):
        page_number = i + 1
        Image.new("RGB", (width, height), (255, 255, 255)).save(conv / f"page_{page_number:04d}.png")
        (conv / "pages" / f"page_{page_number:04d}.json").write_text(
            json.dumps(
                {
                    "page_index": i,
                    "image": {"width_px": width, "height_px": height, "origin": "top-left"},
                    "mapping": {"image_to_pdf_scale_x": 0.36, "image_to_pdf_scale_y": 0.36},
                }
            ),
            encoding="utf-8",
        )
        ld_dir = output / "line_detection" / f"page_{page_number:04d}"
        ld_dir.mkdir(parents=True)
        (ld_dir / "detected_lines.json").write_text(
            json.dumps(_detected_lines(layout, width=width, height=height)), encoding="utf-8"
        )

    if with_fields:
        fg = output / "field_grounding"
        fg.mkdir(parents=True, exist_ok=True)
        for page_index, fields in with_fields.items():
            (fg / f"page_{page_index + 1:04d}.fields.json").write_text(
                json.dumps(
                    {
                        "page_index": page_index,
                        "width_px": width,
                        "height_px": height,
                        "unit": "px",
                        "origin": "top-left",
                        "fields": fields,
                    }
                ),
                encoding="utf-8",
            )

    manifest = new_job_manifest(job_id=job_id, source_filename="form.pdf")
    manifest["page_count"] = len(page_layouts)
    manifest["status"] = "grounding"
    write_job_manifest(root, manifest)
    return output


# --------------------------------------------------------------------------- match


def test_match_empty_index_returns_nothing(tmp_path: Path) -> None:
    output = _build_job(tmp_path / _JOB_ID, page_layouts=[_reference_layout()])
    settings = Settings(jobs_dir=tmp_path, templates_dir=tmp_path / "templates")
    assert match_templates_for_job(output_dir=output, settings=settings) == {}


def test_match_returns_reused_page(tmp_path: Path) -> None:
    settings = Settings(jobs_dir=tmp_path, templates_dir=tmp_path / "templates")
    store = TemplateStore(settings.templates_dir)
    fp = page_fingerprint(_detected_lines(_reference_layout()), bins=settings.template_fingerprint_bins)
    store.upsert(
        fp,
        source_job_id=_JOB_ID,
        source_page_number=1,
        width_px=1700,
        height_px=2200,
        line_count=5,
        fields=_corrected_fields(),
    )

    output = _build_job(tmp_path / _JOB_ID_2, page_layouts=[_reference_layout()], job_id=_JOB_ID_2)
    matched = match_templates_for_job(output_dir=output, settings=settings)
    assert set(matched) == {0}
    assert matched[0]["fields"][0]["grounding_source"] == "template"


def test_match_respects_min_lines_guard(tmp_path: Path) -> None:
    settings = Settings(jobs_dir=tmp_path, templates_dir=tmp_path / "templates", template_min_lines=10)
    store = TemplateStore(settings.templates_dir)
    fp = page_fingerprint(_detected_lines(_reference_layout()), bins=settings.template_fingerprint_bins)
    store.upsert(
        fp, source_job_id=_JOB_ID, source_page_number=1, width_px=1700, height_px=2200, line_count=5,
        fields=_corrected_fields(),
    )
    output = _build_job(tmp_path / _JOB_ID_2, page_layouts=[_reference_layout()], job_id=_JOB_ID_2)
    # Only 5 lines < min 10 → never matched.
    assert match_templates_for_job(output_dir=output, settings=settings) == {}


# --------------------------------------------------------------------------- capture


def test_capture_corrected_page_stores_template(tmp_path: Path) -> None:
    settings = Settings(jobs_dir=tmp_path, templates_dir=tmp_path / "templates")
    output = _build_job(
        tmp_path / _JOB_ID,
        page_layouts=[_reference_layout()],
        with_fields={0: _corrected_fields()},
    )
    fp = capture_corrected_page(output_dir=output, settings=settings, source_job_id=_JOB_ID, page_number=1)
    assert fp is not None

    store = TemplateStore(settings.templates_dir)
    found = store.lookup(fp)
    assert found is not None
    assert found["record"]["source_job_id"] == _JOB_ID
    assert [f["field_id"] for f in found["payload"]["fields"]] == ["field_001", "field_002"]


def test_capture_disabled_when_memory_off(tmp_path: Path) -> None:
    settings = Settings(jobs_dir=tmp_path, templates_dir=tmp_path / "templates", template_memory_enabled=False)
    output = _build_job(tmp_path / _JOB_ID, page_layouts=[_reference_layout()], with_fields={0: _corrected_fields()})
    assert capture_corrected_page(output_dir=output, settings=settings, source_job_id=_JOB_ID, page_number=1) is None


# --------------------------------------------------------------------------- pipeline integration


def _forbid_llm(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace ground_one_page with a guard that records (and would fail) any LLM call."""
    calls: list[int] = []

    def _boom(*, page_index: int, **_: Any) -> dict[str, Any]:
        calls.append(page_index)
        raise AssertionError(f"LLM grounding must not run for page {page_index}")

    monkeypatch.setattr(sg, "ground_one_page", _boom)
    return calls


def test_all_pages_templated_zero_llm_no_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Acceptance: full re-upload reuse — ready, zero LLM calls, no API key required."""
    settings = Settings(jobs_dir=tmp_path, templates_dir=tmp_path / "templates", openai_api_key="")
    store = TemplateStore(settings.templates_dir)
    fp = page_fingerprint(_detected_lines(_reference_layout()), bins=settings.template_fingerprint_bins)
    store.upsert(
        fp, source_job_id=_JOB_ID, source_page_number=1, width_px=1700, height_px=2200, line_count=5,
        fields=_corrected_fields(),
    )

    output = _build_job(tmp_path / _JOB_ID_2, page_layouts=[_reference_layout()], job_id=_JOB_ID_2)
    matched = match_templates_for_job(output_dir=output, settings=settings)
    assert set(matched) == {0}

    calls = _forbid_llm(monkeypatch)
    summary = run_semantic_grounding_for_job(
        job_id=_JOB_ID_2,
        output_dir=output,
        settings=settings,
        provider="openai",
        model="gpt-5",
        template_page_results=matched,
    )

    assert calls == []  # zero LLM calls
    assert summary["template_count"] == 1
    assert summary["succeeded_count"] == 1

    manifest = read_job_manifest(output.parent)
    assert manifest["status"] == "ready"
    assert manifest["stages"]["grounding"]["template_pages"] == [0]

    fields = json.loads((output / "field_grounding" / "page_0001.fields.json").read_text())["fields"]
    assert [f["field_id"] for f in fields] == ["field_001", "field_002"]
    assert all(f["grounding_source"] == "template" for f in fields)
    # Corrected layout is preserved exactly (same dims → identity scale).
    assert fields[0]["bbox"] == {"x": 120, "y": 292, "w": 300, "h": 26}


def test_mixed_matched_and_unmatched_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Page 0 reuses a template (no LLM); page 1 is a near-miss and is grounded by the LLM."""
    settings = Settings(jobs_dir=tmp_path, templates_dir=tmp_path / "templates", openai_api_key="test-key")
    store = TemplateStore(settings.templates_dir)
    fp = page_fingerprint(_detected_lines(_reference_layout()), bins=settings.template_fingerprint_bins)
    store.upsert(
        fp, source_job_id=_JOB_ID, source_page_number=1, width_px=1700, height_px=2200, line_count=5,
        fields=_corrected_fields(),
    )

    near_miss = _reference_layout()
    near_miss[1] = _line("horizontal", 100, 900, 800, 4)  # clearly different page
    output = _build_job(
        tmp_path / _JOB_ID_2,
        page_layouts=[_reference_layout(), near_miss],
        job_id=_JOB_ID_2,
    )
    matched = match_templates_for_job(output_dir=output, settings=settings)
    assert set(matched) == {0}

    llm_calls: list[int] = []

    def _fake_ground(*, page_index: int, **_: Any) -> dict[str, Any]:
        llm_calls.append(page_index)
        return {
            "page_index": page_index,
            "grounding": {
                "page_index": page_index,
                "width_px": 1700,
                "height_px": 2200,
                "unit": "px",
                "origin": "top-left",
                "fields": [
                    {
                        "field_id": "llm_field",
                        "type": "text",
                        "bbox": {"x": 200, "y": 950, "w": 250, "h": 24},
                        "confidence": 0.7,
                        "label": "LLM",
                        "evidence": {"line_ids": []},
                        "grounding_source": "pixel",
                    }
                ],
            },
        }

    monkeypatch.setattr(sg, "ground_one_page", _fake_ground)

    summary = run_semantic_grounding_for_job(
        job_id=_JOB_ID_2,
        output_dir=output,
        settings=settings,
        provider="openai",
        model="gpt-5",
        template_page_results=matched,
    )

    # LLM ran only for the unmatched page.
    assert llm_calls == [1]
    assert summary["template_count"] == 1
    assert summary["succeeded_count"] == 2

    page1 = json.loads((output / "field_grounding" / "page_0001.fields.json").read_text())["fields"]
    page2 = json.loads((output / "field_grounding" / "page_0002.fields.json").read_text())["fields"]
    assert all(f["grounding_source"] == "template" for f in page1)
    assert page2[0]["grounding_source"] == "pixel"
    assert page2[0]["field_id"] == "llm_field"


def test_capture_then_reupload_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: capture a corrected page, then a fresh job with the same layout reuses it."""
    settings = Settings(jobs_dir=tmp_path, templates_dir=tmp_path / "templates", openai_api_key="")

    # Job 1: user corrected fields saved via editor → capture.
    output1 = _build_job(
        tmp_path / _JOB_ID,
        page_layouts=[_reference_layout()],
        with_fields={0: _corrected_fields()},
    )
    fp = capture_corrected_page(output_dir=output1, settings=settings, source_job_id=_JOB_ID, page_number=1)
    assert fp is not None

    # Job 2: same layout re-uploaded.
    output2 = _build_job(tmp_path / _JOB_ID_2, page_layouts=[_reference_layout()], job_id=_JOB_ID_2)
    matched = match_templates_for_job(output_dir=output2, settings=settings)
    assert set(matched) == {0}

    calls = _forbid_llm(monkeypatch)
    run_semantic_grounding_for_job(
        job_id=_JOB_ID_2,
        output_dir=output2,
        settings=settings,
        template_page_results=matched,
    )
    assert calls == []
    assert read_job_manifest(output2.parent)["status"] == "ready"

    fields = json.loads((output2 / "field_grounding" / "page_0001.fields.json").read_text())["fields"]
    assert [f["field_id"] for f in fields] == ["field_001", "field_002"]
    assert fields[0]["bbox"] == {"x": 120, "y": 292, "w": 300, "h": 26}


def test_near_miss_reupload_does_not_match(tmp_path: Path) -> None:
    """A different layout must not reuse the captured template (no false positive)."""
    settings = Settings(jobs_dir=tmp_path, templates_dir=tmp_path / "templates")
    output1 = _build_job(
        tmp_path / _JOB_ID,
        page_layouts=[_reference_layout()],
        with_fields={0: _corrected_fields()},
    )
    capture_corrected_page(output_dir=output1, settings=settings, source_job_id=_JOB_ID, page_number=1)

    different = _reference_layout()
    different.append(_line("horizontal", 100, 1200, 800, 4))  # extra line → different form
    output2 = _build_job(tmp_path / _JOB_ID_2, page_layouts=[different], job_id=_JOB_ID_2)
    assert match_templates_for_job(output_dir=output2, settings=settings) == {}
