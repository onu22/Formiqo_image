"""E7 template memory: line fingerprint, durable index, template reuse, near-miss safety."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.services.semantic_grounding as sg
from app.config import Settings
from app.dependencies import get_settings
from app.main import create_app
from app.services.jobs import new_job_manifest, read_job_manifest, write_job_manifest
from app.services.template_memory import (
    compute_page_fingerprint,
    load_template_grounding_for_page,
    page_line_fingerprint,
    record_corrected_pages,
)

_PNG_BYTES = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- fingerprint


def _lines(scale: float = 1.0) -> dict[str, Any]:
    """A four-line single-cell layout, optionally scaled (dims + coords) to test invariance."""
    return {
        "image": {"width": int(500 * scale), "height": int(300 * scale)},
        "lines": [
            {"line_id": "line_h_001", "orientation": "horizontal", "bbox": {"x": int(0 * scale), "y": int(100 * scale), "w": int(500 * scale), "h": 2}},
            {"line_id": "line_h_002", "orientation": "horizontal", "bbox": {"x": int(0 * scale), "y": int(200 * scale), "w": int(500 * scale), "h": 2}},
            {"line_id": "line_v_001", "orientation": "vertical", "bbox": {"x": int(100 * scale), "y": int(0 * scale), "w": 2, "h": int(300 * scale)}},
            {"line_id": "line_v_002", "orientation": "vertical", "bbox": {"x": int(400 * scale), "y": int(0 * scale), "w": 2, "h": int(300 * scale)}},
        ],
    }


def _fp(detected: dict[str, Any], *, quantize: int = 256, min_lines: int = 2) -> str | None:
    return page_line_fingerprint(detected, quantize=quantize, min_lines=min_lines)


def test_fingerprint_is_deterministic() -> None:
    assert _fp(_lines()) == _fp(_lines())
    assert _fp(_lines()) is not None


def test_fingerprint_is_scale_invariant() -> None:
    # Same layout rasterized at 2x DPI normalizes to the same fingerprint.
    assert _fp(_lines(scale=1.0)) == _fp(_lines(scale=2.0))


def test_fingerprint_line_order_invariant() -> None:
    detected = _lines()
    reversed_lines = {"image": detected["image"], "lines": list(reversed(detected["lines"]))}
    assert _fp(detected) == _fp(reversed_lines)


def test_fingerprint_near_miss_differs() -> None:
    base = _lines()
    near = json.loads(json.dumps(base))
    # Move one vertical line materially (100px on a 500px-wide page = 20% shift).
    near["lines"][2]["bbox"]["x"] = 250
    assert _fp(base) != _fp(near)


def test_fingerprint_different_line_count_differs() -> None:
    base = _lines()
    fewer = {"image": base["image"], "lines": base["lines"][:3]}
    assert _fp(base) != _fp(fewer)


def test_fingerprint_none_when_too_few_lines() -> None:
    sparse = {"image": {"width": 500, "height": 300}, "lines": _lines()["lines"][:1]}
    assert page_line_fingerprint(sparse, quantize=256, min_lines=2) is None


def test_fingerprint_none_when_no_dimensions() -> None:
    assert page_line_fingerprint({"lines": _lines()["lines"]}, quantize=256, min_lines=2) is None


# --------------------------------------------------------------------------- job fixtures


def _write_page(output_dir: Path, page_index: int, *, detected: dict[str, Any], fields: list[dict[str, Any]] | None = None) -> None:
    stem = f"page_{page_index + 1:04d}"
    conv = output_dir / "converted_images"
    (conv / "pages").mkdir(parents=True, exist_ok=True)
    (conv / f"{stem}.png").write_bytes(_PNG_BYTES)
    (conv / "pages" / f"{stem}.json").write_text(
        json.dumps(
            {
                "page_index": page_index,
                "image": {"width_px": detected["image"].get("width", detected["image"].get("width_px")),
                          "height_px": detected["image"].get("height", detected["image"].get("height_px"))},
                "mapping": {"image_to_pdf_scale_x": 0.36, "image_to_pdf_scale_y": 0.36},
            }
        ),
        encoding="utf-8",
    )
    ld = output_dir / "line_detection" / stem
    ld.mkdir(parents=True, exist_ok=True)
    (ld / "lines_highlighted.png").write_bytes(_PNG_BYTES)
    (ld / "detected_lines.json").write_text(json.dumps(detected), encoding="utf-8")

    if fields is not None:
        fg = output_dir / "field_grounding"
        fg.mkdir(parents=True, exist_ok=True)
        payload = {
            "page_index": page_index,
            "width_px": detected["image"].get("width", detected["image"].get("width_px")),
            "height_px": detected["image"].get("height", detected["image"].get("height_px")),
            "unit": "px",
            "origin": "top-left",
            "fields": fields,
        }
        (fg / f"{stem}.fields.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _corrected_fields() -> list[dict[str, Any]]:
    return [
        {
            "field_id": "field_001",
            "type": "text",
            "bbox": {"x": 120, "y": 150, "w": 260, "h": 40},
            "confidence": 1.0,
            "label": "Full Name",
            "evidence": {"line_ids": ["line_h_001", "line_h_002"]},
            "grounding_type": "text",
            "grounding_source": "cell",
            "qa_status": None,
            "reviewed": True,
            "font_size_pt": 11,
        }
    ]


def _make_job(tmp_path: Path, job_id: str, *, detected: dict[str, Any], fields: list[dict[str, Any]] | None = None) -> Path:
    root = tmp_path / job_id
    output_dir = root / "output"
    _write_page(output_dir, 0, detected=detected, fields=fields)
    manifest = new_job_manifest(job_id=job_id, source_filename="form.pdf")
    manifest["page_count"] = 1
    write_job_manifest(root, manifest)
    return output_dir


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = dict(
        jobs_dir=tmp_path / "jobs",
        templates_dir=tmp_path / "templates",
        template_memory_enabled=True,
        grounding_provider="openai",
        grounding_model="gpt-test",
    )
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------------- record + lookup


def test_record_and_lookup_roundtrip(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    output_dir = _make_job(tmp_path, "aaaaaaaa-1111-4111-8111-111111111111", detected=_lines(), fields=_corrected_fields())

    recorded = record_corrected_pages(
        settings=settings,
        output_dir=output_dir,
        source_job_id="aaaaaaaa-1111-4111-8111-111111111111",
        source_filename="form.pdf",
        page_numbers=[1],
    )
    assert len(recorded) == 1

    index_path = Path(settings.templates_dir) / "index.json"
    assert index_path.is_file()
    index = json.loads(index_path.read_text(encoding="utf-8"))["templates"]
    assert recorded[0] in index

    tpl = load_template_grounding_for_page(settings=settings, output_dir=output_dir, page_index=0)
    assert tpl is not None
    assert tpl["fields"][0]["field_id"] == "field_001"
    assert tpl["fields"][0]["grounding_source"] == "template"


def test_lookup_none_when_no_match(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    output_dir = _make_job(tmp_path, "bbbbbbbb-1111-4111-8111-111111111111", detected=_lines(), fields=_corrected_fields())
    # Nothing recorded yet.
    assert load_template_grounding_for_page(settings=settings, output_dir=output_dir, page_index=0) is None


def test_record_skips_when_disabled(tmp_path: Path) -> None:
    settings = _settings(tmp_path, template_memory_enabled=False)
    output_dir = _make_job(tmp_path, "cccccccc-1111-4111-8111-111111111111", detected=_lines(), fields=_corrected_fields())
    recorded = record_corrected_pages(
        settings=settings,
        output_dir=output_dir,
        source_job_id="cccccccc-1111-4111-8111-111111111111",
        source_filename="form.pdf",
        page_numbers=[1],
    )
    assert recorded == []


# --------------------------------------------------------------------------- reuse (zero LLM)


class _ExplodingOpenAI:
    def __init__(self, *_: Any, **__: Any) -> None:
        raise AssertionError("LLM client must not be constructed when all pages match a template")


def test_reupload_reuses_template_with_zero_llm_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path, openai_api_key="")  # no key: template path must not need one

    # 1) A previously human-corrected job seeds the template store.
    source_out = _make_job(tmp_path, "dddddddd-1111-4111-8111-111111111111", detected=_lines(), fields=_corrected_fields())
    record_corrected_pages(
        settings=settings,
        output_dir=source_out,
        source_job_id="dddddddd-1111-4111-8111-111111111111",
        source_filename="form.pdf",
        page_numbers=[1],
    )

    # 2) A fresh upload of the same layout (no fields yet) grounds via template only.
    new_job_id = "eeeeeeee-1111-4111-8111-111111111111"
    new_out = _make_job(tmp_path, new_job_id, detected=_lines(), fields=None)

    # Any attempt to touch the provider client is a hard failure.
    monkeypatch.setattr(sg, "OpenAI", _ExplodingOpenAI)
    monkeypatch.setattr(sg, "Anthropic", _ExplodingOpenAI)

    summary = sg.run_semantic_grounding_for_job(
        job_id=new_job_id,
        output_dir=new_out,
        settings=settings,
        provider="openai",
        model="gpt-test",
    )

    assert summary["succeeded_count"] == 1
    assert summary["failed_count"] == 0
    assert summary["template_pages"] == 1
    assert summary["llm_pages"] == 0

    fields_file = new_out / "field_grounding" / "page_0001.fields.json"
    data = json.loads(fields_file.read_text(encoding="utf-8"))
    assert data["fields"][0]["field_id"] == "field_001"
    assert data["fields"][0]["grounding_source"] == "template"

    manifest = read_job_manifest(new_out.parent)
    assert manifest["status"] == "ready"
    assert manifest["stages"]["grounding"]["grounded_pages"] == 1


def test_near_miss_does_not_reuse_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    source_out = _make_job(tmp_path, "ffffffff-1111-4111-8111-111111111111", detected=_lines(), fields=_corrected_fields())
    record_corrected_pages(
        settings=settings,
        output_dir=source_out,
        source_job_id="ffffffff-1111-4111-8111-111111111111",
        source_filename="form.pdf",
        page_numbers=[1],
    )

    # Different layout: template must not match, so this page routes to the LLM.
    near = json.loads(json.dumps(_lines()))
    near["lines"][2]["bbox"]["x"] = 250
    new_out = _make_job(tmp_path, "a0000000-1111-4111-8111-111111111111", detected=near, fields=None)

    tpl = load_template_grounding_for_page(settings=settings, output_dir=new_out, page_index=0)
    assert tpl is None


def test_disabled_memory_never_reuses(tmp_path: Path) -> None:
    settings_on = _settings(tmp_path)
    source_out = _make_job(tmp_path, "a1000000-1111-4111-8111-111111111111", detected=_lines(), fields=_corrected_fields())
    record_corrected_pages(
        settings=settings_on,
        output_dir=source_out,
        source_job_id="a1000000-1111-4111-8111-111111111111",
        source_filename="form.pdf",
        page_numbers=[1],
    )
    settings_off = _settings(tmp_path, template_memory_enabled=False)
    new_out = _make_job(tmp_path, "a2000000-1111-4111-8111-111111111111", detected=_lines(), fields=None)
    assert load_template_grounding_for_page(settings=settings_off, output_dir=new_out, page_index=0) is None


# --------------------------------------------------------------------------- router hook


def _client(settings: Settings) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_patch_fields_records_template(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job_id = "a3000000-1111-4111-8111-111111111111"
    (settings.jobs_dir).mkdir(parents=True, exist_ok=True)
    # Job lives under settings.jobs_dir/job_id per job_paths().
    root = settings.jobs_dir / job_id
    output_dir = root / "output"
    _write_page(output_dir, 0, detected=_lines(), fields=_corrected_fields())
    manifest = new_job_manifest(job_id=job_id, source_filename="repeat-form.pdf")
    manifest["page_count"] = 1
    write_job_manifest(root, manifest)
    # stamping.json is required for the fields payload readers elsewhere; not needed for PATCH.

    client = _client(settings)
    resp = client.patch(
        f"/api/v1/jobs/{job_id}/fields",
        json={"fields": [{"field_id": "field_001", "page_number": 1, "bbox": {"x": 130, "y": 160, "w": 250, "h": 38}}]},
    )
    assert resp.status_code == 200, resp.text

    index_path = Path(settings.templates_dir) / "index.json"
    assert index_path.is_file()
    index = json.loads(index_path.read_text(encoding="utf-8"))["templates"]
    assert len(index) == 1
    # The stored fingerprint now resolves for a matching new upload.
    fp = compute_page_fingerprint(settings=settings, output_dir=output_dir, page_index=0)
    assert fp in index
