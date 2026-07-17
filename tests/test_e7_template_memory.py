"""E7 template memory: fingerprint, index, reuse, near-miss safety, zero-LLM re-upload."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import app.services.semantic_grounding as sg
from app.config import Settings
from app.services.jobs import new_job_manifest, read_job_manifest, write_job_manifest
from app.services.template_memory import (
    TemplateStore,
    build_template_grounding_for_page,
    capture_corrected_page,
    match_templates_for_job,
    page_fingerprint,
)

_PNG_BYTES = b"\x89PNG\r\n\x1a\n"


def _lines(width: int = 500, height: int = 300) -> dict[str, Any]:
    # Scale a canonical 4-line single-cell layout to (width, height).
    sx, sy = width / 500.0, height / 300.0
    def s(bbox: dict[str, int]) -> dict[str, int]:
        return {
            "x": round(bbox["x"] * sx),
            "y": round(bbox["y"] * sy),
            "w": max(1, round(bbox["w"] * sx)),
            "h": max(1, round(bbox["h"] * sy)),
        }
    return {
        "image": {"width": width, "height": height},
        "lines": [
            {"line_id": "line_h_001", "orientation": "horizontal", "bbox": s({"x": 0, "y": 100, "w": 500, "h": 2})},
            {"line_id": "line_h_002", "orientation": "horizontal", "bbox": s({"x": 0, "y": 200, "w": 500, "h": 2})},
            {"line_id": "line_v_001", "orientation": "vertical", "bbox": s({"x": 100, "y": 0, "w": 2, "h": 300})},
            {"line_id": "line_v_002", "orientation": "vertical", "bbox": s({"x": 400, "y": 0, "w": 2, "h": 300})},
        ],
    }


# --------------------------------------------------------------------------- fingerprint (T040)


def test_fingerprint_is_scale_invariant() -> None:
    fp_small = page_fingerprint(_lines(500, 300))
    fp_large = page_fingerprint(_lines(1000, 600))  # same layout, 2x DPI
    assert fp_small is not None
    assert fp_small == fp_large


def test_fingerprint_none_below_min_lines() -> None:
    sparse = {"image": {"width": 500, "height": 300}, "lines": _lines()["lines"][:2]}
    assert page_fingerprint(sparse, min_lines=4) is None


def test_fingerprint_near_miss_differs() -> None:
    base = _lines()
    near = _lines()
    # Move one vertical line materially → different layout, must not collide.
    near["lines"][2]["bbox"]["x"] = 250
    assert page_fingerprint(base) != page_fingerprint(near)


def test_fingerprint_none_when_dims_missing() -> None:
    assert page_fingerprint({"lines": _lines()["lines"]}) is None


# --------------------------------------------------------------------------- store (T041)


def test_template_store_roundtrip(tmp_path: Path) -> None:
    store = TemplateStore(tmp_path / "templates")
    fp = page_fingerprint(_lines())
    assert fp is not None
    payload = {"page_index": 0, "width_px": 500, "height_px": 300, "fields": [{"field_id": "a"}]}
    assert store.has(fp) is False
    store.put(fp, payload, meta={"source_job_id": "j", "field_count": 1})
    assert store.has(fp) is True
    assert store.get(fp)["fields"][0]["field_id"] == "a"
    index = json.loads((tmp_path / "templates" / "index.json").read_text())
    assert index[fp]["field_count"] == 1


def test_store_get_missing_returns_none(tmp_path: Path) -> None:
    store = TemplateStore(tmp_path / "templates")
    assert store.get("f" * 64) is None
    assert store.get("not-a-hash") is None


def test_build_template_grounding_scales_and_marks_source() -> None:
    template = {
        "page_index": 0,
        "width_px": 500,
        "height_px": 300,
        "fields": [{"field_id": "a", "type": "text", "bbox": {"x": 100, "y": 100, "w": 200, "h": 20}}],
    }
    out = build_template_grounding_for_page(template, page_index=0, width_px=1000, height_px=600)
    assert out["width_px"] == 1000 and out["height_px"] == 600
    field = out["fields"][0]
    assert field["grounding_source"] == "template"
    # bbox scaled 2x on both axes.
    assert field["bbox"] == {"x": 200, "y": 200, "w": 400, "h": 40}


# --------------------------------------------------------------------------- job fixtures


def _write_page(output_dir: Path, page_index: int, *, width: int = 500, height: int = 300) -> None:
    stem = f"page_{page_index + 1:04d}"
    conv = output_dir / "converted_images"
    (conv / "pages").mkdir(parents=True, exist_ok=True)
    (conv / f"{stem}.png").write_bytes(_PNG_BYTES)
    (conv / "pages" / f"{stem}.json").write_text(
        json.dumps(
            {
                "page_index": page_index,
                "pdf": {"width_pt": 180.0, "height_pt": 108.0},
                "image": {"path": f"converted_images/{stem}.png", "width_px": width, "height_px": height},
                "mapping": {"image_to_pdf_scale_x": 0.36, "image_to_pdf_scale_y": 0.36},
            }
        ),
        encoding="utf-8",
    )
    ld = output_dir / "line_detection" / stem
    ld.mkdir(parents=True, exist_ok=True)
    (ld / "lines_highlighted.png").write_bytes(_PNG_BYTES)
    (ld / "detected_lines.json").write_text(json.dumps(_lines(width, height)), encoding="utf-8")


def _make_job(tmp_path: Path, job_id: str, pages: int, *, width: int = 500, height: int = 300) -> Path:
    root = tmp_path / job_id
    output_dir = root / "output"
    for i in range(pages):
        _write_page(output_dir, i, width=width, height=height)
    manifest = new_job_manifest(job_id=job_id, source_filename="form.pdf")
    manifest["page_count"] = pages
    write_job_manifest(root, manifest)
    return output_dir


def _corrected_fields(page_index: int, width: int = 500, height: int = 300) -> dict[str, Any]:
    return {
        "page_index": page_index,
        "width_px": width,
        "height_px": height,
        "unit": "px",
        "origin": "top-left",
        "fields": [
            {
                "field_id": "name",
                "type": "text",
                "bbox": {"x": 150, "y": 150, "w": 200, "h": 30},
                "confidence": 0.99,
                "label": "Name",
                "evidence": {"line_ids": ["line_h_001"]},
                "grounding_source": "cell",
                "qa_status": "confirmed",
                "reviewed": True,
                "font_size_pt": None,
            }
        ],
    }


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = dict(jobs_dir=tmp_path, templates_dir=tmp_path / "templates")
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------------- capture + match (T041/T042)


def test_capture_then_match_for_job(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    src = _make_job(tmp_path, "11111111-1111-4111-8111-111111111111", pages=1)
    (src / "field_grounding").mkdir(parents=True, exist_ok=True)
    (src / "field_grounding" / "page_0001.fields.json").write_text(
        json.dumps(_corrected_fields(0)), encoding="utf-8"
    )

    assert capture_corrected_page(src, 0, settings) is True

    # A fresh job with the same layout (different DPI) matches and gets template grounding.
    dst = _make_job(tmp_path, "22222222-2222-4222-8222-222222222222", pages=1, width=1000, height=600)
    matches = match_templates_for_job(dst, settings)
    assert 0 in matches
    field = matches[0]["fields"][0]
    assert field["grounding_source"] == "template"
    assert field["bbox"] == {"x": 300, "y": 300, "w": 400, "h": 60}  # scaled 2x


def test_near_miss_does_not_match(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    src = _make_job(tmp_path, "11111111-1111-4111-8111-111111111111", pages=1)
    (src / "field_grounding").mkdir(parents=True, exist_ok=True)
    (src / "field_grounding" / "page_0001.fields.json").write_text(
        json.dumps(_corrected_fields(0)), encoding="utf-8"
    )
    assert capture_corrected_page(src, 0, settings) is True

    # Different layout: perturb a detected line so the fingerprint differs.
    dst = _make_job(tmp_path, "22222222-2222-4222-8222-222222222222", pages=1)
    ld_path = dst / "line_detection" / "page_0001" / "detected_lines.json"
    detected = json.loads(ld_path.read_text())
    detected["lines"][2]["bbox"]["x"] = 250
    ld_path.write_text(json.dumps(detected), encoding="utf-8")

    assert match_templates_for_job(dst, settings) == {}


def test_disabled_flag_skips_capture_and_match(tmp_path: Path) -> None:
    settings = _settings(tmp_path, template_memory_enabled=False)
    src = _make_job(tmp_path, "11111111-1111-4111-8111-111111111111", pages=1)
    (src / "field_grounding").mkdir(parents=True, exist_ok=True)
    (src / "field_grounding" / "page_0001.fields.json").write_text(
        json.dumps(_corrected_fields(0)), encoding="utf-8"
    )
    assert capture_corrected_page(src, 0, settings) is False
    assert match_templates_for_job(src, settings) == {}


# --------------------------------------------------------------------------- zero-LLM reuse (T042)


class _CountingOpenAI:
    """Fake OpenAI client that counts construction and create calls."""

    constructions = 0
    creates = 0

    def __init__(self, **_: Any) -> None:
        type(self).constructions += 1

        def _create(**kwargs: Any) -> Any:
            type(self).creates += 1
            content = json.dumps(
                {
                    "page_index": 0,
                    "width": 500,
                    "height": 300,
                    "unit": "px",
                    "origin": "top-left",
                    "fields": [
                        {
                            "field_id": "llm",
                            "type": "text",
                            "bbox": {"x": 150, "y": 110, "w": 100, "h": 30},
                            "confidence": 0.9,
                            "evidence": {"label": "L", "line_ids": ["line_h_001"]},
                        }
                    ],
                }
            )
            choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")
            return SimpleNamespace(choices=[choice], usage=None)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


def test_fully_templated_reupload_makes_zero_llm_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    # Capture a corrected page.
    src = _make_job(tmp_path, "11111111-1111-4111-8111-111111111111", pages=1)
    (src / "field_grounding").mkdir(parents=True, exist_ok=True)
    (src / "field_grounding" / "page_0001.fields.json").write_text(
        json.dumps(_corrected_fields(0)), encoding="utf-8"
    )
    assert capture_corrected_page(src, 0, settings) is True

    # Re-upload the same form.
    dst = _make_job(tmp_path, "22222222-2222-4222-8222-222222222222", pages=1)
    matches = match_templates_for_job(dst, settings)
    assert 0 in matches

    _CountingOpenAI.constructions = 0
    _CountingOpenAI.creates = 0
    monkeypatch.setattr(sg, "OpenAI", lambda **kw: _CountingOpenAI(**kw))

    # No API key configured — a fully templated job must not need one.
    summary = sg.run_semantic_grounding_for_job(
        job_id="22222222-2222-4222-8222-222222222222",
        output_dir=dst,
        settings=settings,
        provider="openai",
        model="gpt-test",
        template_page_results=matches,
    )

    assert _CountingOpenAI.constructions == 0
    assert _CountingOpenAI.creates == 0
    assert summary["succeeded_count"] == 1
    data = json.loads((dst / "field_grounding" / "page_0001.fields.json").read_text())
    assert data["fields"][0]["grounding_source"] == "template"
    assert data["fields"][0]["field_id"] == "name"
    assert read_job_manifest(dst.parent)["status"] == "ready"


def test_partial_template_grounds_only_unmatched_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path, openai_api_key="test-key")
    src = _make_job(tmp_path, "11111111-1111-4111-8111-111111111111", pages=1)
    (src / "field_grounding").mkdir(parents=True, exist_ok=True)
    (src / "field_grounding" / "page_0001.fields.json").write_text(
        json.dumps(_corrected_fields(0)), encoding="utf-8"
    )
    assert capture_corrected_page(src, 0, settings) is True

    # Two-page job: page 0 matches the template; page 1 has a different layout → LLM.
    dst = _make_job(tmp_path, "22222222-2222-4222-8222-222222222222", pages=2)
    ld_path = dst / "line_detection" / "page_0002" / "detected_lines.json"
    detected = json.loads(ld_path.read_text())
    detected["lines"][2]["bbox"]["x"] = 250
    ld_path.write_text(json.dumps(detected), encoding="utf-8")

    matches = match_templates_for_job(dst, settings)
    assert set(matches) == {0}

    _CountingOpenAI.constructions = 0
    _CountingOpenAI.creates = 0
    monkeypatch.setattr(sg, "OpenAI", lambda **kw: _CountingOpenAI(**kw))

    summary = sg.run_semantic_grounding_for_job(
        job_id="22222222-2222-4222-8222-222222222222",
        output_dir=dst,
        settings=settings,
        provider="openai",
        model="gpt-test",
        template_page_results=matches,
    )

    assert _CountingOpenAI.creates == 1  # only the unmatched page hit the model
    assert summary["succeeded_count"] == 2
    page0 = json.loads((dst / "field_grounding" / "page_0001.fields.json").read_text())
    page1 = json.loads((dst / "field_grounding" / "page_0002.fields.json").read_text())
    assert page0["fields"][0]["grounding_source"] == "template"
    assert page1["fields"][0]["field_id"] == "llm"
