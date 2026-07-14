"""E4 grounding accuracy: structured outputs, anchor-first bboxes, parallel grounding."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import app.services.semantic_grounding as sg
from app.config import Settings
from app.services.form_geometry import (
    apply_anchor_grounding,
    build_geometry_index,
    normalize_page_grounding,
    resolve_anchor_bbox,
)
from app.services.grid_overlay import build_grid_overlay_image
from app.services.grounding_schema import (
    GROUNDING_TOOL_NAME,
    anthropic_grounding_tool,
    grounding_response_schema,
    openai_response_format,
)
from app.services.jobs import new_job_manifest, read_job_manifest, write_job_manifest
from app.services.label_anchors import extract_label_anchors

_PNG_BYTES = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- schema


def test_openai_strict_schema_is_closed_and_fully_required() -> None:
    fmt = openai_response_format()
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"].keys())
    field_schema = schema["properties"]["fields"]["items"]
    assert field_schema["additionalProperties"] is False
    assert set(field_schema["required"]) == set(field_schema["properties"].keys())
    assert "anchor" in field_schema["properties"]


def test_anthropic_tool_uses_relaxed_schema() -> None:
    tool = anthropic_grounding_tool()
    assert tool["name"] == GROUNDING_TOOL_NAME
    schema = tool["input_schema"]
    assert schema["type"] == "object"
    # Relaxed schema does not force every optional field to be required.
    assert "fields" in schema["properties"]
    field_schema = schema["properties"]["fields"]["items"]
    assert field_schema["required"] == ["field_id", "type", "bbox"]


def test_relaxed_and_strict_share_field_vocabulary() -> None:
    strict = grounding_response_schema(strict=True)
    relaxed = grounding_response_schema(strict=False)
    strict_types = strict["properties"]["fields"]["items"]["properties"]["type"]["enum"]
    relaxed_types = relaxed["properties"]["fields"]["items"]["properties"]["type"]["enum"]
    assert strict_types == relaxed_types


# --------------------------------------------------------------------------- geometry


def _single_cell_lines() -> dict[str, Any]:
    return {
        "image": {"width": 500, "height": 300},
        "lines": [
            {"line_id": "line_h_001", "orientation": "horizontal", "bbox": {"x": 0, "y": 100, "w": 500, "h": 2}},
            {"line_id": "line_h_002", "orientation": "horizontal", "bbox": {"x": 0, "y": 200, "w": 500, "h": 2}},
            {"line_id": "line_v_001", "orientation": "vertical", "bbox": {"x": 100, "y": 0, "w": 2, "h": 300}},
            {"line_id": "line_v_002", "orientation": "vertical", "bbox": {"x": 400, "y": 0, "w": 2, "h": 300}},
        ],
    }


def test_resolve_cell_anchor_sets_source_cell() -> None:
    geometry = build_geometry_index(_single_cell_lines())
    field = {
        "field_id": "f",
        "type": "text",
        "bbox": {"x": 0, "y": 0, "w": 10, "h": 10},
        "anchor": {"kind": "cell", "line_ids": ["line_h_001", "line_h_002", "line_v_001", "line_v_002"]},
    }
    resolved = resolve_anchor_bbox(field, geometry, [], page_w=500, page_h=300, stamp_inset_px=2)
    assert resolved is not None
    bbox, source, lines = resolved
    assert source == "cell"
    assert bbox["x"] >= 100 and bbox["x"] + bbox["w"] <= 400
    assert "line_h_001" in lines


def test_resolve_line_anchor_sets_source_line_anchor() -> None:
    geometry = build_geometry_index(_single_cell_lines())
    field = {
        "field_id": "f",
        "type": "text",
        "bbox": {"x": 0, "y": 0, "w": 10, "h": 10},
        "anchor": {"kind": "line_anchor", "line_ids": ["line_h_001"]},
    }
    resolved = resolve_anchor_bbox(field, geometry, [], page_w=500, page_h=300, stamp_inset_px=2)
    assert resolved is not None
    bbox, source, _ = resolved
    assert source == "line_anchor"
    assert 100 <= bbox["y"] <= 200


def test_resolve_label_anchor_sets_source_label_anchor() -> None:
    geometry = build_geometry_index(_single_cell_lines())
    anchors = [{"text": "Full Name", "bbox": {"x": 10, "y": 120, "w": 60, "h": 14}}]
    field = {
        "field_id": "f",
        "type": "text",
        "bbox": {"x": 0, "y": 0, "w": 10, "h": 10},
        "anchor": {"kind": "label_anchor", "label": "Full Name:", "relation": "right_of"},
    }
    resolved = resolve_anchor_bbox(field, geometry, anchors, page_w=500, page_h=300, stamp_inset_px=2)
    assert resolved is not None
    bbox, source, _ = resolved
    assert source == "label_anchor"
    # Field sits to the right of the label.
    assert bbox["x"] >= 70


def test_apply_anchor_grounding_marks_and_normalize_preserves() -> None:
    geometry = build_geometry_index(_single_cell_lines())
    payload = {
        "page_index": 0,
        "fields": [
            {
                "field_id": "f",
                "type": "text",
                "bbox": {"x": 0, "y": 0, "w": 10, "h": 10},
                "anchor": {"kind": "cell", "line_ids": ["line_h_001", "line_h_002", "line_v_001", "line_v_002"]},
            }
        ],
    }
    anchored = apply_anchor_grounding(payload, geometry, [], page_w=500, page_h=300, stamp_inset_px=2)
    assert anchored["fields"][0]["_anchored"] is True
    assert anchored["fields"][0]["grounding_source"] == "cell"
    anchored_bbox = dict(anchored["fields"][0]["bbox"])
    normalized = normalize_page_grounding(anchored, geometry, stamp_inset_px=2, page_w=500, page_h=300)
    # Normalization must not move an already-anchored field.
    assert normalized["fields"][0]["bbox"] == anchored_bbox
    assert normalized["fields"][0]["grounding_source"] == "cell"


def test_normalize_marks_unanchored_pixel_field() -> None:
    geometry = build_geometry_index(_single_cell_lines())
    payload = {
        "page_index": 0,
        "fields": [
            {"field_id": "f", "type": "checkbox", "field_surface": "checkbox", "bbox": {"x": 5, "y": 5, "w": 8, "h": 8}}
        ],
    }
    out = normalize_page_grounding(payload, geometry, stamp_inset_px=2, page_w=500, page_h=300)
    assert out["fields"][0]["grounding_source"] == "pixel"


# --------------------------------------------------------------------------- grid overlay


def test_grid_overlay_preserves_dimensions(tmp_path: Path) -> None:
    from PIL import Image

    src = tmp_path / "page.png"
    Image.new("RGB", (240, 160), (255, 255, 255)).save(src)
    dst = build_grid_overlay_image(src, tmp_path / "grid.png", spacing_px=50)
    assert dst.is_file()
    with Image.open(dst) as img:
        assert img.size == (240, 160)


def test_extract_label_anchors_missing_pdf_returns_empty(tmp_path: Path) -> None:
    anchors = extract_label_anchors(
        input_pdf=tmp_path / "nope.pdf",
        page_index=0,
        page_manifest={"mapping": {"image_to_pdf_scale_x": 0.36, "image_to_pdf_scale_y": 0.36}},
    )
    assert anchors == []


# --------------------------------------------------------------------------- pipeline


def _write_page_fixture(output_dir: Path, page_index: int) -> None:
    stem = f"page_{page_index + 1:04d}"
    conv = output_dir / "converted_images"
    (conv / "pages").mkdir(parents=True, exist_ok=True)
    (conv / f"{stem}.png").write_bytes(_PNG_BYTES)
    (conv / "pages" / f"{stem}.json").write_text(
        json.dumps(
            {
                "page_index": page_index,
                "image": {"width_px": 500, "height_px": 300},
                "mapping": {"image_to_pdf_scale_x": 0.36, "image_to_pdf_scale_y": 0.36},
            }
        ),
        encoding="utf-8",
    )
    ld = output_dir / "line_detection" / stem
    ld.mkdir(parents=True, exist_ok=True)
    (ld / "lines_highlighted.png").write_bytes(_PNG_BYTES)
    (ld / "detected_lines.json").write_text(json.dumps(_single_cell_lines()), encoding="utf-8")


def _make_job(tmp_path: Path, pages: int) -> tuple[str, Path]:
    job_id = "11111111-1111-4111-8111-111111111111"
    root = tmp_path / job_id
    output_dir = root / "output"
    for i in range(pages):
        _write_page_fixture(output_dir, i)
    manifest = new_job_manifest(job_id=job_id, source_filename="form.pdf")
    manifest["page_count"] = pages
    write_job_manifest(root, manifest)
    return job_id, output_dir


def _structured_response(page_index: int) -> dict[str, Any]:
    return {
        "page_index": page_index,
        "width": 500,
        "height": 300,
        "unit": "px",
        "origin": "top-left",
        "fields": [
            {
                "field_id": "name",
                "type": "text",
                "bbox": {"x": 150, "y": 110, "w": 100, "h": 30},
                "confidence": 0.92,
                "anchor": {
                    "kind": "cell",
                    "line_ids": ["line_h_001", "line_h_002", "line_v_001", "line_v_002"],
                    "label": None,
                    "relation": None,
                },
                "evidence": {"label": "Name", "line_ids": ["line_h_001"]},
            }
        ],
    }


def _page_index_from_openai_messages(messages: list[dict[str, Any]]) -> int:
    text = messages[2]["content"][0]["text"]
    marker = "page_metadata_json:\n"
    start = text.index(marker) + len(marker)
    chunk = text[start:].split("\n\n")[0]
    return int(json.loads(chunk)["page_index"])


class _FakeOpenAI:
    def __init__(self, *, fail_pages: set[int] | None = None, **_: Any) -> None:
        fail_pages = fail_pages or set()

        def _create(**kwargs: Any) -> Any:
            page_index = _page_index_from_openai_messages(kwargs["messages"])
            if page_index in fail_pages:
                raise RuntimeError(f"simulated provider error page {page_index}")
            content = json.dumps(_structured_response(page_index))
            choice = SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
            return SimpleNamespace(choices=[choice], usage=None)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    base = dict(
        jobs_dir=tmp_path,
        openai_api_key="test-key",
        grounding_provider="openai",
        grounding_model="gpt-test",
        grounding_structured_outputs=True,
        grounding_max_concurrency=4,
        grounding_label_anchors=True,
    )
    base.update(overrides)
    return Settings(**base)


def test_parallel_grounding_all_pages_succeed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id, output_dir = _make_job(tmp_path, pages=3)
    monkeypatch.setattr(sg, "OpenAI", lambda **kw: _FakeOpenAI(**kw))

    summary = sg.run_semantic_grounding_for_job(
        job_id=job_id,
        output_dir=output_dir,
        settings=_settings(tmp_path),
        provider="openai",
        model="gpt-test",
    )

    assert summary["succeeded_count"] == 3
    assert summary["failed_count"] == 0
    for i in range(3):
        fields_file = output_dir / "field_grounding" / f"page_{i + 1:04d}.fields.json"
        data = json.loads(fields_file.read_text(encoding="utf-8"))
        field = data["fields"][0]
        assert field["grounding_source"] == "cell"
        assert "confidence" in field
        assert "anchor" not in field  # transient key dropped on storage
        assert 100 <= field["bbox"]["x"] <= 400

    manifest = read_job_manifest(output_dir.parent)
    assert manifest["status"] == "ready"
    assert manifest["stages"]["grounding"]["grounded_pages"] == 3


def test_per_page_error_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id, output_dir = _make_job(tmp_path, pages=3)
    monkeypatch.setattr(sg, "OpenAI", lambda **kw: _FakeOpenAI(fail_pages={1}, **kw))

    summary = sg.run_semantic_grounding_for_job(
        job_id=job_id,
        output_dir=output_dir,
        settings=_settings(tmp_path),
        provider="openai",
        model="gpt-test",
    )

    assert summary["succeeded_count"] == 2
    assert summary["failed_count"] == 1
    assert summary["failed_pages"][0]["page_index"] == 1
    # Surviving pages still produced field files.
    assert (output_dir / "field_grounding" / "page_0001.fields.json").is_file()
    assert (output_dir / "field_grounding" / "page_0003.fields.json").is_file()
    assert not (output_dir / "field_grounding" / "page_0002.fields.json").is_file()


def test_anthropic_tool_use_structured_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id, output_dir = _make_job(tmp_path, pages=1)

    def _extract_page_index(user_content: list[dict[str, Any]]) -> int:
        text = user_content[0]["text"]
        marker = "page_metadata_json:\n"
        start = text.index(marker) + len(marker)
        return int(json.loads(text[start:].split("\n\n")[0])["page_index"])

    class _FakeAnthropic:
        def __init__(self, **_: Any) -> None:
            def _create(**kwargs: Any) -> Any:
                assert kwargs.get("tools"), "structured Anthropic path must pass a tool"
                page_index = _extract_page_index(kwargs["messages"][0]["content"])
                block = SimpleNamespace(type="tool_use", input=_structured_response(page_index))
                return SimpleNamespace(content=[block], stop_reason="tool_use", usage=None)

            self.messages = SimpleNamespace(create=_create)

    monkeypatch.setattr(sg, "Anthropic", lambda **kw: _FakeAnthropic(**kw))

    summary = sg.run_semantic_grounding_for_job(
        job_id=job_id,
        output_dir=output_dir,
        settings=_settings(tmp_path, anthropic_api_key="test-key", grounding_provider="anthropic"),
        provider="anthropic",
        model="claude-test",
    )
    assert summary["succeeded_count"] == 1
    data = json.loads((output_dir / "field_grounding" / "page_0001.fields.json").read_text(encoding="utf-8"))
    assert data["fields"][0]["grounding_source"] == "cell"
