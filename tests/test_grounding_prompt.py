"""Tests for grounding prompt assembly and response adaptation."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.grounding_prompt import (
    _compact_json,
    adapt_grounding_response,
    build_openai_messages,
    build_user_text_bundle,
    line_detection_payload_for_prompt,
    load_grounding_developer_prompt,
    slim_line_detection_for_prompt,
)

_REPO = Path(__file__).resolve().parents[1]
_FORBIDDEN_STEMS = ("page_0001", "lines_highlighted", "converted_images/")


def test_prompt_files_exist_and_no_hardcoded_stems() -> None:
    for name in ("grounding_system.md", "grounding_developer.md", "grounding_user.md"):
        text = (_REPO / "prompts" / name).read_text(encoding="utf-8")
        for stem in _FORBIDDEN_STEMS:
            assert stem not in text, f"{name} must not contain {stem!r}"
    assert "page_NNN.json" in load_grounding_developer_prompt()
    assert "detected lines.json" in load_grounding_developer_prompt()


def test_user_bundle_uses_role_labels() -> None:
    detected = {"image": {"width": 100, "height": 100}, "lines": []}
    manifest = {"page_index": 0, "image": {"saved_image_width_px": 100, "saved_image_height_px": 100}}
    text = build_user_text_bundle(detected_lines=detected, page_manifest=manifest)
    assert "highlighted_line_image" in text
    assert "line_detection_json:" in text
    assert "page_metadata_json:" in text
    assert "geometry summary" not in text.lower()
    for stem in _FORBIDDEN_STEMS:
        assert stem not in text


def test_openai_messages_use_highlighted_image(tmp_path: Path) -> None:
    png = tmp_path / "highlighted.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    detected = {"image": {"width": 10, "height": 10}, "lines": []}
    manifest = {"page_index": 0, "image": {"saved_image_width_px": 10, "saved_image_height_px": 10}}
    messages = build_openai_messages(
        highlighted_png=png,
        detected_lines=detected,
        page_manifest=manifest,
    )
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "developer"
    user = messages[2]["content"]
    assert isinstance(user, list)
    assert user[1]["type"] == "image_url"
    assert "detected_lines.json:\n" not in user[0]["text"]


def test_adapt_grounding_response_maps_llm_schema() -> None:
    raw = {
        "page_index": 0,
        "width": 100,
        "height": 100,
        "fields": [
            {
                "field_id": "notes",
                "type": "textarea",
                "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
                "confidence": 0.9,
                "evidence": {"label": "Notes", "line_ids": ["line_h_001"]},
            },
            {
                "field_id": "cell_a",
                "type": "table_cell",
                "bbox": {"x": 5, "y": 6, "w": 7, "h": 8},
                "confidence": 0.8,
                "evidence": {"label": "", "line_ids": []},
            },
        ],
    }
    out = adapt_grounding_response(raw)
    assert out["fields"][0]["type"] == "multiline_text"
    assert out["fields"][0]["supporting_lines"] == ["line_h_001"]
    assert out["fields"][0]["label"] == "Notes"
    assert out["fields"][1]["type"] == "text"
    assert out["fields"][1]["field_surface"] == "solid_box"


def _full_line_fixture() -> dict:
    return {
        "image": {"width": 100, "height": 100, "path": "converted_images/page_0001.png"},
        "detector": {"method": "opencv_morphology", "config": {"min_horizontal_length_px": 45}},
        "counts": {"horizontal": 1, "vertical": 0, "total": 1},
        "lines": [
            {
                "orientation": "horizontal",
                "x1": 10,
                "y1": 20,
                "x2": 50,
                "y2": 20,
                "bbox": {"x": 10, "y": 19, "w": 40, "h": 2},
                "thickness": 2,
                "line_id": "line_h_001",
                "line_style": "solid",
            }
        ],
    }


def test_slim_line_detection_strips_metadata() -> None:
    full = _full_line_fixture()
    slim = slim_line_detection_for_prompt(full)
    assert "detector" not in slim
    assert "counts" not in slim
    assert slim["width"] == 100
    assert slim["height"] == 100
    assert len(slim["lines"]) == 1
    line = slim["lines"][0]
    assert set(line.keys()) == {"line_id", "orientation", "bbox"}
    assert line["line_id"] == "line_h_001"
    assert "x1" not in line


def test_user_bundle_uses_slim_lines() -> None:
    full = _full_line_fixture()
    manifest = {"page_index": 0, "image": {"saved_image_width_px": 100, "saved_image_height_px": 100}}
    text = build_user_text_bundle(detected_lines=full, page_manifest=manifest, slim_line_detection=True)
    prefix = "line_detection_json:\n"
    start = text.index(prefix) + len(prefix)
    end = text.index("\n\npage_metadata_json:")
    payload = json.loads(text[start:end])
    assert "detector" not in payload
    assert payload["lines"][0]["line_id"] == "line_h_001"
    assert "thickness" not in payload["lines"][0]


def test_slim_reduces_size_vs_full_payload() -> None:
    full = _full_line_fixture()
    for i in range(2, 61):
        full["lines"].append(
            {
                "orientation": "horizontal",
                "x1": 10 + i,
                "y1": 20 + i,
                "x2": 50 + i,
                "y2": 20 + i,
                "bbox": {"x": 10 + i, "y": 19 + i, "w": 40, "h": 2},
                "thickness": 2,
                "line_id": f"line_h_{i:03d}",
                "line_style": "solid",
            }
        )
    slim = line_detection_payload_for_prompt(full, slim=True)
    assert len(_compact_json(slim)) < len(_compact_json(full))


def test_slim_disabled_sends_full_payload() -> None:
    full = _full_line_fixture()
    payload = line_detection_payload_for_prompt(full, slim=False)
    assert "detector" in payload


def test_compact_json_in_bundle() -> None:
    text = build_user_text_bundle(
        detected_lines={"lines": []},
        page_manifest={"page_index": 0},
        compact_json=True,
    )
    assert "compact JSON" in text
    assert "\n" not in json.dumps({"lines": []}, separators=(",", ":"))
