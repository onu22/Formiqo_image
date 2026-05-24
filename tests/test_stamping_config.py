"""Tests for stamping.json sample generation."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.stamping_config import build_stamping_json_sample, write_stamping_json_sample


def test_build_stamping_json_sample_from_fields(tmp_path: Path) -> None:
    fg = tmp_path / "field_grounding"
    fg.mkdir()
    (fg / "page_0001.fields.json").write_text(
        json.dumps(
            {
                "page_index": 0,
                "fields": [
                    {"field_id": "first_name", "type": "text", "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}},
                    {"field_id": "agree_terms", "type": "checkbox", "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}},
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = build_stamping_json_sample(fg)
    assert payload["values"]["first_name"] == "first_name"
    assert payload["values"]["agree_terms"] == "true"
    assert payload["require_all_values"] is False
    assert "image_style" in payload


def test_write_stamping_json_sample_overwrites(tmp_path: Path) -> None:
    fg = tmp_path / "field_grounding"
    fg.mkdir()
    (fg / "stamping.json").write_text('{"values": {}}', encoding="utf-8")
    (fg / "page_0001.fields.json").write_text(
        json.dumps({"fields": [{"field_id": "email", "type": "text", "bbox": {}}]}),
        encoding="utf-8",
    )

    write_stamping_json_sample(fg)
    data = json.loads((fg / "stamping.json").read_text(encoding="utf-8"))
    assert data["values"]["email"] == "email"
