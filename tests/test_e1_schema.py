"""Tests for E1 schema deduplication helpers."""

from __future__ import annotations

from app.services.semantic_grounding import _field_for_stamp_storage


def test_field_for_stamp_storage_drops_duplicate_keys() -> None:
    raw = {
        "field_id": "field_001",
        "type": "text",
        "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
        "label": "Name",
        "nearby_label_text": "Name",
        "supporting_lines": ["line_h_1"],
        "field_surface": "solid_box",
        "evidence": {"line_ids": ["line_h_1"], "label": "Name"},
    }
    stored = _field_for_stamp_storage(raw)
    assert stored["label"] == "Name"
    assert "nearby_label_text" not in stored
    assert "supporting_lines" not in stored
    assert "field_surface" not in stored
    assert "label" not in stored["evidence"]
    assert stored["evidence"]["line_ids"] == ["line_h_1"]
    assert stored["reviewed"] is False
    assert stored["grounding_source"] is None
