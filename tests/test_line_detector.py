from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.line_detector import DEFAULT_CONFIG, detect_form_lines


def test_detect_form_lines_synthetic_horizontal_and_vertical(tmp_path: Path) -> None:
    h, w = 400, 400
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # Horizontal stroke (black)
    cv2.rectangle(img, (40, 198), (360, 202), (0, 0, 0), thickness=-1)
    # Vertical stroke
    cv2.rectangle(img, (98, 40), (102, 360), (0, 0, 0), thickness=-1)

    in_path = tmp_path / "synthetic.png"
    out_json = tmp_path / "out.json"
    out_png = tmp_path / "highlight.png"
    cv2.imwrite(str(in_path), img)

    cfg = dict(DEFAULT_CONFIG)
    cfg["min_horizontal_length_px"] = 40
    cfg["min_vertical_length_px"] = 40

    result = detect_form_lines(str(in_path), str(out_png), str(out_json), cfg)

    assert out_json.is_file()
    assert out_png.is_file()
    disk = json.loads(out_json.read_text(encoding="utf-8"))
    assert disk == result

    assert result["image"]["width"] == w
    assert result["image"]["height"] == h
    assert result["image"]["unit"] == "px"
    assert result["image"]["origin"] == "top-left"
    assert result["detector"]["method"] == "opencv_morphology"
    assert result["counts"]["horizontal"] >= 1
    assert result["counts"]["vertical"] >= 1
    assert result["counts"]["total"] == result["counts"]["horizontal"] + result["counts"]["vertical"]

    for ln in result["lines"]:
        for k in ("x1", "y1", "x2", "y2"):
            assert 0 <= ln[k] < max(w, h), ln
        bb = ln["bbox"]
        assert bb["x"] >= 0 and bb["y"] >= 0
        assert bb["x"] + bb["w"] <= w
        assert bb["y"] + bb["h"] <= h


def test_detect_form_lines_rejects_unknown_config_key() -> None:
    with pytest.raises(ValueError, match="Unknown config keys"):
        detect_form_lines("nope.png", "a.png", "a.json", {"bad_key": 1})


def test_detect_form_lines_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        detect_form_lines(
            str(tmp_path / "missing.png"),
            str(tmp_path / "o.png"),
            str(tmp_path / "o.json"),
            None,
        )
