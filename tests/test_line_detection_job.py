from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.line_detection_job import (
    NO_CONVERTED_PAGE_PNGS,
    list_converted_page_pngs,
    run_detect_form_lines_for_job_output_dir,
)


def test_list_converted_page_pngs_missing_dir(tmp_path: Path) -> None:
    out = tmp_path / "job_out"
    out.mkdir()
    with pytest.raises(FileNotFoundError, match="converted_images"):
        list_converted_page_pngs(out)


def test_run_detect_form_lines_empty_glob(tmp_path: Path) -> None:
    out = tmp_path / "job_out"
    (out / "converted_images").mkdir(parents=True)
    with pytest.raises(ValueError, match=NO_CONVERTED_PAGE_PNGS):
        run_detect_form_lines_for_job_output_dir(
            job_id="test-job",
            output_dir=out,
            detector_config={
                "min_horizontal_length_px": 40,
                "min_vertical_length_px": 40,
                "max_horizontal_thickness_px": 12,
                "max_vertical_thickness_px": 12,
                "horizontal_kernel_divisor": 35,
                "vertical_kernel_divisor": 35,
            },
        )


def test_run_detect_form_lines_one_page(tmp_path: Path) -> None:
    out = tmp_path / "job_out"
    img_dir = out / "converted_images"
    img_dir.mkdir(parents=True)
    h, w = 200, 200
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (20, 98), (180, 102), (0, 0, 0), thickness=-1)
    cv2.rectangle(img, (98, 20), (102, 180), (0, 0, 0), thickness=-1)
    cv2.imwrite(str(img_dir / "page_0001.png"), img)

    result = run_detect_form_lines_for_job_output_dir(
        job_id="j1",
        output_dir=out,
        detector_config={
            "min_horizontal_length_px": 30,
            "min_vertical_length_px": 30,
            "max_horizontal_thickness_px": 12,
            "max_vertical_thickness_px": 12,
            "horizontal_kernel_divisor": 35,
            "vertical_kernel_divisor": 35,
        },
    )
    assert result.job_id == "j1"
    assert result.page_count == 1
    assert (out / "line_detection" / "page_0001" / "detected_lines.json").is_file()
    assert (out / "line_detection" / "page_0001" / "lines_highlighted.png").is_file()
    assert result.pages[0].detection.counts.total >= 1
