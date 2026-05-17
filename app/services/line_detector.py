"""Deterministic OpenCV morphology-based horizontal/vertical form line detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, int] = {
    "min_horizontal_length_px": 45,
    "min_vertical_length_px": 45,
    "max_horizontal_thickness_px": 12,
    "max_vertical_thickness_px": 12,
    "horizontal_kernel_divisor": 35,
    "vertical_kernel_divisor": 35,
    "min_dotted_horizontal_length_px": 80,
    "max_dotted_line_thickness_px": 8,
    "dotted_horizontal_connect_width_px": 18,
    "dotted_horizontal_close_iterations": 2,
}

_CONFIG_KEYS = frozenset(DEFAULT_CONFIG.keys())


def _merge_config(config: dict[str, Any] | None) -> dict[str, int]:
    merged = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - _CONFIG_KEYS
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        merged.update(config)
    out: dict[str, int] = {}
    for k, v in merged.items():
        if not isinstance(v, int):
            raise ValueError(f"config[{k!r}] must be int, got {type(v).__name__}")
        if v < 1:
            raise ValueError(f"config[{k!r}] must be >= 1, got {v}")
        out[k] = v
    return out


def _x_interval_overlap_ratio(x1_a: int, x2_a: int, x1_b: int, x2_b: int) -> float:
    """Overlap length divided by the shorter span; 0 if disjoint."""
    left = max(x1_a, x1_b)
    right = min(x2_a, x2_b)
    if right <= left:
        return 0.0
    overlap = right - left
    span_a = max(1, x2_a - x1_a)
    span_b = max(1, x2_b - x1_b)
    return overlap / min(span_a, span_b)


def _filter_dotted_overlapping_solid(
    dotted_lines: list[dict[str, Any]],
    solid_horizontal_lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop dotted candidates that substantially overlap a solid horizontal line."""
    if not solid_horizontal_lines:
        return dotted_lines

    kept: list[dict[str, Any]] = []
    for dotted in dotted_lines:
        y_mid_d = dotted["y1"]
        thickness_d = dotted["thickness"]
        overlaps_solid = False
        for solid in solid_horizontal_lines:
            y_mid_s = solid["y1"]
            thickness_s = solid["thickness"]
            if abs(y_mid_d - y_mid_s) > max(thickness_d, thickness_s) + 2:
                continue
            if _x_interval_overlap_ratio(dotted["x1"], dotted["x2"], solid["x1"], solid["x2"]) >= 0.5:
                overlaps_solid = True
                break
        if not overlaps_solid:
            kept.append(dotted)
    return kept


def _contours_to_horizontal_lines(
    contours: list[Any],
    *,
    min_length: int,
    max_thickness: int,
) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    import cv2  # noqa: PLC0415

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < 1 or w < min_length or h > max_thickness:
            continue
        y_mid = y + h // 2
        lines.append(
            {
                "orientation": "horizontal",
                "line_style": "solid",
                "x1": int(x),
                "y1": int(y_mid),
                "x2": int(x + w),
                "y2": int(y_mid),
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "thickness": int(h),
            }
        )
    return lines


def _contours_to_vertical_lines(
    contours: list[Any],
    *,
    min_length: int,
    max_thickness: int,
) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    import cv2  # noqa: PLC0415

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 1 or h < min_length or w > max_thickness:
            continue
        x_mid = x + w // 2
        lines.append(
            {
                "orientation": "vertical",
                "line_style": "solid",
                "x1": int(x_mid),
                "y1": int(y),
                "x2": int(x_mid),
                "y2": int(y + h),
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "thickness": int(w),
            }
        )
    return lines


def _detect_dotted_horizontal_lines(binary: Any, cfg: dict[str, int]) -> list[dict[str, Any]]:
    """
    Detect dotted/dashed horizontal lines on THRESH_BINARY_INV image (ink = white).

    Uses a small horizontal open to keep dashes, then horizontal close to bridge gaps.
    """
    import cv2  # noqa: PLC0415

    min_length = cfg["min_dotted_horizontal_length_px"]
    max_thickness = cfg["max_dotted_line_thickness_px"]
    connect_width = cfg["dotted_horizontal_connect_width_px"]
    close_iterations = cfg["dotted_horizontal_close_iterations"]

    small_horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    dotted_seed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, small_horizontal_kernel, iterations=1)

    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (connect_width, 1))
    dotted_connected = cv2.morphologyEx(
        dotted_seed,
        cv2.MORPH_CLOSE,
        connect_kernel,
        iterations=close_iterations,
    )
    dotted_connected = cv2.dilate(
        dotted_connected,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)),
        iterations=1,
    )

    contours, _ = cv2.findContours(dotted_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dotted_lines: list[dict[str, Any]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_length or h > max_thickness or h < 1:
            continue
        y_mid = y + h // 2
        dotted_lines.append(
            {
                "orientation": "horizontal",
                "line_style": "dotted_or_dashed",
                "x1": int(x),
                "y1": int(y_mid),
                "x2": int(x + w),
                "y2": int(y_mid),
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "thickness": int(h),
            }
        )
    return dotted_lines


def _draw_debug_overlay(
    bgr: Any,
    solid_horizontal_lines: list[dict[str, Any]],
    dotted_horizontal_lines: list[dict[str, Any]],
    vertical_lines: list[dict[str, Any]],
) -> Any:
    """Draw centerlines: red = solid horizontal, green = dotted, blue = vertical (BGR)."""
    import cv2  # noqa: PLC0415

    out = bgr.copy()
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    thick = 2
    for ln in solid_horizontal_lines:
        cv2.line(out, (ln["x1"], ln["y1"]), (ln["x2"], ln["y2"]), red, thick, lineType=cv2.LINE_AA)
    for ln in dotted_horizontal_lines:
        cv2.line(out, (ln["x1"], ln["y1"]), (ln["x2"], ln["y2"]), green, thick, lineType=cv2.LINE_AA)
    for ln in vertical_lines:
        cv2.line(out, (ln["x1"], ln["y1"]), (ln["x2"], ln["y2"]), blue, thick, lineType=cv2.LINE_AA)
    return out


def compute_raw_line_detection(
    input_image_path: str,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    """
    Run morphology line detection without writing files.

    Returns ``(raw_payload, bgr_image)`` where ``raw_payload`` matches the JSON
    written by :func:`detect_form_lines` (image, detector, counts, lines).
    """
    import cv2  # noqa: PLC0415

    cfg = _merge_config(config)
    in_path = Path(input_image_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    bgr = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image: {input_image_path}")

    height, width = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal_kernel_len = max(25, width // cfg["horizontal_kernel_divisor"])
    vertical_kernel_len = max(25, height // cfg["vertical_kernel_divisor"])

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_len, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))

    horizontal_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    horizontal_mask = cv2.dilate(
        horizontal_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)),
        iterations=1,
    )
    vertical_mask = cv2.dilate(
        vertical_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)),
        iterations=1,
    )

    h_contours, _ = cv2.findContours(horizontal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_contours, _ = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_lines = _contours_to_horizontal_lines(
        h_contours,
        min_length=cfg["min_horizontal_length_px"],
        max_thickness=cfg["max_horizontal_thickness_px"],
    )
    v_lines = _contours_to_vertical_lines(
        v_contours,
        min_length=cfg["min_vertical_length_px"],
        max_thickness=cfg["max_vertical_thickness_px"],
    )

    dotted_h_lines = _detect_dotted_horizontal_lines(binary, cfg)
    dotted_h_lines = _filter_dotted_overlapping_solid(dotted_h_lines, h_lines)

    lines = [*h_lines, *v_lines, *dotted_h_lines]
    path_str = str(in_path.resolve())

    payload: dict[str, Any] = {
        "image": {
            "path": path_str,
            "width": int(width),
            "height": int(height),
            "unit": "px",
            "origin": "top-left",
        },
        "detector": {
            "method": "opencv_morphology",
            "config": dict(cfg),
        },
        "counts": {
            "horizontal": len(h_lines),
            "horizontal_dotted": len(dotted_h_lines),
            "vertical": len(v_lines),
            "total": len(lines),
        },
        "lines": lines,
    }
    return payload, bgr


def write_raw_line_detection(payload: dict[str, Any], bgr: Any, output_json_path: str, output_image_path: str) -> None:
    """Persist raw detector JSON and debug overlay PNG (red/green/blue)."""
    import cv2  # noqa: PLC0415
    import json as _json

    solid_h_lines = [
        ln
        for ln in payload["lines"]
        if ln["orientation"] == "horizontal" and ln.get("line_style", "solid") == "solid"
    ]
    dotted_h_lines = [
        ln
        for ln in payload["lines"]
        if ln["orientation"] == "horizontal" and ln.get("line_style") == "dotted_or_dashed"
    ]
    v_lines = [ln for ln in payload["lines"] if ln["orientation"] == "vertical"]
    out_img = Path(output_image_path)
    out_json = Path(output_json_path)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    overlay = _draw_debug_overlay(bgr, solid_h_lines, dotted_h_lines, v_lines)
    if not cv2.imwrite(str(out_img), overlay):
        raise OSError(f"Failed to write debug image: {output_image_path}")
    out_json.write_text(_json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def detect_form_lines(
    input_image_path: str,
    output_image_path: str,
    output_json_path: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Detect horizontal and vertical form lines via morphology on OTSU-inverted binary image.

    Writes ``output_json_path`` (UTF-8 JSON) and ``output_image_path`` (debug overlay PNG).
    Returns the same structure as written to JSON.
    """
    payload, bgr = compute_raw_line_detection(input_image_path, config)
    write_raw_line_detection(payload, bgr, output_json_path, output_image_path)
    return payload
