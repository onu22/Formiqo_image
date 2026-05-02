"""Geometry for drawing a vector checkmark inside a rectangle (y increases downward)."""

from __future__ import annotations


def tick_points_in_rect(x0: float, y0: float, w: float, h: float) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Three points forming a ``✓``: short-leg start, inner vertex, long-leg end.

    Fractions are tuned for small square checkboxes; coordinates match image/PyMuPDF page space (top-left origin, y down).
    """
    x1 = x0 + 0.14 * w
    y1 = y0 + 0.52 * h
    x2 = x0 + 0.40 * w
    y2 = y0 + 0.76 * h
    x3 = x0 + 0.88 * w
    y3 = y0 + 0.22 * h
    return (x1, y1), (x2, y2), (x3, y3)


def tick_stroke_width_px(min_side: float) -> int:
    return max(2, min(int(round(min_side * 0.14)), max(2, int(min_side * 0.42))))


def tick_stroke_width_pt(min_side_pt: float) -> float:
    return max(0.75, min(min_side_pt * 0.14, min_side_pt * 0.35))
