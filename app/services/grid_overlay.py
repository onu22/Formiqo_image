"""Render a labeled coordinate grid onto a page image (pixel-fallback accuracy aid).

Overlaying ticks every ``spacing_px`` with printed pixel coordinates materially improves a
vision model's raw coordinate estimates for fields that cannot be anchored to a line/label.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOG = logging.getLogger(__name__)

_GRID_COLOR = (0, 128, 255, 90)
_LABEL_COLOR = (0, 90, 200, 220)


def build_grid_overlay_image(src_png: Path, dst_png: Path, *, spacing_px: int = 100) -> Path:
    """Write ``src_png`` with a labeled coordinate grid to ``dst_png``; return ``dst_png``.

    Falls back to copying the source unchanged if Pillow is unavailable so the pipeline
    never fails solely because the overlay could not be drawn.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover - Pillow is a hard dependency in this repo
        dst_png.write_bytes(src_png.read_bytes())
        return dst_png

    if spacing_px < 1:
        spacing_px = 100

    with Image.open(src_png) as base:
        rgba = base.convert("RGBA")
    width, height = rgba.size
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for x in range(0, width, spacing_px):
        draw.line([(x, 0), (x, height)], fill=_GRID_COLOR, width=1)
        if x > 0:
            draw.text((x + 2, 2), str(x), fill=_LABEL_COLOR)
    for y in range(0, height, spacing_px):
        draw.line([(0, y), (width, y)], fill=_GRID_COLOR, width=1)
        if y > 0:
            draw.text((2, y + 2), str(y), fill=_LABEL_COLOR)

    combined = Image.alpha_composite(rgba, overlay).convert("RGB")
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    combined.save(dst_png, format="PNG")
    return dst_png
