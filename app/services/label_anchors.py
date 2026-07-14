"""Extract printed label anchors from digital (text-layer) PDFs via PyMuPDF.

For non-scanned PDFs the page carries a real text layer. Passing label positions to the
model lets it name an anchor label instead of guessing pixel coordinates; the field bbox
is then computed deterministically from line geometry + the anchor position.

All returned coordinates are in top-left **pixel** space of the rendered page PNG, matching
the grounding coordinate system (PRD 2.2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

# Keep prompt payloads bounded on dense pages.
_MAX_ANCHORS_PER_PAGE = 120
_MAX_LABEL_CHARS = 80


def _manifest_scale(page_manifest: dict[str, Any]) -> tuple[float, float]:
    mapping = page_manifest.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError("Page manifest missing mapping.image_to_pdf_scale_*")
    sx = float(mapping["image_to_pdf_scale_x"])
    sy = float(mapping["image_to_pdf_scale_y"])
    if sx <= 0 or sy <= 0:
        raise ValueError("image_to_pdf_scale_* must be positive")
    return sx, sy


def _pt_to_px(value: float, scale: float) -> int:
    # scale converts px -> pt, so px = pt / scale.
    return int(round(value / scale))


def extract_label_anchors(
    *,
    input_pdf: Path,
    page_index: int,
    page_manifest: dict[str, Any],
    max_anchors: int = _MAX_ANCHORS_PER_PAGE,
) -> list[dict[str, Any]]:
    """Return printed label phrases as ``{"text", "bbox": {x,y,w,h}}`` in image pixels.

    Returns an empty list when the page has no usable text layer, is rotated, or PyMuPDF
    is unavailable — callers treat "no anchors" as a normal (scanned-PDF) case.
    """
    if not input_pdf.is_file():
        return []
    try:
        import fitz
    except ImportError:  # pragma: no cover - fitz is a hard dependency in this repo
        return []

    try:
        sx, sy = _manifest_scale(page_manifest)
    except (KeyError, TypeError, ValueError) as exc:
        LOG.warning("label anchors: bad manifest scale page=%d: %s", page_index, exc)
        return []

    anchors: list[dict[str, Any]] = []
    try:
        with fitz.open(input_pdf) as doc:
            if page_index < 0 or page_index >= doc.page_count:
                return []
            page = doc[page_index]
            if int(getattr(page, "rotation", 0) or 0) != 0:
                # Rotated pages are out of MVP scope; skip to avoid mismapped anchors.
                return []
            words = page.get_text("words") or []
    except Exception as exc:  # pragma: no cover - defensive against corrupt PDFs
        LOG.warning("label anchors: extraction failed page=%d: %s", page_index, exc)
        return []

    # words: (x0, y0, x1, y1, word, block_no, line_no, word_no) in PDF points (top-left).
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for w in words:
        if len(w) < 8:
            continue
        x0, y0, x1, y1, text, block_no, line_no, _ = w[:8]
        text = str(text).strip()
        if not text:
            continue
        key = (int(block_no), int(line_no))
        node = grouped.get(key)
        if node is None:
            grouped[key] = {
                "words": [text],
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
            }
        else:
            node["words"].append(text)
            node["x0"] = min(node["x0"], float(x0))
            node["y0"] = min(node["y0"], float(y0))
            node["x1"] = max(node["x1"], float(x1))
            node["y1"] = max(node["y1"], float(y1))

    for node in grouped.values():
        label = " ".join(node["words"]).strip()
        if not label:
            continue
        if len(label) > _MAX_LABEL_CHARS:
            label = label[:_MAX_LABEL_CHARS].rstrip()
        x_px = _pt_to_px(node["x0"], sx)
        y_px = _pt_to_px(node["y0"], sy)
        w_px = max(1, _pt_to_px(node["x1"] - node["x0"], sx))
        h_px = max(1, _pt_to_px(node["y1"] - node["y0"], sy))
        anchors.append({"text": label, "bbox": {"x": x_px, "y": y_px, "w": w_px, "h": h_px}})

    anchors.sort(key=lambda a: (a["bbox"]["y"], a["bbox"]["x"]))
    return anchors[:max_anchors]
