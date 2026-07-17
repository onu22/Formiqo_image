"""E7 template memory: reuse human-corrected grounding for repeat form layouts.

A page is fingerprinted by its *normalized detected-line layout* (a stable hash of the
line set from ``line_detector.py``). When a user corrects a job in the editor
(``PATCH /jobs/{id}/fields``), the corrected per-page fields are stored under
``data/templates/`` keyed by that fingerprint. On a later upload, any page whose
fingerprint matches reuses the stored fields outright and skips the vision LLM; those
fields are marked ``grounding_source: template``.

The store is a small JSON index plus one corrected-fields file per fingerprint — no
database, per PRD E7. Writes are serialized with a process-wide lock; the index is
rewritten atomically so a crashed writer never leaves a partial file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings

LOG = logging.getLogger(__name__)

_INDEX_LOCK = threading.Lock()
_INDEX_NAME = "index.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _page_dimensions(detected_lines: dict[str, Any]) -> tuple[float, float]:
    """Return ``(width, height)`` from a detected-lines payload (tolerates key variants)."""
    image = detected_lines.get("image")
    if not isinstance(image, dict):
        raise ValueError("detected_lines payload missing image object")
    width = image.get("width_px", image.get("width"))
    height = image.get("height_px", image.get("height"))
    if not width or not height:
        raise ValueError("detected_lines image missing width/height")
    return float(width), float(height)


def page_line_fingerprint(
    detected_lines: dict[str, Any],
    *,
    quantize: int,
    min_lines: int,
) -> str | None:
    """Stable hash of a page's normalized detected-line layout.

    Each line contributes an orientation token plus its center and length normalized to
    the page dimensions and snapped to a ``quantize``-cell grid. Normalization by page
    size makes the hash scale-invariant (DPI-independent); quantization absorbs sub-cell
    jitter while keeping genuinely different layouts distinct.

    Returns ``None`` when the page has fewer than ``min_lines`` lines — too sparse to
    fingerprint safely (near-empty pages would otherwise collide).
    """
    lines = detected_lines.get("lines")
    if not isinstance(lines, list):
        return None

    try:
        page_w, page_h = _page_dimensions(detected_lines)
    except ValueError:
        return None
    if page_w <= 0 or page_h <= 0:
        return None

    tokens: list[tuple[str, int, int, int]] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        bbox = line.get("bbox")
        orientation = line.get("orientation")
        if not isinstance(bbox, dict) or orientation not in ("horizontal", "vertical"):
            continue
        try:
            x = float(bbox["x"])
            y = float(bbox["y"])
            w = float(bbox["w"])
            h = float(bbox["h"])
        except (KeyError, TypeError, ValueError):
            continue
        cx = (x + w / 2.0) / page_w
        cy = (y + h / 2.0) / page_h
        length = (w / page_w) if orientation == "horizontal" else (h / page_h)
        tokens.append(
            (
                orientation[0],
                _snap(cx, quantize),
                _snap(cy, quantize),
                _snap(length, quantize),
            )
        )

    if len(tokens) < min_lines:
        return None

    tokens.sort()
    canonical = json.dumps({"q": quantize, "n": len(tokens), "lines": tokens}, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _snap(normalized: float, quantize: int) -> int:
    """Snap a 0..1 normalized coordinate to an integer grid cell (clamped to [0, quantize])."""
    cell = int(round(normalized * quantize))
    if cell < 0:
        return 0
    if cell > quantize:
        return quantize
    return cell


def _index_path(settings: Settings) -> Path:
    return Path(settings.templates_dir) / _INDEX_NAME


def _read_index(settings: Settings) -> dict[str, Any]:
    path = _index_path(settings)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover - defensive
        LOG.warning("template index unreadable at %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    entries = data.get("templates")
    return entries if isinstance(entries, dict) else {}


def _write_index(settings: Settings, entries: dict[str, Any]) -> None:
    path = _index_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "templates": entries}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _fields_file_path(settings: Settings, fingerprint: str) -> Path:
    return Path(settings.templates_dir) / f"{fingerprint}.fields.json"


def _read_detected_lines(output_dir: Path, page_index: int) -> dict[str, Any] | None:
    stem = f"page_{page_index + 1:04d}"
    path = output_dir / "line_detection" / stem / "detected_lines.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):  # pragma: no cover - defensive
        return None
    return data if isinstance(data, dict) else None


def _read_page_fields(output_dir: Path, page_index: int) -> dict[str, Any] | None:
    stem = f"page_{page_index + 1:04d}"
    path = output_dir / "field_grounding" / f"{stem}.fields.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):  # pragma: no cover - defensive
        return None
    return data if isinstance(data, dict) else None


def compute_page_fingerprint(
    *,
    settings: Settings,
    output_dir: Path,
    page_index: int,
) -> str | None:
    """Fingerprint the given 0-based page from its on-disk detected-lines file."""
    detected = _read_detected_lines(output_dir, page_index)
    if detected is None:
        return None
    return page_line_fingerprint(
        detected,
        quantize=settings.template_fingerprint_quantize,
        min_lines=settings.template_min_lines,
    )


def record_corrected_pages(
    *,
    settings: Settings,
    output_dir: Path,
    source_job_id: str,
    source_filename: str,
    page_numbers: list[int],
) -> list[str]:
    """Persist human-corrected pages as reusable templates keyed by line fingerprint.

    ``page_numbers`` are 1-based (as used by the fields API). Returns the fingerprints
    that were recorded. Best-effort: a page that cannot be fingerprinted (too few lines,
    missing files) is silently skipped so it never breaks the calling PATCH.
    """
    if not settings.template_memory_enabled:
        return []

    recorded: list[str] = []
    with _INDEX_LOCK:
        entries = _read_index(settings)
        for page_number in page_numbers:
            page_index = int(page_number) - 1
            if page_index < 0:
                continue
            fingerprint = compute_page_fingerprint(
                settings=settings,
                output_dir=output_dir,
                page_index=page_index,
            )
            if fingerprint is None:
                continue
            fields_payload = _read_page_fields(output_dir, page_index)
            if fields_payload is None:
                continue

            _fields_file_path(settings, fingerprint).parent.mkdir(parents=True, exist_ok=True)
            _fields_file_path(settings, fingerprint).write_text(
                json.dumps(fields_payload, indent=2) + "\n", encoding="utf-8"
            )
            entries[fingerprint] = {
                "fingerprint": fingerprint,
                "source_job_id": source_job_id,
                "source_filename": source_filename,
                "page_number": int(page_number),
                "field_count": len(fields_payload.get("fields", []) or []),
                "fields_file": f"{fingerprint}.fields.json",
                "updated_at": _utc_now_iso(),
            }
            recorded.append(fingerprint)
        if recorded:
            _write_index(settings, entries)
    return recorded


def _load_template_fields(settings: Settings, fingerprint: str) -> dict[str, Any] | None:
    entries = _read_index(settings)
    entry = entries.get(fingerprint)
    if not isinstance(entry, dict):
        return None
    fields_file = entry.get("fields_file")
    if not isinstance(fields_file, str) or not fields_file:
        return None
    path = Path(settings.templates_dir) / fields_file
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):  # pragma: no cover - defensive
        return None
    return data if isinstance(data, dict) else None


def load_template_grounding_for_page(
    *,
    settings: Settings,
    output_dir: Path,
    page_index: int,
) -> dict[str, Any] | None:
    """Return a grounding payload for ``page_index`` if its fingerprint matches a template.

    The returned payload matches the ``grounding`` shape consumed by
    ``write_field_grounding_outputs`` (``page_index``, ``width_px``/``height_px``,
    ``fields``). Every field is stamped ``grounding_source: template``. Returns ``None``
    when template memory is disabled or no fingerprint match exists.
    """
    if not settings.template_memory_enabled:
        return None
    fingerprint = compute_page_fingerprint(
        settings=settings,
        output_dir=output_dir,
        page_index=page_index,
    )
    if fingerprint is None:
        return None
    stored = _load_template_fields(settings, fingerprint)
    if stored is None:
        return None

    fields: list[dict[str, Any]] = []
    for field in stored.get("fields", []) or []:
        if not isinstance(field, dict):
            continue
        out = dict(field)
        out["grounding_source"] = "template"
        fields.append(out)

    return {
        "page_index": page_index,
        "width_px": stored.get("width_px", stored.get("width")),
        "height_px": stored.get("height_px", stored.get("height")),
        "unit": stored.get("unit", "px"),
        "origin": stored.get("origin", "top-left"),
        "fields": fields,
        "_from_template": True,
        "_template_fingerprint": fingerprint,
    }
