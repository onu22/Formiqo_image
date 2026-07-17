"""E7 template memory: reuse human-corrected grounding by detected-line fingerprint.

When a page's normalized detected-line layout matches a page a human previously corrected in
the editor, we reuse that corrected grounding outright and skip the LLM for that page
(``grounding_source: template``). The index is a small on-disk store under ``data/`` — no
database.

Coordinate system is unchanged: bboxes are top-left pixels on 200-DPI page PNGs (PRD §2.2).
Stored bboxes are scaled by the ratio of the new page's pixel dimensions to the template's,
so a re-render at a different DPI still lands correctly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

_PAGE_LINES_RE = re.compile(r"^page_(\d{4})$")
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------- #
# Fingerprint (T040)
# --------------------------------------------------------------------------- #


def _page_dims(detected_lines: dict[str, Any]) -> tuple[int, int]:
    """Return ``(width, height)`` from a detected_lines image node (px or legacy keys)."""
    image = detected_lines.get("image")
    if not isinstance(image, dict):
        return 0, 0
    width = image.get("width_px", image.get("width"))
    height = image.get("height_px", image.get("height"))
    try:
        return int(width), int(height)
    except (TypeError, ValueError):
        return 0, 0


def _line_tokens(detected_lines: dict[str, Any], *, bins: int) -> list[str]:
    """Scale-invariant quantized tokens, one per valid detected line."""
    width, height = _page_dims(detected_lines)
    if width <= 0 or height <= 0:
        return []
    lines = detected_lines.get("lines")
    if not isinstance(lines, list):
        return []
    tokens: set[str] = set()
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        bbox = ln.get("bbox")
        if not isinstance(bbox, dict):
            continue
        try:
            x = int(bbox["x"])
            y = int(bbox["y"])
            w = int(bbox["w"])
            h = int(bbox["h"])
        except (KeyError, TypeError, ValueError):
            continue
        orientation = ln.get("orientation")
        o = "h" if orientation == "horizontal" else "v" if orientation == "vertical" else "?"
        qx = round(x / width * bins)
        qy = round(y / height * bins)
        qw = round(w / width * bins)
        qh = round(h / height * bins)
        tokens.add(f"{o}:{qx},{qy},{qw},{qh}")
    return sorted(tokens)


def page_fingerprint(
    detected_lines: dict[str, Any],
    *,
    bins: int = 200,
    min_lines: int = 4,
) -> str | None:
    """Stable SHA-256 of a page's normalized detected-line layout.

    Returns ``None`` when the page has too few lines to fingerprint reliably (avoids
    false-positive matches on near-empty pages).
    """
    tokens = _line_tokens(detected_lines, bins=bins)
    if len(tokens) < max(1, min_lines):
        return None
    digest = hashlib.sha256("|".join(tokens).encode("utf-8")).hexdigest()
    return digest


# --------------------------------------------------------------------------- #
# On-disk index (T041)
# --------------------------------------------------------------------------- #


class TemplateStore:
    """Filesystem-backed fingerprint → corrected-grounding index under ``templates_dir``."""

    def __init__(self, templates_dir: Path) -> None:
        self.dir = Path(templates_dir)
        self.index_path = self.dir / "index.json"

    def _read_index(self) -> dict[str, Any]:
        if not self.index_path.is_file():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_index(self, index: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    def _payload_path(self, fingerprint: str) -> Path:
        if not _FINGERPRINT_RE.match(fingerprint):
            raise ValueError("fingerprint must be a 64-char hex digest")
        return self.dir / f"{fingerprint}.fields.json"

    def has(self, fingerprint: str) -> bool:
        return fingerprint in self._read_index() and self._payload_path(fingerprint).is_file()

    def get(self, fingerprint: str) -> dict[str, Any] | None:
        """Return the stored corrected page payload for ``fingerprint``, or ``None``."""
        if not _FINGERPRINT_RE.match(fingerprint):
            return None
        path = self._payload_path(fingerprint)
        if fingerprint not in self._read_index() or not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    def put(self, fingerprint: str, payload: dict[str, Any], *, meta: dict[str, Any]) -> None:
        """Store (or overwrite) a corrected page payload keyed by ``fingerprint``."""
        path = self._payload_path(fingerprint)
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        index = self._read_index()
        entry = dict(meta)
        entry["fingerprint"] = fingerprint
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        index[fingerprint] = entry
        self._write_index(index)


# --------------------------------------------------------------------------- #
# Reuse / capture helpers (T041/T042)
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _detected_lines_path(output_dir: Path, page_index: int) -> Path:
    return output_dir / "line_detection" / f"page_{page_index + 1:04d}" / "detected_lines.json"


def _fields_path(output_dir: Path, page_index: int) -> Path:
    return output_dir / "field_grounding" / f"page_{page_index + 1:04d}.fields.json"


def _page_manifest_dims(output_dir: Path, page_index: int) -> tuple[int, int] | None:
    manifest = _read_json(output_dir / "converted_images" / "pages" / f"page_{page_index + 1:04d}.json")
    if manifest is None:
        return None
    image = manifest.get("image")
    if not isinstance(image, dict):
        return None
    try:
        return int(image["width_px"]), int(image["height_px"])
    except (KeyError, TypeError, ValueError):
        return None


def _list_detected_line_pages(output_dir: Path) -> list[int]:
    ld_dir = output_dir / "line_detection"
    if not ld_dir.is_dir():
        return []
    pages: list[int] = []
    for child in sorted(ld_dir.iterdir()):
        if not child.is_dir():
            continue
        m = _PAGE_LINES_RE.match(child.name)
        if m and (child / "detected_lines.json").is_file():
            pages.append(int(m.group(1)) - 1)
    return sorted(pages)


def build_template_grounding_for_page(
    template_payload: dict[str, Any],
    *,
    page_index: int,
    width_px: int,
    height_px: int,
) -> dict[str, Any]:
    """Adapt a stored corrected page to the current page's dims and index.

    Field bboxes are scaled from the template's pixel space to the current page's, and every
    field is marked ``grounding_source: template``.
    """
    t_w = int(template_payload.get("width_px") or width_px) or width_px
    t_h = int(template_payload.get("height_px") or height_px) or height_px
    sx = width_px / t_w if t_w else 1.0
    sy = height_px / t_h if t_h else 1.0

    fields_out: list[dict[str, Any]] = []
    for field in template_payload.get("fields", []):
        if not isinstance(field, dict):
            continue
        out = dict(field)
        bbox = field.get("bbox")
        if isinstance(bbox, dict):
            try:
                x = float(bbox["x"]) * sx
                y = float(bbox["y"]) * sy
                w = float(bbox["w"]) * sx
                h = float(bbox["h"]) * sy
            except (KeyError, TypeError, ValueError):
                x = y = w = h = None  # type: ignore[assignment]
            if x is not None:
                nx = max(0, min(int(round(x)), width_px - 1))
                ny = max(0, min(int(round(y)), height_px - 1))
                nw = max(1, min(int(round(w)), width_px - nx))
                nh = max(1, min(int(round(h)), height_px - ny))
                out["bbox"] = {"x": nx, "y": ny, "w": nw, "h": nh}
        out["grounding_source"] = "template"
        fields_out.append(out)

    return {
        "page_index": page_index,
        "width_px": width_px,
        "height_px": height_px,
        "unit": "px",
        "origin": "top-left",
        "fields": fields_out,
    }


def match_templates_for_job(output_dir: Path, settings: Any) -> dict[int, dict[str, Any]]:
    """Return ``{page_index: template_grounding}`` for pages whose fingerprint is known.

    Best-effort: a read/parse error for one page never blocks grounding of the others.
    """
    if not getattr(settings, "template_memory_enabled", False):
        return {}
    store = TemplateStore(Path(settings.templates_dir))
    bins = int(settings.template_fingerprint_bins)
    min_lines = int(settings.template_min_lines)

    matches: dict[int, dict[str, Any]] = {}
    for page_index in _list_detected_line_pages(output_dir):
        detected = _read_json(_detected_lines_path(output_dir, page_index))
        if detected is None:
            continue
        try:
            fp = page_fingerprint(detected, bins=bins, min_lines=min_lines)
        except Exception as exc:  # noqa: BLE001 - never fail grounding on fingerprinting
            LOG.warning("fingerprint failed page=%d: %s", page_index, exc)
            continue
        if fp is None:
            continue
        payload = store.get(fp)
        if payload is None:
            continue
        dims = _page_manifest_dims(output_dir, page_index)
        if dims is not None:
            width_px, height_px = dims
        else:
            width_px, height_px = _page_dims(detected)
        if width_px <= 0 or height_px <= 0:
            continue
        matches[page_index] = build_template_grounding_for_page(
            payload, page_index=page_index, width_px=width_px, height_px=height_px
        )
        LOG.info("template match page=%d fingerprint=%s", page_index, fp[:12])
    return matches


def capture_corrected_page(output_dir: Path, page_index: int, settings: Any) -> bool:
    """Store the current corrected grounding of one page as a reusable template.

    Returns ``True`` when a template was written. Best-effort and side-effect free on
    failure; callers (the editor save hook) must never let this break the save.
    """
    if not getattr(settings, "template_memory_enabled", False):
        return False
    detected = _read_json(_detected_lines_path(output_dir, page_index))
    if detected is None:
        return False
    fp = page_fingerprint(
        detected,
        bins=int(settings.template_fingerprint_bins),
        min_lines=int(settings.template_min_lines),
    )
    if fp is None:
        return False
    fields_payload = _read_json(_fields_path(output_dir, page_index))
    if fields_payload is None or not isinstance(fields_payload.get("fields"), list):
        return False

    dims = _page_manifest_dims(output_dir, page_index)
    if dims is not None:
        width_px, height_px = dims
    else:
        width_px, height_px = (
            int(fields_payload.get("width_px") or 0),
            int(fields_payload.get("height_px") or 0),
        )

    payload = {
        "page_index": page_index,
        "width_px": width_px,
        "height_px": height_px,
        "unit": "px",
        "origin": "top-left",
        "fields": fields_payload["fields"],
    }
    source_job_id = None
    job_manifest = _read_json(output_dir.parent / "job.json")
    if isinstance(job_manifest, dict):
        source_job_id = job_manifest.get("job_id")

    store = TemplateStore(Path(settings.templates_dir))
    store.put(
        fp,
        payload,
        meta={
            "source_job_id": source_job_id,
            "page_index": page_index,
            "width_px": width_px,
            "height_px": height_px,
            "field_count": len(fields_payload["fields"]),
        },
    )
    LOG.info("captured template page=%d fingerprint=%s", page_index, fp[:12])
    return True


def capture_corrected_pages(
    output_dir: Path,
    settings: Any,
    *,
    page_indices: list[int] | None = None,
) -> int:
    """Capture templates for the given page indices (default: all with detected lines)."""
    if not getattr(settings, "template_memory_enabled", False):
        return 0
    targets = page_indices if page_indices is not None else _list_detected_line_pages(output_dir)
    captured = 0
    for page_index in targets:
        try:
            if capture_corrected_page(output_dir, page_index, settings):
                captured += 1
        except Exception as exc:  # noqa: BLE001 - capture is best-effort
            LOG.warning("template capture failed page=%d: %s", page_index, exc)
    return captured
