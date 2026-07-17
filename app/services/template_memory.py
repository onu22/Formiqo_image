"""E7 template memory: reuse human-corrected grounding on matching re-uploads.

A page is fingerprinted by its **normalized detected-line layout** (a stable hash of the
line set from :mod:`app.services.line_detector`). When a user saves field corrections in the
editor, that page's corrected fields are captured into a small on-disk index under
``data/templates`` keyed by the fingerprint. On a later upload, any page whose fingerprint
matches a stored template reuses the corrected grounding outright and skips the LLM for that
page, persisting ``grounding_source: template``.

No database: the index is ``templates/index.json`` (small metadata) plus one
``templates/<fingerprint>.fields.json`` payload per template. Fingerprints are hex SHA-256
digests, so they are always safe filesystem names; we still validate them defensively.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.config import Settings
from app.services.line_detection_job import list_converted_page_pngs

LOG = logging.getLogger(__name__)

_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_FINGERPRINT_VERSION = "v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quantize(value: float, size: float, bins: int) -> int:
    """Map ``value`` in ``[0, size]`` onto an integer bucket in ``[0, bins]``."""
    if size <= 0:
        return 0
    return int(round((float(value) / float(size)) * bins))


def normalized_line_tokens(detected_lines: dict[str, Any], *, bins: int) -> list[str]:
    """Return sorted, scale-normalized, quantized tokens for each detected line.

    Coordinates are normalized against the page image dimensions so the same layout at a
    different DPI/scale still produces identical tokens; quantizing to ``bins`` buckets
    absorbs sub-bucket jitter while keeping distinct layouts distinct.
    """
    image = detected_lines.get("image") or {}
    try:
        width = float(image.get("width") or 0)
        height = float(image.get("height") or 0)
    except (TypeError, ValueError):
        return []
    if width <= 0 or height <= 0:
        return []

    tokens: list[str] = []
    for line in detected_lines.get("lines") or []:
        if not isinstance(line, dict):
            continue
        bbox = line.get("bbox")
        if not isinstance(bbox, dict):
            continue
        orient = str(line.get("orientation", "?"))[:1] or "?"
        qx = _quantize(bbox.get("x", 0), width, bins)
        qy = _quantize(bbox.get("y", 0), height, bins)
        qw = _quantize(bbox.get("w", 0), width, bins)
        qh = _quantize(bbox.get("h", 0), height, bins)
        tokens.append(f"{orient}:{qx}:{qy}:{qw}:{qh}")
    tokens.sort()
    return tokens


def page_fingerprint(detected_lines: dict[str, Any], *, bins: int) -> str | None:
    """Deterministic SHA-256 fingerprint of a page's normalized line layout.

    Returns ``None`` when the page has no usable line geometry (so empty pages never match).
    """
    tokens = normalized_line_tokens(detected_lines, bins=bins)
    if not tokens:
        return None
    h_count = sum(1 for t in tokens if t[:1] == "h")
    v_count = sum(1 for t in tokens if t[:1] == "v")
    canonical = f"{_FINGERPRINT_VERSION}|H{h_count}|V{v_count}|" + "|".join(tokens)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _valid_fingerprint(fingerprint: str) -> bool:
    return bool(_FINGERPRINT_RE.match(fingerprint))


class TemplateStore:
    """Filesystem-backed template index (no database)."""

    def __init__(self, templates_dir: Path) -> None:
        self.templates_dir = Path(templates_dir)

    @property
    def index_path(self) -> Path:
        return self.templates_dir / "index.json"

    def _payload_path(self, fingerprint: str) -> Path:
        if not _valid_fingerprint(fingerprint):
            raise ValueError(f"Unsafe template fingerprint: {fingerprint!r}")
        return self.templates_dir / f"{fingerprint}.fields.json"

    def load_index(self) -> dict[str, Any]:
        path = self.index_path
        if not path.is_file():
            return {"version": 1, "templates": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            LOG.warning("template index unreadable; treating as empty: %s", path)
            return {"version": 1, "templates": {}}
        if not isinstance(data, dict):
            return {"version": 1, "templates": {}}
        templates = data.get("templates")
        if not isinstance(templates, dict):
            data["templates"] = {}
        data.setdefault("version", 1)
        return data

    def _save_index(self, index: dict[str, Any]) -> None:
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_name(self.index_path.name + ".tmp")
        tmp.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.index_path)

    def lookup(self, fingerprint: str) -> dict[str, Any] | None:
        """Return ``{"record", "payload"}`` for a stored template, or ``None``."""
        if not fingerprint or not _valid_fingerprint(fingerprint):
            return None
        index = self.load_index()
        record = index.get("templates", {}).get(fingerprint)
        if not isinstance(record, dict):
            return None
        payload_path = self._payload_path(fingerprint)
        if not payload_path.is_file():
            return None
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            LOG.warning("template payload unreadable for %s", fingerprint)
            return None
        if not isinstance(payload, dict):
            return None
        return {"record": record, "payload": payload}

    def upsert(
        self,
        fingerprint: str,
        *,
        source_job_id: str,
        source_page_number: int,
        width_px: Any,
        height_px: Any,
        line_count: int,
        fields: list[dict[str, Any]],
    ) -> None:
        """Insert or refresh a template payload + index record for ``fingerprint``."""
        if not _valid_fingerprint(fingerprint):
            raise ValueError(f"Unsafe template fingerprint: {fingerprint!r}")

        payload = {
            "page_index": 0,
            "width_px": width_px,
            "height_px": height_px,
            "unit": "px",
            "origin": "top-left",
            "fields": [copy.deepcopy(f) for f in fields if isinstance(f, dict)],
        }
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        payload_path = self._payload_path(fingerprint)
        payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        index = self.load_index()
        templates = index.setdefault("templates", {})
        existing = templates.get(fingerprint)
        created_at = existing.get("created_at") if isinstance(existing, dict) else None
        now = _utc_now_iso()
        templates[fingerprint] = {
            "source_job_id": source_job_id,
            "source_page_number": int(source_page_number),
            "created_at": created_at or now,
            "updated_at": now,
            "width_px": width_px,
            "height_px": height_px,
            "line_count": int(line_count),
            "field_count": len(payload["fields"]),
            "fields_file": payload_path.name,
        }
        self._save_index(index)


def _scale_bbox(bbox: dict[str, Any], sx: float, sy: float) -> dict[str, int]:
    return {
        "x": int(round(float(bbox.get("x", 0)) * sx)),
        "y": int(round(float(bbox.get("y", 0)) * sy)),
        "w": int(round(float(bbox.get("w", 0)) * sx)),
        "h": int(round(float(bbox.get("h", 0)) * sy)),
    }


def build_template_grounding_for_page(
    template_payload: dict[str, Any],
    *,
    page_index: int,
    cur_w: int,
    cur_h: int,
) -> dict[str, Any]:
    """Materialize a reusable grounding payload for a matched page.

    Bboxes are scaled from the template's page dimensions to the current page's dimensions so
    a matching layout captured at a different DPI still lands correctly (identity scale for the
    common same-form re-upload case). Every field is marked ``grounding_source: template``.
    """
    try:
        tpl_w = float(template_payload.get("width_px") or 0)
        tpl_h = float(template_payload.get("height_px") or 0)
    except (TypeError, ValueError):
        tpl_w = tpl_h = 0.0
    sx = (cur_w / tpl_w) if tpl_w > 0 else 1.0
    sy = (cur_h / tpl_h) if tpl_h > 0 else 1.0
    rescale = abs(sx - 1.0) > 1e-9 or abs(sy - 1.0) > 1e-9

    out_fields: list[dict[str, Any]] = []
    for field in template_payload.get("fields") or []:
        if not isinstance(field, dict):
            continue
        nf = copy.deepcopy(field)
        bbox = nf.get("bbox")
        if isinstance(bbox, dict) and rescale:
            nf["bbox"] = _scale_bbox(bbox, sx, sy)
        nf["grounding_source"] = "template"
        out_fields.append(nf)

    return {
        "page_index": page_index,
        "width_px": cur_w,
        "height_px": cur_h,
        "unit": "px",
        "origin": "top-left",
        "fields": out_fields,
    }


def _read_detected_lines(output_dir: Path, page_number: int) -> dict[str, Any] | None:
    path = output_dir / "line_detection" / f"page_{page_number:04d}" / "detected_lines.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def match_templates_for_job(
    *,
    output_dir: Path,
    settings: Settings,
) -> dict[int, dict[str, Any]]:
    """Return ``{page_index: grounding_payload}`` for every page matching a stored template.

    Pages with fewer than ``template_min_lines`` detected lines are never matched (guards
    against trivial/near-empty pages false-positive matching). One bad page never aborts the
    scan — matching is best-effort and additive to LLM grounding of the remaining pages.
    """
    if not settings.template_memory_enabled:
        return {}

    store = TemplateStore(settings.templates_dir)
    index = store.load_index()
    if not index.get("templates"):
        return {}

    results: dict[int, dict[str, Any]] = {}
    try:
        page_entries = list_converted_page_pngs(output_dir)
    except FileNotFoundError:
        return {}

    for page_index, _png in page_entries:
        page_number = page_index + 1
        detected = _read_detected_lines(output_dir, page_number)
        if detected is None:
            continue
        line_count = len(detected.get("lines") or [])
        if line_count < settings.template_min_lines:
            continue
        fingerprint = page_fingerprint(detected, bins=settings.template_fingerprint_bins)
        if not fingerprint:
            continue
        found = store.lookup(fingerprint)
        if not found:
            continue
        payload = found["payload"]
        image = detected.get("image") or {}
        try:
            cur_w = int(image.get("width") or payload.get("width_px") or 0)
            cur_h = int(image.get("height") or payload.get("height_px") or 0)
        except (TypeError, ValueError):
            continue
        if cur_w <= 0 or cur_h <= 0:
            continue
        results[page_index] = build_template_grounding_for_page(
            payload,
            page_index=page_index,
            cur_w=cur_w,
            cur_h=cur_h,
        )
        LOG.info(
            "template match page=%d fingerprint=%s source_job=%s",
            page_index,
            fingerprint[:12],
            found["record"].get("source_job_id"),
        )
    return results


def capture_corrected_page(
    *,
    output_dir: Path,
    settings: Settings,
    source_job_id: str,
    page_number: int,
) -> str | None:
    """Capture one editor-saved page's corrected fields as a reusable template.

    Only human-corrected, editor-saved fields become templates: this is invoked from the
    ``PATCH /jobs/{id}/fields`` path after a successful save. Returns the fingerprint stored,
    or ``None`` when the page is ineligible (missing line detection, too few lines, no fields).
    """
    if not settings.template_memory_enabled:
        return None

    detected = _read_detected_lines(output_dir, page_number)
    if detected is None:
        return None
    line_count = len(detected.get("lines") or [])
    if line_count < settings.template_min_lines:
        return None
    fingerprint = page_fingerprint(detected, bins=settings.template_fingerprint_bins)
    if not fingerprint:
        return None

    fields_path = output_dir / "field_grounding" / f"page_{page_number:04d}.fields.json"
    if not fields_path.is_file():
        return None
    try:
        payload = json.loads(fields_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    fields = payload.get("fields")
    if not isinstance(fields, list) or not fields:
        return None

    store = TemplateStore(settings.templates_dir)
    store.upsert(
        fingerprint,
        source_job_id=source_job_id,
        source_page_number=page_number,
        width_px=payload.get("width_px"),
        height_px=payload.get("height_px"),
        line_count=line_count,
        fields=[f for f in fields if isinstance(f, dict)],
    )
    LOG.info(
        "captured template page=%d fingerprint=%s job=%s fields=%d",
        page_number,
        fingerprint[:12],
        source_job_id,
        len(fields),
    )
    return fingerprint


def capture_corrected_pages(
    *,
    output_dir: Path,
    settings: Settings,
    source_job_id: str,
    page_numbers: Iterable[int],
) -> dict[int, str]:
    """Capture several editor-saved pages; return ``{page_number: fingerprint}`` for stored ones."""
    stored: dict[int, str] = {}
    for page_number in sorted({int(p) for p in page_numbers}):
        try:
            fingerprint = capture_corrected_page(
                output_dir=output_dir,
                settings=settings,
                source_job_id=source_job_id,
                page_number=page_number,
            )
        except Exception as exc:  # noqa: BLE001 - capture is best-effort; never fail a save
            LOG.warning("template capture failed page=%d job=%s: %s", page_number, source_job_id, exc)
            continue
        if fingerprint:
            stored[page_number] = fingerprint
    return stored
