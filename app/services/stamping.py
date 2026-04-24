"""Stamp plan generation: map grounded image bboxes back to PDF points.

Given a completed conversion job (``converted_images/pages/page_XXXX.json``)
and a completed field grounding run (``field_grounding/{provider}_{model}/``),
produce a library-agnostic "stamp plan" JSON per page under
``stamp_plans/{provider}_{model}/``.

Every field in a plan carries both coordinate representations so downstream
stampers do not need to recompute geometry:

- ``pdf_bbox_bl`` — PDF points, bottom-left origin (ReportLab, pypdf).
- ``pdf_rect_tl`` — PDF points, top-left origin (PyMuPDF ``fitz.Rect``).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

_GROUNDING_PAGE_RE = re.compile(r"^page_(\d{4})\.fields\.json$")
_MODEL_DIR_SAFE_RE = re.compile(r"[^a-z0-9._-]+")

STAMP_PLAN_SCHEMA_VERSION = "1.0"


class ImageBBox(TypedDict):
    x: int
    y: int
    w: int
    h: int


class PDFRectBL(TypedDict):
    """PDF rectangle in points with bottom-left origin (ReportLab / pypdf)."""

    pdf_x: float
    pdf_y: float
    pdf_w: float
    pdf_h: float


class PDFRectTL(TypedDict):
    """PDF rectangle in points with top-left origin (PyMuPDF / ``fitz.Rect``)."""

    x0: float
    y0: float
    x1: float
    y1: float


def image_bbox_to_pdf_rects(
    bbox: ImageBBox | dict[str, Any],
    *,
    pdf_width_pt: float,
    pdf_height_pt: float,
    image_width_px: int,
    image_height_px: int,
) -> tuple[PDFRectBL, PDFRectTL]:
    """Convert an image-space bbox (top-left origin, pixels) into PDF rectangles.

    Returns ``(pdf_bbox_bl, pdf_rect_tl)`` where:
      - ``pdf_bbox_bl`` uses PDF bottom-left origin (``pdf_x``, ``pdf_y`` at the lower-left).
      - ``pdf_rect_tl`` uses PDF top-left origin (``x0, y0, x1, y1``) matching ``fitz.Rect``.

    Raises ``ValueError`` on invalid inputs.
    """
    if pdf_width_pt <= 0 or pdf_height_pt <= 0:
        raise ValueError("pdf_width_pt and pdf_height_pt must be positive.")
    if image_width_px <= 0 or image_height_px <= 0:
        raise ValueError("image_width_px and image_height_px must be positive.")

    try:
        x = int(bbox["x"])
        y = int(bbox["y"])
        w = int(bbox["w"])
        h = int(bbox["h"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"bbox must contain integer keys x, y, w, h: {exc}") from exc

    if w <= 0 or h <= 0:
        raise ValueError("bbox requires w > 0 and h > 0.")
    if x < 0 or y < 0:
        raise ValueError("bbox requires x >= 0 and y >= 0.")
    if x + w > image_width_px or y + h > image_height_px:
        raise ValueError(
            f"bbox exceeds image bounds ({image_width_px}x{image_height_px}): "
            f"x={x}, y={y}, w={w}, h={h}"
        )

    sx = pdf_width_pt / image_width_px
    sy = pdf_height_pt / image_height_px

    pdf_bl: PDFRectBL = {
        "pdf_x": x * sx,
        "pdf_y": pdf_height_pt - (y + h) * sy,
        "pdf_w": w * sx,
        "pdf_h": h * sy,
    }
    pdf_tl: PDFRectTL = {
        "x0": x * sx,
        "y0": y * sy,
        "x1": (x + w) * sx,
        "y1": (y + h) * sy,
    }
    return pdf_bl, pdf_tl


def _provider_model_dir_name(provider: str, model: str) -> str:
    safe_provider = _MODEL_DIR_SAFE_RE.sub("-", provider.lower()).strip("-")
    safe_model = _MODEL_DIR_SAFE_RE.sub("-", model.lower()).strip("-")
    return f"{safe_provider}_{safe_model}"


def _load_conversion_page_manifest(output_dir: Path, page_index: int) -> dict[str, Any]:
    """Load ``converted_images/pages/page_{N:04d}.json`` for a 0-based page index."""
    rel = f"converted_images/pages/page_{page_index + 1:04d}.json"
    path = output_dir / rel
    if not path.is_file():
        raise FileNotFoundError(f"Conversion page manifest not found: {rel}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_stamp_plan_for_page(
    *,
    job_id: str,
    source_pdf: str,
    page_index: int,
    conversion_page_manifest: dict[str, Any],
    grounding_page_json: dict[str, Any],
    provider: str,
    model: str,
    run_id: str,
) -> dict[str, Any]:
    """Combine a conversion page manifest and a grounding page JSON into a stamp plan."""
    try:
        pdf_w = float(conversion_page_manifest["pdf"]["width_pt"])
        pdf_h = float(conversion_page_manifest["pdf"]["height_pt"])
        image_node = conversion_page_manifest["image"]
        image_path = str(image_node["path"])
        img_w = int(image_node["saved_image_width_px"])
        img_h = int(image_node["saved_image_height_px"])
        mapping = conversion_page_manifest.get("mapping", {})
        scale_x = float(mapping.get("image_to_pdf_scale_x", pdf_w / img_w))
        scale_y = float(mapping.get("image_to_pdf_scale_y", pdf_h / img_h))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid conversion page manifest: {exc}") from exc

    g_width = int(grounding_page_json.get("width", -1))
    g_height = int(grounding_page_json.get("height", -1))
    g_page_index = int(grounding_page_json.get("page_index", -1))

    if g_page_index != page_index:
        raise ValueError(
            f"Grounding page_index ({g_page_index}) does not match conversion page_index ({page_index})."
        )
    if g_width != img_w or g_height != img_h:
        raise ValueError(
            f"Grounding image dimensions do not match conversion manifest "
            f"(grounding={g_width}x{g_height}, manifest={img_w}x{img_h})."
        )
    if grounding_page_json.get("origin") != "top-left":
        raise ValueError('Grounding origin must be "top-left".')
    if grounding_page_json.get("unit") != "px":
        raise ValueError('Grounding unit must be "px".')

    fields_in = grounding_page_json.get("fields", [])
    if not isinstance(fields_in, list):
        raise ValueError("Grounding fields must be a list.")

    fields_out: list[dict[str, Any]] = []
    for idx, field in enumerate(fields_in):
        if not isinstance(field, dict):
            raise ValueError(f"Grounding fields[{idx}] must be an object.")
        try:
            field_id = field["field_id"]
            field_type = field["type"]
            bbox = field["bbox"]
        except KeyError as exc:
            raise ValueError(f"Grounding fields[{idx}] missing key: {exc}") from exc

        pdf_bl, pdf_tl = image_bbox_to_pdf_rects(
            bbox,
            pdf_width_pt=pdf_w,
            pdf_height_pt=pdf_h,
            image_width_px=img_w,
            image_height_px=img_h,
        )
        fields_out.append(
            {
                "field_id": field_id,
                "type": field_type,
                "image_bbox": {
                    "x": int(bbox["x"]),
                    "y": int(bbox["y"]),
                    "w": int(bbox["w"]),
                    "h": int(bbox["h"]),
                },
                "pdf_bbox_bl": pdf_bl,
                "pdf_rect_tl": pdf_tl,
            }
        )

    return {
        "schema_version": STAMP_PLAN_SCHEMA_VERSION,
        "job_id": job_id,
        "source_pdf": source_pdf,
        "page_index": page_index,
        "pdf": {
            "width_pt": pdf_w,
            "height_pt": pdf_h,
            "origin": "bottom-left",
        },
        "image_reference": {
            "path": image_path,
            "width_px": img_w,
            "height_px": img_h,
            "origin": "top-left",
        },
        "scales": {
            "image_to_pdf_scale_x": scale_x,
            "image_to_pdf_scale_y": scale_y,
        },
        "grounding": {
            "provider": provider,
            "model": model,
            "run_id": run_id,
        },
        "unit": "pt",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fields": fields_out,
    }


def _discover_grounding_pages(grounding_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for path in sorted(grounding_dir.glob("page_*.fields.json")):
        m = _GROUNDING_PAGE_RE.match(path.name)
        if not m:
            continue
        page_index = int(m.group(1)) - 1
        out.append((page_index, path))
    return out


def _read_grounding_manifest_run_id(grounding_dir: Path) -> str:
    manifest_path = grounding_dir / "manifest.json"
    if not manifest_path.is_file():
        return ""
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    run_id = data.get("run_id")
    return str(run_id) if isinstance(run_id, str) else ""


def run_prepare_stamp_for_job(
    *,
    job_id: str,
    output_dir: Path,
    source_pdf: str,
    provider: str,
    model: str,
) -> dict[str, Any]:
    """Build stamp plans for every grounded page of a job.

    Writes:
      - ``output/stamp_plans/{provider}_{model}/page_XXXX.stamp.json`` (one per grounded page)
      - ``output/stamp_plans/{provider}_{model}/manifest.json``

    Overwrites in place on re-run.
    """
    provider_norm = provider.strip().lower()
    if not provider_norm:
        raise ValueError("provider must be a non-empty string.")
    if not model.strip():
        raise ValueError("model must be a non-empty string.")

    run_dir_name = _provider_model_dir_name(provider_norm, model)
    grounding_dir = output_dir / "field_grounding" / run_dir_name
    if not grounding_dir.is_dir():
        raise FileNotFoundError(
            f"Field grounding run not found for provider={provider_norm}, model={model} "
            f"(expected directory: field_grounding/{run_dir_name})."
        )

    conversion_pages_dir = output_dir / "converted_images" / "pages"
    if not conversion_pages_dir.is_dir():
        raise FileNotFoundError(
            f"Conversion page manifests not found: converted_images/pages"
        )

    grounding_pages = _discover_grounding_pages(grounding_dir)
    if not grounding_pages:
        raise FileNotFoundError(
            f"No grounded page files found under field_grounding/{run_dir_name}."
        )

    run_id_from_manifest = _read_grounding_manifest_run_id(grounding_dir)

    stamp_root = output_dir / "stamp_plans" / run_dir_name
    stamp_root.mkdir(parents=True, exist_ok=True)

    run_id = run_id_from_manifest or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir_rel = f"stamp_plans/{run_dir_name}"

    page_results: list[dict[str, Any]] = []
    output_files: list[str] = []

    for page_index, grounding_path in grounding_pages:
        grounding_rel = str(grounding_path.relative_to(output_dir)).replace("\\", "/")
        try:
            grounding_data = json.loads(grounding_path.read_text(encoding="utf-8"))
            conv_manifest = _load_conversion_page_manifest(output_dir, page_index)

            page_run_id = str(grounding_data.get("run_id") or run_id_from_manifest or run_id)
            page_provider = str(grounding_data.get("provider") or provider_norm)
            page_model = str(grounding_data.get("model") or model)

            plan = build_stamp_plan_for_page(
                job_id=job_id,
                source_pdf=source_pdf,
                page_index=page_index,
                conversion_page_manifest=conv_manifest,
                grounding_page_json=grounding_data,
                provider=page_provider,
                model=page_model,
                run_id=page_run_id,
            )

            out_name = f"page_{page_index + 1:04d}.stamp.json"
            out_path = stamp_root / out_name
            out_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
            out_rel = f"{run_dir_rel}/{out_name}"
            output_files.append(out_rel)

            page_results.append(
                {
                    "page_index": page_index,
                    "grounding_file": grounding_rel,
                    "status": "succeeded",
                    "stamp_plan_path": out_rel,
                    "field_count": len(plan["fields"]),
                }
            )
        except Exception as exc:  # noqa: BLE001 - surface per-page errors in response
            page_results.append(
                {
                    "page_index": page_index,
                    "grounding_file": grounding_rel,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    succeeded_count = sum(1 for p in page_results if p["status"] == "succeeded")
    failed_count = len(page_results) - succeeded_count

    manifest_rel = f"{run_dir_rel}/manifest.json"
    manifest = {
        "schema_version": STAMP_PLAN_SCHEMA_VERSION,
        "job_id": job_id,
        "source_pdf": source_pdf,
        "provider": provider_norm,
        "model": model,
        "run_id": run_id,
        "run_dir": run_dir_rel,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(page_results),
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "files": output_files,
        "pages": page_results,
    }
    (stamp_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "job_id": job_id,
        "provider": provider_norm,
        "model": model,
        "run_id": run_id,
        "run_dir": run_dir_rel,
        "manifest_path": manifest_rel,
        "page_count": len(page_results),
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "pages": page_results,
    }
