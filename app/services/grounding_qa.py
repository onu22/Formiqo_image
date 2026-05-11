"""Vision QA loop: refine grounding bboxes using stamped previews (delta patches only)."""

from __future__ import annotations

import base64
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel, Field

from app.grounding_field_types import (
    TEXT_LIKE_GROUNDING_TYPES,
    TOGGLE_GROUNDING_TYPES,
    is_toggle_value_truthy,
    stamps_as_text,
    stamps_as_toggle,
)
from app.services.field_grounding import (
    GroundingJsonParseError,
    _create_provider_client,
    _extract_json_text,
    _extract_json_text_anthropic,
    _validate_field_grounding_json,
)
from app.services.image_stamping import (
    StampImageStyle,
    _discover_grounding_pages,
    stamp_qa_preview_pages,
)

LOG = logging.getLogger(__name__)

_CORE_KEYS = frozenset({"page_index", "width", "height", "unit", "origin", "fields"})


class BboxDeltaPayload(BaseModel):
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0


class QACorrectionPayload(BaseModel):
    field_id: str = Field(min_length=1)
    issue: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Literal["low", "medium", "high"] | None = None
    bbox_delta: BboxDeltaPayload


class PageTranslationPayload(BaseModel):
    """Uniform pixel translation applied to every eligible field bbox (x,y only; w,h unchanged)."""

    x: int = 0
    y: int = 0


class VisionQAParsed(BaseModel):
    status: Literal["acceptable", "needs_correction"]
    corrections: list[QACorrectionPayload] = Field(default_factory=list)
    page_translation: PageTranslationPayload | None = None


_QA_PROMPT_TEMPLATE = """You are a vision QA assistant for form field placement.

You receive TWO images of the SAME page at identical resolution:
1) ORIGINAL — empty form (reference).
2) PREVIEW — the same page AFTER values were stamped using bounding boxes.

Authoritative page pixel size: width={WIDTH}px height={HEIGHT}px (integer). Origin top-left.

Eligible fields for this review (ONLY these may appear in corrections — ignore any other field_id):
{ELIGIBLE_SUMMARY}

TEXT-LIKE types ({TEXT_TYPES}): Judge whether stamped text aligns with intended writable areas:
baseline vs printed rules, overlap with labels, clipping, vertical placement in cells.

TOGGLE types ({TOGGLE_TYPES}): Judge whether the vector checkmark sits correctly inside the printed
checkbox/radio control (centering, not bleeding past borders).

GLOBAL OFFSET (systematic bias): If eligible stamped controls ALL appear shifted by roughly the SAME
pixel offset versus the blank form (e.g. every label sits a few px too low), set "page_translation"
once with integer {{ "x", "y" }} ADDED to each eligible field's bbox top-left (do NOT move w/h via
page_translation). Then list per-field "corrections" ONLY for outliers that still need adjustment
AFTER that uniform shift (residual bbox_delta). If there is no uniform bias, omit "page_translation"
or set x,y to 0.

Rules:
1. Return JSON only. No markdown or prose outside JSON.
2. If every eligible stamped control looks acceptably placed for production use, set "status": "acceptable",
   "corrections": [], and omit page_translation or set x,y to 0.
3. Otherwise set "status": "needs_correction" and list ONLY misplaced fields from the eligible set (unless
   translation-only fixes everything — then corrections may be empty).
4. Each correction must include bbox_delta as INTEGER pixel shifts ADDED to the CURRENT bbox in grounding JSON:
   new_x = old_x + bbox_delta.x (same for y, w, h). Use small integers; typical fixes are a few pixels vertically.
5. Include "confidence" in [0,1] and optional "severity": "low"|"medium"|"high".
6. Include short "issue" text describing the misalignment (for logs).

Exact response schema:
{{
  "status": "acceptable" | "needs_correction",
  "page_translation": {{ "x": <int>, "y": <int> }},
  "corrections": [
    {{
      "field_id": "<string from eligible list>",
      "issue": "<short string>",
      "confidence": <number 0-1>,
      "severity": "low" | "medium" | "high",
      "bbox_delta": {{ "x": <int>, "y": <int>, "w": <int>, "h": <int> }}
    }}
  ]
}}

Use page_translation with zeros when there is no global shift. Apply page_translation conceptually BEFORE per-field bbox_delta.

Current grounding subset for this page (field_id, type, bbox):
{FIELDS_JSON}
"""


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _validate_stored_page_json(data: dict[str, Any], *, page_index: int, width: int, height: int) -> None:
    core = {k: data[k] for k in sorted(_CORE_KEYS)}
    _validate_field_grounding_json(core, page_index=page_index, width=width, height=height)


def seed_refined_from_canonical(*, grounding_dir: Path, refined_dir: Path) -> None:
    refined_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(grounding_dir.glob("page_*.fields.json"))
    if not paths:
        raise FileNotFoundError(f"No page_*.fields.json under {grounding_dir}")
    for src in paths:
        shutil.copy2(src, refined_dir / src.name)


def _sync_refined_json_to_canonical(*, refined_dir: Path, grounding_dir: Path) -> None:
    paths = sorted(refined_dir.glob("page_*.fields.json"))
    if not paths:
        raise FileNotFoundError(f"No page_*.fields.json under {refined_dir}")
    for src in paths:
        shutil.copy2(src, grounding_dir / src.name)


def eligible_field_ids_for_page(fields: list[Any], values: dict[str, str]) -> set[str]:
    eligible: set[str] = set()
    if not isinstance(fields, list):
        return eligible
    for field in fields:
        if not isinstance(field, dict):
            continue
        fid = field.get("field_id")
        ftype = field.get("type")
        if not isinstance(fid, str) or not fid.strip():
            continue
        if not isinstance(ftype, str):
            continue
        if fid not in values:
            continue
        raw = values[fid]
        if stamps_as_text(ftype):
            if raw.strip():
                eligible.add(fid)
        elif stamps_as_toggle(ftype):
            if is_toggle_value_truthy(raw):
                eligible.add(fid)
    return eligible


def _clamp_delta(v: int, max_abs: int) -> int:
    return max(-max_abs, min(max_abs, v))


def apply_qa_corrections_to_payload(
    payload: dict[str, Any],
    corrections: list[QACorrectionPayload],
    *,
    eligible_ids: set[str],
    width: int,
    height: int,
    max_delta_px: int,
) -> dict[str, Any]:
    """Apply corrections in-place to payload['fields']. Returns summary dict."""
    fields = payload.get("fields")
    if not isinstance(fields, list):
        raise ValueError("payload.fields must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for f in fields:
        if isinstance(f, dict) and isinstance(f.get("field_id"), str):
            by_id[f["field_id"]] = f

    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for c in corrections:
        if c.field_id not in eligible_ids:
            skipped.append({"field_id": c.field_id, "reason": "not_eligible"})
            continue
        target = by_id.get(c.field_id)
        if target is None:
            skipped.append({"field_id": c.field_id, "reason": "unknown_field_id"})
            continue
        bbox = target.get("bbox")
        if not isinstance(bbox, dict):
            skipped.append({"field_id": c.field_id, "reason": "bad_bbox"})
            continue

        dx = _clamp_delta(c.bbox_delta.x, max_delta_px)
        dy = _clamp_delta(c.bbox_delta.y, max_delta_px)
        dw = _clamp_delta(c.bbox_delta.w, max_delta_px)
        dh = _clamp_delta(c.bbox_delta.h, max_delta_px)

        nx = int(bbox["x"]) + dx
        ny = int(bbox["y"]) + dy
        nw = int(bbox["w"]) + dw
        nh = int(bbox["h"]) + dh

        nx = max(0, nx)
        ny = max(0, ny)
        nw = max(1, nw)
        nh = max(1, nh)
        if nx + nw > width:
            nw = width - nx
        if ny + nh > height:
            nh = height - ny
        if nw <= 0 or nh <= 0:
            skipped.append({"field_id": c.field_id, "reason": "collapsed_bbox_after_clamp"})
            continue

        bbox["x"] = nx
        bbox["y"] = ny
        bbox["w"] = nw
        bbox["h"] = nh
        applied.append({"field_id": c.field_id, "bbox_delta_applied": {"x": dx, "y": dy, "w": dw, "h": dh}})

    return {"applied": applied, "skipped": skipped}


def _median_int(values: list[int]) -> int:
    if not values:
        return 0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) // 2


def merge_consensus_translation_from_corrections(
    corrections: list[QACorrectionPayload],
    *,
    min_fields: int,
    max_spread_px: int,
    max_delta_px: int,
) -> tuple[list[QACorrectionPayload], int, int, bool]:
    """Extract median translation per axis when spreads are tight; return residual corrections."""
    if len(corrections) < min_fields:
        return corrections, 0, 0, False
    xs = [c.bbox_delta.x for c in corrections]
    ys = [c.bbox_delta.y for c in corrections]
    tx = 0
    ty = 0
    if max(xs) - min(xs) <= max_spread_px:
        tx = _clamp_delta(_median_int(xs), max_delta_px)
    if max(ys) - min(ys) <= max_spread_px:
        ty = _clamp_delta(_median_int(ys), max_delta_px)
    if tx == 0 and ty == 0:
        return corrections, 0, 0, False

    out: list[QACorrectionPayload] = []
    for c in corrections:
        rx = c.bbox_delta.x - tx
        ry = c.bbox_delta.y - ty
        rw = c.bbox_delta.w
        rh = c.bbox_delta.h
        if rx == 0 and ry == 0 and rw == 0 and rh == 0:
            continue
        out.append(
            c.model_copy(
                update={"bbox_delta": BboxDeltaPayload(x=rx, y=ry, w=rw, h=rh)},
            )
        )
    return out, tx, ty, True


def resolve_translation_and_corrections(
    qa_result: VisionQAParsed,
    filtered_corrections: list[QACorrectionPayload],
    *,
    consensus_enabled: bool,
    consensus_min_fields: int,
    consensus_max_spread_px: int,
    max_delta_px: int,
) -> tuple[list[QACorrectionPayload], int, int, bool, int, int]:
    """Returns final_corrections, eff_tx, eff_ty, consensus_used, llm_tx, llm_ty (all clamped)."""
    llm_tx = llm_ty = 0
    if qa_result.page_translation is not None:
        llm_tx = _clamp_delta(qa_result.page_translation.x, max_delta_px)
        llm_ty = _clamp_delta(qa_result.page_translation.y, max_delta_px)
    if llm_tx != 0 or llm_ty != 0:
        return filtered_corrections, llm_tx, llm_ty, False, llm_tx, llm_ty
    if not consensus_enabled:
        return filtered_corrections, 0, 0, False, llm_tx, llm_ty
    new_corrs, ctx, cty, used = merge_consensus_translation_from_corrections(
        filtered_corrections,
        min_fields=consensus_min_fields,
        max_spread_px=consensus_max_spread_px,
        max_delta_px=max_delta_px,
    )
    return new_corrs, ctx, cty, used, llm_tx, llm_ty


def apply_page_translation_to_eligible(
    payload: dict[str, Any],
    *,
    eligible_ids: set[str],
    tx: int,
    ty: int,
    width: int,
    height: int,
    max_delta_px: int,
) -> dict[str, Any]:
    """Translate bbox (x,y) only for eligible fields; clip boxes to the page."""
    tx = _clamp_delta(tx, max_delta_px)
    ty = _clamp_delta(ty, max_delta_px)
    fields = payload.get("fields")
    if not isinstance(fields, list):
        raise ValueError("payload.fields must be a list")
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for f in fields:
        if not isinstance(f, dict):
            continue
        fid = f.get("field_id")
        if not isinstance(fid, str) or fid not in eligible_ids:
            continue
        bbox = f.get("bbox")
        if not isinstance(bbox, dict):
            skipped.append({"field_id": fid, "reason": "bad_bbox"})
            continue

        dx = tx
        dy = ty
        dw = 0
        dh = 0

        nx = int(bbox["x"]) + dx
        ny = int(bbox["y"]) + dy
        nw = int(bbox["w"]) + dw
        nh = int(bbox["h"]) + dh

        nx = max(0, nx)
        ny = max(0, ny)
        nw = max(1, nw)
        nh = max(1, nh)
        if nx + nw > width:
            nw = width - nx
        if ny + nh > height:
            nh = height - ny
        if nw <= 0 or nh <= 0:
            skipped.append({"field_id": fid, "reason": "collapsed_bbox_after_clamp"})
            continue

        bbox["x"] = nx
        bbox["y"] = ny
        bbox["w"] = nw
        bbox["h"] = nh
        applied.append({"field_id": fid, "translation_applied": {"x": dx, "y": dy}})

    return {"applied": applied, "skipped": skipped}


def apply_vision_qa_adjustments(
    payload: dict[str, Any],
    *,
    eligible_ids: set[str],
    page_tx: int,
    page_ty: int,
    corrections: list[QACorrectionPayload],
    width: int,
    height: int,
    max_delta_px: int,
) -> dict[str, Any]:
    """Apply uniform page translation to eligible fields, then per-field deltas."""
    summary: dict[str, Any] = {}
    if page_tx != 0 or page_ty != 0:
        summary["translation"] = apply_page_translation_to_eligible(
            payload,
            eligible_ids=eligible_ids,
            tx=page_tx,
            ty=page_ty,
            width=width,
            height=height,
            max_delta_px=max_delta_px,
        )
    else:
        summary["translation"] = {"applied": [], "skipped": []}
    summary["corrections"] = apply_qa_corrections_to_payload(
        payload,
        corrections,
        eligible_ids=eligible_ids,
        width=width,
        height=height,
        max_delta_px=max_delta_px,
    )
    return summary


def _build_qa_prompt(
    *,
    width: int,
    height: int,
    eligible: set[str],
    fields_subset: list[dict[str, Any]],
) -> str:
    eligible_summary = (
        ", ".join(sorted(eligible)) if eligible else "(none — return acceptable with empty corrections)"
    )
    fields_json = json.dumps(fields_subset, indent=2)
    text_types = ", ".join(sorted(TEXT_LIKE_GROUNDING_TYPES))
    toggle_types = ", ".join(sorted(TOGGLE_GROUNDING_TYPES))
    return (
        _QA_PROMPT_TEMPLATE.replace("{WIDTH}", str(width))
        .replace("{HEIGHT}", str(height))
        .replace("{ELIGIBLE_SUMMARY}", eligible_summary)
        .replace("{TEXT_TYPES}", text_types)
        .replace("{TOGGLE_TYPES}", toggle_types)
        .replace("{FIELDS_JSON}", fields_json)
    )


def _fields_subset_for_eligible(fields: list[Any], eligible: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        fid = field.get("field_id")
        if fid not in eligible:
            continue
        out.append(
            {
                "field_id": fid,
                "type": field.get("type"),
                "bbox": field.get("bbox"),
            }
        )
    return out


def _call_openai_vision_qa(
    client: OpenAI,
    *,
    model: str,
    timeout_seconds: float,
    max_output_tokens: int,
    prompt: str,
    original_png: Path,
    preview_png: Path,
) -> VisionQAParsed:
    b64_a = base64.b64encode(original_png.read_bytes()).decode("ascii")
    b64_b = base64.b64encode(preview_png.read_bytes()).decode("ascii")
    response = client.responses.create(
        model=model,
        timeout=timeout_seconds,
        max_output_tokens=max_output_tokens,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b64_a}"},
                    {"type": "input_text", "text": "Image above: ORIGINAL form page."},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b64_b}"},
                    {"type": "input_text", "text": "Image above: PREVIEW with stamped values."},
                ],
            }
        ],
        text={"format": {"type": "json_object"}},
    )
    raw = _extract_json_text(response)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GroundingJsonParseError(f"Invalid JSON from OpenAI QA: {exc}", raw) from exc
    return VisionQAParsed.model_validate(parsed)


def _call_anthropic_vision_qa(
    client: Anthropic,
    *,
    model: str,
    timeout_seconds: float,
    max_tokens: int,
    prompt: str,
    original_png: Path,
    preview_png: Path,
) -> VisionQAParsed:
    b64_a = base64.b64encode(original_png.read_bytes()).decode("ascii")
    b64_b = base64.b64encode(preview_png.read_bytes()).decode("ascii")
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        timeout=timeout_seconds,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Image 1 — ORIGINAL form page."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64_a,
                        },
                    },
                    {"type": "text", "text": "Image 2 — PREVIEW with stamped values."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64_b,
                        },
                    },
                ],
            }
        ],
    )
    raw = _extract_json_text_anthropic(response)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GroundingJsonParseError(f"Invalid JSON from Anthropic QA: {exc}", raw) from exc
    return VisionQAParsed.model_validate(parsed)


def run_grounding_qa_refinement_loop(
    *,
    job_id: str,
    output_dir: Path,
    provider: str,
    model: str,
    values: dict[str, str],
    style: StampImageStyle,
    require_all_values: bool,
    openai_api_key: str,
    openai_timeout_seconds: float,
    openai_max_output_tokens: int,
    anthropic_api_key: str,
    anthropic_timeout_seconds: float,
    anthropic_max_tokens: int,
    max_iterations: int,
    max_bbox_delta_px: int,
    consensus_translation_enabled: bool,
    consensus_min_fields: int,
    consensus_max_spread_px: int,
) -> dict[str, Any]:
    provider_norm = provider.strip().lower()
    model_norm = model.strip()
    grounding_dir = output_dir / "field_grounding"
    refined_dir = grounding_dir / "refined"
    if not grounding_dir.is_dir():
        raise FileNotFoundError(f"field_grounding not found under {output_dir}")

    seed_refined_from_canonical(grounding_dir=grounding_dir, refined_dir=refined_dir)

    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    qa_session_dir = grounding_dir / "qa_refinement" / session_id
    qa_session_dir.mkdir(parents=True, exist_ok=True)

    client = _create_provider_client(
        provider=provider_norm,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
    )
    timeout_seconds = openai_timeout_seconds if provider_norm == "openai" else anthropic_timeout_seconds

    pages_meta = _discover_grounding_pages(refined_dir)

    iterations_log: list[list[dict[str, Any]]] = []
    promoted = False
    stopped_reason = "max_iterations"
    iterations_run = 0

    for iteration in range(1, max_iterations + 1):
        iterations_run = iteration
        preview_dir = output_dir / "stamped_images" / "qa_refinement" / session_id / f"iter_{iteration}"
        stamp_qa_preview_pages(
            output_dir=output_dir,
            provider=provider_norm,
            refined_grounding_dir=refined_dir,
            preview_run_dir=preview_dir,
            values=values,
            style=style,
            require_all_values=require_all_values,
        )

        iter_pages: list[dict[str, Any]] = []
        patch_pages: list[dict[str, Any]] = []
        all_acceptable = True
        pending_apply: list[
            tuple[Path, dict[str, Any], list[QACorrectionPayload], set[str], int, int, int, int]
        ] = []

        for page_index, refined_path in pages_meta:
            payload = _load_json(refined_path)
            fields = payload.get("fields")
            width = int(payload["width"])
            height = int(payload["height"])
            eligible = eligible_field_ids_for_page(fields if isinstance(fields, list) else [], values)

            preview_png = preview_dir / f"page_{page_index + 1:04d}.{provider_norm}.stamped.png"
            original_png = output_dir / "converted_images" / f"page_{page_index + 1:04d}.png"
            if not preview_png.is_file():
                raise FileNotFoundError(f"QA preview missing: {preview_png}")
            if not original_png.is_file():
                raise FileNotFoundError(f"Original page PNG missing: {original_png}")

            if not eligible:
                iter_pages.append(
                    {
                        "page_index": page_index,
                        "qa_status": "acceptable",
                        "corrections_requested": 0,
                        "preview_image": str(preview_png.relative_to(output_dir)).replace("\\", "/"),
                        "note": "no_eligible_stamped_fields",
                    }
                )
                patch_pages.append(
                    {
                        "page_index": page_index,
                        "status": "acceptable",
                        "corrections": [],
                        "skipped_ineligible": [],
                    }
                )
                continue

            subset = _fields_subset_for_eligible(fields if isinstance(fields, list) else [], eligible)
            prompt = _build_qa_prompt(width=width, height=height, eligible=eligible, fields_subset=subset)

            if provider_norm == "openai":
                qa_result = _call_openai_vision_qa(
                    client,
                    model=model_norm,
                    timeout_seconds=timeout_seconds,
                    max_output_tokens=openai_max_output_tokens,
                    prompt=prompt,
                    original_png=original_png,
                    preview_png=preview_png,
                )
            else:
                qa_result = _call_anthropic_vision_qa(
                    client,
                    model=model_norm,
                    timeout_seconds=timeout_seconds,
                    max_tokens=anthropic_max_tokens,
                    prompt=prompt,
                    original_png=original_png,
                    preview_png=preview_png,
                )

            filtered_corrections: list[QACorrectionPayload] = []
            skipped_ineligible: list[str] = []
            for c in qa_result.corrections:
                if c.field_id not in eligible:
                    skipped_ineligible.append(c.field_id)
                    continue
                filtered_corrections.append(c)

            final_corrections, eff_tx, eff_ty, consensus_used, llm_tx, llm_ty = resolve_translation_and_corrections(
                qa_result,
                filtered_corrections,
                consensus_enabled=consensus_translation_enabled,
                consensus_min_fields=consensus_min_fields,
                consensus_max_spread_px=consensus_max_spread_px,
                max_delta_px=max_bbox_delta_px,
            )

            needs_apply = eff_tx != 0 or eff_ty != 0 or len(final_corrections) > 0
            effective_status: Literal["acceptable", "needs_correction"] = (
                "needs_correction" if needs_apply else "acceptable"
            )

            if effective_status != "acceptable":
                all_acceptable = False

            iter_pages.append(
                {
                    "page_index": page_index,
                    "qa_status": effective_status,
                    "corrections_requested": len(final_corrections),
                    "preview_image": str(preview_png.relative_to(output_dir)).replace("\\", "/"),
                }
            )

            patch_entry: dict[str, Any] = {
                "page_index": page_index,
                "qa_status_raw": qa_result.status,
                "effective_status": effective_status,
                "page_translation_llm": {"x": llm_tx, "y": llm_ty},
                "page_translation_applied": {"x": eff_tx, "y": eff_ty},
                "consensus_used": consensus_used,
                "corrections": [c.model_dump() for c in final_corrections],
                "skipped_ineligible_field_ids": skipped_ineligible,
            }

            if needs_apply:
                pending_apply.append(
                    (refined_path, payload, final_corrections, eligible, width, height, eff_tx, eff_ty)
                )

            patch_pages.append(patch_entry)

        patch_path = qa_session_dir / f"iter_{iteration}_patch.json"
        patch_path.write_text(
            json.dumps({"iteration": iteration, "pages": patch_pages}, indent=2) + "\n",
            encoding="utf-8",
        )

        iterations_log.append(iter_pages)

        if all_acceptable:
            stopped_reason = "acceptable"
            promoted = True
            break

        for refined_path, payload, corrections, eligible, w, h, ptx, pty in pending_apply:
            summary = apply_vision_qa_adjustments(
                payload,
                eligible_ids=eligible,
                page_tx=ptx,
                page_ty=pty,
                corrections=corrections,
                width=w,
                height=h,
                max_delta_px=max_bbox_delta_px,
            )
            _validate_stored_page_json(payload, page_index=int(payload["page_index"]), width=w, height=h)
            _save_json(refined_path, payload)
            LOG.debug("QA apply %s: %s", refined_path.name, summary)

    _sync_refined_json_to_canonical(refined_dir=refined_dir, grounding_dir=grounding_dir)
    LOG.info(
        "QA refined grounding synced to canonical (promoted=%s stopped_reason=%s)",
        promoted,
        stopped_reason,
    )

    final_preview_dir = output_dir / "stamped_images" / "qa_refinement" / session_id / "final"
    stamp_qa_preview_pages(
        output_dir=output_dir,
        provider=provider_norm,
        refined_grounding_dir=refined_dir,
        preview_run_dir=final_preview_dir,
        values=values,
        style=style,
        require_all_values=require_all_values,
    )

    refined_rel = str(refined_dir.relative_to(output_dir)).replace("\\", "/")
    qa_rel = str(qa_session_dir.relative_to(output_dir)).replace("\\", "/")
    final_preview_rel = str(final_preview_dir.relative_to(output_dir)).replace("\\", "/")

    return {
        "job_id": job_id,
        "provider": provider_norm,
        "model": model_norm,
        "session_id": session_id,
        "promoted": promoted,
        "stopped_reason": stopped_reason,
        "iterations_run": iterations_run,
        "refined_dir": refined_rel,
        "qa_session_dir": qa_rel,
        "final_preview_dir": final_preview_rel,
        "canonical_grounding_updated": True,
        "iterations": iterations_log,
    }
