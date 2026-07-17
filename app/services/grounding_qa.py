"""E5 vision QA refinement loop: stamp → judge → bounded delta → re-stamp.

The loop stamps a numbered debug preview of the current grounding, sends per-field zoomed
crops to a vision *judge* (preferably a different model/provider than the grounder to avoid
correlated blind spots), applies **bounded** per-field deltas (optionally merged into a
page-wide consensus translation), and re-stamps until the judge reports clean or the
iteration cap is reached.

All bboxes stay in top-left pixel space of the 200 DPI page PNGs (PRD §2.2); every applied
delta is clamped to ``grounding_qa_max_bbox_delta_px`` per axis per iteration, so the loop
can never move a field further than the configured bound.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import shutil
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from statistics import median
from typing import Any, Callable

from PIL import Image

from app.config import Settings
from app.services.form_geometry import clamp_bbox_to_page
from app.services.grounding_prompt import load_qa_judge_prompt
from app.services.grounding_schema import (
    QA_JUDGE_TOOL_NAME,
    anthropic_qa_tool,
    anthropic_qa_tool_choice,
    openai_qa_response_format,
)
from app.services.image_stamping import stamp_qa_preview_pages
from app.services.stamping_common import StampStyle, discover_grounding_pages, read_page_manifest
from app.services.jobs import (
    job_grounding_provider_model,
    page_manifest_image_px,
    read_job_manifest,
    write_job_manifest,
)

LOG = logging.getLogger(__name__)

_SUPPORTED_PROVIDERS = frozenset({"openai", "anthropic"})
_DEFAULT_JUDGE_MODELS = {"openai": "gpt-5", "anthropic": "claude-opus-4-7"}

# Fields we can meaningfully judge placement for (they render onto the page).
_JUDGEABLE_TYPES = frozenset({"text", "multiline_text", "checkbox", "radio", "dropdown", "list_box"})


# --------------------------------------------------------------------------- #
# Judge data contracts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FieldCrop:
    """One field crop request handed to the judge."""

    field_id: str
    field_type: str
    label: str
    bbox: dict[str, int]
    page_index: int
    page_w: int
    page_h: int
    image_png: bytes


@dataclass(frozen=True)
class JudgeVerdict:
    field_id: str
    verdict: str  # "ok" | "shift" | "unsure"
    dx: int = 0
    dy: int = 0
    confidence: float = 0.0


@dataclass
class JudgeResult:
    verdicts: list[JudgeVerdict]
    usage: dict[str, int] = dataclass_field(default_factory=dict)


# A judge takes the page index and its field crops, returns verdicts + token usage.
JudgeFn = Callable[[int, list[FieldCrop]], JudgeResult]


# --------------------------------------------------------------------------- #
# Judge configuration + provider clients
# --------------------------------------------------------------------------- #


def resolve_judge_config(settings: Settings, *, grounder_provider: str) -> tuple[str, str]:
    """Resolve ``(provider, model)`` for the judge.

    An empty ``grounding_qa_judge_provider`` picks the provider that differs from the
    grounder, so the judge is a genuine second opinion (PRD E5.1).
    """
    prov = settings.grounding_qa_judge_provider.strip().lower()
    if not prov:
        prov = "anthropic" if grounder_provider.strip().lower() == "openai" else "openai"
    if prov not in _SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported judge provider {prov!r}; use openai or anthropic.")
    model = settings.grounding_qa_judge_model.strip() or _DEFAULT_JUDGE_MODELS[prov]
    return prov, model


def _openai_usage(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    out: dict[str, int] = {}
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is not None:
        out["input_tokens"] = int(prompt)
    if completion is not None:
        out["output_tokens"] = int(completion)
    return out


def _anthropic_usage(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    out: dict[str, int] = {}
    inp = getattr(usage, "input_tokens", None)
    out_tok = getattr(usage, "output_tokens", None)
    if inp is not None:
        out["input_tokens"] = int(inp)
    if out_tok is not None:
        out["output_tokens"] = int(out_tok)
    return out


def _crop_manifest_text(crops: list[FieldCrop]) -> str:
    rows = [
        f"crop {i + 1}: field_id={c.field_id} type={c.field_type} "
        f"bbox={json.dumps(c.bbox, separators=(',', ':'))} page={c.page_w}x{c.page_h}"
        + (f" label={c.label!r}" if c.label else "")
        for i, c in enumerate(crops)
    ]
    return (
        "Fields on this page (one zoomed crop each, in order below):\n"
        + "\n".join(rows)
        + "\n\nReturn one verdict object per field_id."
    )


def _b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


def _parse_verdicts(payload: dict[str, Any]) -> list[JudgeVerdict]:
    raw = payload.get("fields")
    verdicts: list[JudgeVerdict] = []
    if not isinstance(raw, list):
        return verdicts
    for item in raw:
        if not isinstance(item, dict):
            continue
        fid = item.get("field_id")
        if not isinstance(fid, str) or not fid:
            continue
        verdict = item.get("verdict")
        verdict = verdict if verdict in ("ok", "shift", "unsure") else "unsure"
        try:
            dx = int(round(float(item.get("dx", 0) or 0)))
            dy = int(round(float(item.get("dy", 0) or 0)))
        except (TypeError, ValueError):
            dx, dy = 0, 0
        try:
            conf = float(item.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        verdicts.append(JudgeVerdict(field_id=fid, verdict=verdict, dx=dx, dy=dy, confidence=conf))
    return verdicts


def build_judge_callable(settings: Settings, *, grounder_provider: str) -> JudgeFn:
    """Construct the default vision-model judge from config (no hardcoded keys)."""
    provider, model = resolve_judge_config(settings, grounder_provider=grounder_provider)
    system_prompt = load_qa_judge_prompt()

    if provider == "openai":
        if not settings.openai_api_key.strip():
            raise ValueError("FORMIQO_OPENAI_API_KEY is missing for the OpenAI QA judge.")
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)

        def _judge(page_index: int, crops: list[FieldCrop]) -> JudgeResult:
            content: list[dict[str, Any]] = [{"type": "text", "text": _crop_manifest_text(crops)}]
            for crop in crops:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_b64(crop.image_png)}"},
                    }
                )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                response_format=openai_qa_response_format(),
                timeout=settings.openai_timeout_seconds,
                max_completion_tokens=settings.grounding_qa_judge_max_tokens,
            )
            choice = response.choices[0]
            raw = choice.message.content or "{}"
            payload = json.loads(raw)
            return JudgeResult(_parse_verdicts(payload), _openai_usage(getattr(response, "usage", None)))

        return _judge

    if not settings.anthropic_api_key.strip():
        raise ValueError("FORMIQO_ANTHROPIC_API_KEY is missing for the Anthropic QA judge.")
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)

    def _judge_anthropic(page_index: int, crops: list[FieldCrop]) -> JudgeResult:
        content: list[dict[str, Any]] = [{"type": "text", "text": _crop_manifest_text(crops)}]
        for crop in crops:
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": _b64(crop.image_png)},
                }
            )
        response = client.messages.create(
            model=model,
            max_tokens=settings.grounding_qa_judge_max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
            tools=[anthropic_qa_tool()],
            tool_choice=anthropic_qa_tool_choice(),
            timeout=settings.anthropic_timeout_seconds,
        )
        payload: dict[str, Any] = {}
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", "") == QA_JUDGE_TOOL_NAME:
                data = getattr(block, "input", None)
                if isinstance(data, dict):
                    payload = data
                break
        return JudgeResult(_parse_verdicts(payload), _anthropic_usage(getattr(response, "usage", None)))

    return _judge_anthropic


# --------------------------------------------------------------------------- #
# Crops
# --------------------------------------------------------------------------- #


def crop_field_region(
    image: Image.Image,
    bbox: dict[str, int],
    *,
    zoom: float,
    context_px: int,
    page_w: int,
    page_h: int,
) -> bytes:
    """Return PNG bytes of a zoomed crop around ``bbox`` (with context), for the judge."""
    x0 = max(0, int(bbox["x"]) - context_px)
    y0 = max(0, int(bbox["y"]) - context_px)
    x1 = min(page_w, int(bbox["x"]) + int(bbox["w"]) + context_px)
    y1 = min(page_h, int(bbox["y"]) + int(bbox["h"]) + context_px)
    if x1 <= x0:
        x1 = min(page_w, x0 + 1)
    if y1 <= y0:
        y1 = min(page_h, y0 + 1)
    crop = image.crop((x0, y0, x1, y1))
    if zoom > 1.0:
        crop = crop.resize((max(1, int(crop.width * zoom)), max(1, int(crop.height * zoom))), Image.BICUBIC)
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Bounded delta application + consensus translation
# --------------------------------------------------------------------------- #


def _clamp_delta(value: int, bound: int) -> int:
    return max(-bound, min(bound, value))


def _consensus_translation(
    deltas: list[tuple[int, int]],
    *,
    min_fields: int,
    max_spread: int,
) -> tuple[int, int] | None:
    """Return a single (dx, dy) when enough fields agree on one shift (page misregistration)."""
    if len(deltas) < min_fields:
        return None
    xs = [d[0] for d in deltas]
    ys = [d[1] for d in deltas]
    if (max(xs) - min(xs)) > max_spread or (max(ys) - min(ys)) > max_spread:
        return None
    return int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys)))


def apply_verdicts_to_fields(
    fields: list[dict[str, Any]],
    verdicts: list[JudgeVerdict],
    *,
    max_delta: int,
    page_w: int,
    page_h: int,
    consensus_enabled: bool,
    consensus_min_fields: int,
    consensus_max_spread: int,
) -> dict[str, tuple[int, int]]:
    """Apply bounded deltas in place; return the actual per-field shift applied.

    Every delta is clamped to ``±max_delta`` per axis first. When many fields agree on the
    same shift (within ``consensus_max_spread``), the merged consensus translation is applied
    to every shifted field instead of their individual estimates.
    """
    clamped: dict[str, tuple[int, int]] = {}
    for v in verdicts:
        if v.verdict != "shift":
            continue
        dx = _clamp_delta(v.dx, max_delta)
        dy = _clamp_delta(v.dy, max_delta)
        if dx == 0 and dy == 0:
            continue
        clamped[v.field_id] = (dx, dy)

    if consensus_enabled and clamped:
        merged = _consensus_translation(
            list(clamped.values()),
            min_fields=consensus_min_fields,
            max_spread=consensus_max_spread,
        )
        if merged is not None and (merged[0] != 0 or merged[1] != 0):
            clamped = {fid: merged for fid in clamped}

    applied: dict[str, tuple[int, int]] = {}
    for field in fields:
        if not isinstance(field, dict):
            continue
        fid = field.get("field_id")
        if fid not in clamped:
            continue
        bbox = field.get("bbox")
        if not isinstance(bbox, dict):
            continue
        dx, dy = clamped[fid]
        try:
            cur = {k: int(bbox[k]) for k in ("x", "y", "w", "h")}
        except (KeyError, TypeError, ValueError):
            continue
        moved = clamp_bbox_to_page(
            {"x": cur["x"] + dx, "y": cur["y"] + dy, "w": cur["w"], "h": cur["h"]},
            page_w,
            page_h,
        )
        real_dx = moved["x"] - cur["x"]
        real_dy = moved["y"] - cur["y"]
        if real_dx == 0 and real_dy == 0:
            continue
        field["bbox"] = moved
        applied[fid] = (real_dx, real_dy)
    return applied


# --------------------------------------------------------------------------- #
# Refinement loop
# --------------------------------------------------------------------------- #


@dataclass
class _FieldState:
    ever_adjusted: bool = False
    last_verdict: str = "ok"
    last_confidence: float = 1.0
    total_shift: int = 0


def _load_page_fields(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _judgeable_fields(payload: dict[str, Any]) -> list[dict[str, Any]]:
    fields = payload.get("fields")
    if not isinstance(fields, list):
        return []
    out = []
    for f in fields:
        if isinstance(f, dict) and f.get("type") in _JUDGEABLE_TYPES and isinstance(f.get("bbox"), dict):
            out.append(f)
    return out


def _chunk(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def run_grounding_qa_refinement(
    *,
    job_id: str,
    output_dir: Path,
    settings: Settings,
    judge: JudgeFn | None = None,
    max_iterations: int | None = None,
) -> dict[str, Any]:
    """Run the closed QA loop for a job; persist qa_status + bboxes and job.json stages.

    Returns a summary with iteration/convergence metrics and judge cost.
    """
    job_root = output_dir.parent
    provider, model = job_grounding_provider_model(job_root)

    fg_dir = output_dir / "field_grounding"
    if not fg_dir.is_dir():
        raise FileNotFoundError("field_grounding directory not found")

    stamping_path = fg_dir / "stamping.json"
    if not stamping_path.is_file():
        raise FileNotFoundError("field_grounding/stamping.json not found")
    stamping = json.loads(stamping_path.read_text(encoding="utf-8"))
    values = {k: v for k, v in (stamping.get("values") or {}).items() if isinstance(v, str)}
    style_cfg = stamping.get("style") or {}
    overrides = stamping.get("overrides") or {}
    require_all_values = bool(stamping.get("require_all_values", False))

    # Debug boxes on so the judge sees the exact target box in each crop.
    style = StampStyle(
        font_size_pt=float(style_cfg.get("font_size_pt", 11.0)),
        text_color=str(style_cfg.get("text_color", "#111111")),
        draw_debug_boxes=True,
    )

    if judge is None:
        judge = build_judge_callable(settings, grounder_provider=provider)

    max_iters = max_iterations if max_iterations is not None else settings.grounding_qa_max_iterations
    max_delta = settings.grounding_qa_max_bbox_delta_px

    # Work on a copy so an interrupted loop never corrupts canonical field files.
    run_id = _new_run_id()
    qa_root = output_dir / "qa_refine" / run_id
    work_dir = qa_root / "grounding"
    work_dir.mkdir(parents=True, exist_ok=True)
    canonical_pages = {pi: p for pi, p in discover_grounding_pages(fg_dir)}
    for page_index, src in canonical_pages.items():
        shutil.copyfile(src, work_dir / src.name)

    # Per-page image dims (needed to clamp deltas and crop).
    page_dims: dict[int, tuple[int, int]] = {}
    for page_index in canonical_pages:
        _, manifest = read_page_manifest(output_dir, page_index)
        page_dims[page_index] = page_manifest_image_px(manifest)

    states: dict[str, _FieldState] = {}
    iteration_metrics: list[dict[str, Any]] = []
    judge_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    converged = False

    for iteration in range(1, max_iters + 1):
        preview_dir = qa_root / f"iter_{iteration:02d}"
        stamp_qa_preview_pages(
            output_dir=output_dir,
            provider=provider,
            refined_grounding_dir=work_dir,
            preview_run_dir=preview_dir,
            values=values,
            style=style,
            overrides=overrides,
            require_all_values=require_all_values,
        )

        iter_shifts: list[int] = []
        for page_index, work_path in sorted(_iter_work_pages(work_dir)):
            page_w, page_h = page_dims.get(page_index, (0, 0))
            if page_w <= 0 or page_h <= 0:
                continue
            payload = _load_page_fields(work_path)
            fields = _judgeable_fields(payload)
            if not fields:
                continue

            stamped_png = preview_dir / f"page_{page_index + 1:04d}.{provider.strip().lower()}.stamped.png"
            if not stamped_png.is_file():
                continue
            with Image.open(stamped_png) as img:
                page_image = img.convert("RGB")

                crops: list[FieldCrop] = []
                for field in fields:
                    bbox = {k: int(field["bbox"][k]) for k in ("x", "y", "w", "h")}
                    crop_png = crop_field_region(
                        page_image,
                        bbox,
                        zoom=settings.grounding_qa_crop_zoom,
                        context_px=settings.grounding_qa_crop_context_px,
                        page_w=page_w,
                        page_h=page_h,
                    )
                    crops.append(
                        FieldCrop(
                            field_id=str(field.get("field_id")),
                            field_type=str(field.get("type")),
                            label=str(field.get("label") or ""),
                            bbox=bbox,
                            page_index=page_index,
                            page_w=page_w,
                            page_h=page_h,
                            image_png=crop_png,
                        )
                    )

            verdicts: list[JudgeVerdict] = []
            for batch in _chunk(crops, settings.grounding_qa_max_fields_per_call):
                result = judge(page_index, batch)
                judge_calls += 1
                total_input_tokens += int(result.usage.get("input_tokens", 0) or 0)
                total_output_tokens += int(result.usage.get("output_tokens", 0) or 0)
                verdicts.extend(result.verdicts)

            for v in verdicts:
                st = states.setdefault(v.field_id, _FieldState())
                st.last_verdict = v.verdict
                st.last_confidence = v.confidence

            applied = apply_verdicts_to_fields(
                payload["fields"],
                verdicts,
                max_delta=max_delta,
                page_w=page_w,
                page_h=page_h,
                consensus_enabled=settings.grounding_qa_consensus_translation_enabled,
                consensus_min_fields=settings.grounding_qa_consensus_min_fields,
                consensus_max_spread=settings.grounding_qa_consensus_max_spread_px,
            )
            for fid, (dx, dy) in applied.items():
                st = states.setdefault(fid, _FieldState())
                st.ever_adjusted = True
                st.total_shift += abs(dx) + abs(dy)
                iter_shifts.append(abs(dx) + abs(dy))

            work_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        iteration_metrics.append(
            {
                "iteration": iteration,
                "fields_shifted": len(iter_shifts),
                "total_shift_px": sum(iter_shifts),
                "median_shift_px": int(median(iter_shifts)) if iter_shifts else 0,
            }
        )
        if not iter_shifts:
            converged = True
            break

    # Assign final qa_status/confidence and write back to canonical field files.
    fields_total = 0
    fields_adjusted = 0
    fields_flagged = 0
    fields_confirmed = 0

    for page_index, canonical_path in canonical_pages.items():
        work_path = work_dir / canonical_path.name
        work_payload = _load_page_fields(work_path)
        canonical_payload = _load_page_fields(canonical_path)
        work_by_id = {
            f.get("field_id"): f for f in work_payload.get("fields", []) if isinstance(f, dict)
        }
        for field in canonical_payload.get("fields", []):
            if not isinstance(field, dict):
                continue
            fid = field.get("field_id")
            work_field = work_by_id.get(fid)
            if isinstance(work_field, dict) and isinstance(work_field.get("bbox"), dict):
                field["bbox"] = work_field["bbox"]

            if field.get("type") not in _JUDGEABLE_TYPES:
                field["qa_status"] = None
                continue

            fields_total += 1
            st = states.get(fid, _FieldState())
            status = _final_status(
                st,
                converged=converged,
                flag_low_confidence=settings.grounding_qa_flag_low_confidence,
            )
            field["qa_status"] = status
            if st.last_confidence:
                field["confidence"] = round(float(st.last_confidence), 4)
            if status == "adjusted":
                fields_adjusted += 1
            elif status == "flagged":
                fields_flagged += 1
            else:
                fields_confirmed += 1
        canonical_path.write_text(json.dumps(canonical_payload, indent=2) + "\n", encoding="utf-8")

    iterations_run = len(iteration_metrics)
    summary = {
        "job_id": job_id,
        "judge_provider": resolve_judge_config(settings, grounder_provider=provider)[0],
        "judge_model": resolve_judge_config(settings, grounder_provider=provider)[1],
        "iterations": iterations_run,
        "max_iterations": max_iters,
        "converged": converged,
        "max_bbox_delta_px": max_delta,
        "fields_total": fields_total,
        "fields_confirmed": fields_confirmed,
        "fields_adjusted": fields_adjusted,
        "fields_flagged": fields_flagged,
        "iteration_metrics": iteration_metrics,
        "cost": {
            "judge_calls": judge_calls,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        },
        "run_dir": f"qa_refine/{run_id}",
    }

    _update_job_qa_refine(job_root, summary)
    return summary


def _final_status(state: _FieldState, *, converged: bool, flag_low_confidence: float) -> str:
    if state.last_verdict == "unsure":
        return "flagged"
    if state.last_verdict == "shift" and not converged:
        # Judge still wanted to move it when the iteration cap hit → residual doubt.
        return "flagged"
    if state.last_confidence <= flag_low_confidence:
        return "flagged"
    if state.ever_adjusted:
        return "adjusted"
    return "confirmed"


def _iter_work_pages(work_dir: Path) -> list[tuple[int, Path]]:
    return discover_grounding_pages(work_dir)


def _new_run_id() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _update_job_qa_refine(job_root: Path, summary: dict[str, Any]) -> None:
    try:
        manifest = read_job_manifest(job_root)
    except FileNotFoundError:
        return
    stages = manifest.setdefault("stages", {})
    stages["qa_refine"] = {
        "status": "done",
        "error": None,
        "iterations": summary["iterations"],
        "converged": summary["converged"],
        "fields_total": summary["fields_total"],
        "fields_confirmed": summary["fields_confirmed"],
        "fields_adjusted": summary["fields_adjusted"],
        "fields_flagged": summary["fields_flagged"],
        "judge": {"provider": summary["judge_provider"], "model": summary["judge_model"]},
        "cost": summary["cost"],
        "run_dir": summary["run_dir"],
    }
    write_job_manifest(job_root, manifest)
