"""E5 vision QA refinement loop: stamp -> judge crops -> bounded delta -> re-stamp.

The loop stamps a numbered debug preview, sends per-field zoom crops to a vision judge
(preferably a different provider/model than the grounder to avoid correlated blind spots),
applies **bounded** per-iteration bbox deltas (optionally merged into a page-level consensus
translation), and re-stamps until the judge reports clean or ``grounding_qa_max_iterations``
is reached.

Field bboxes in ``field_grounding/page_*.fields.json`` are refined in place. Per-field QA
outcomes (``qa_status``: ``confirmed | adjusted | flagged`` and ``qa_confidence``) are
persisted, and ``stages.qa_refine`` in ``job.json`` records convergence + judge token cost.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Callable

from app.config import Settings
from app.grounding_field_types import is_toggle_value_truthy, stamps_as_text, stamps_as_toggle
from app.services.grounding_prompt import (
    configure_prompt_dir,
    load_qa_judge_system_prompt,
    load_qa_judge_user_prompt,
)
from app.services.qa_schema import (
    QA_JUDGE_TOOL_NAME,
    anthropic_qa_tool,
    anthropic_qa_tool_choice,
    openai_qa_response_format,
)
from app.services.stamping_common import StampStyle, discover_grounding_pages

LOG = logging.getLogger(__name__)

_SUPPORTED_PROVIDERS = frozenset({"openai", "anthropic"})
_DEFAULT_OPENAI_JUDGE_MODEL = "gpt-5"
_DEFAULT_ANTHROPIC_JUDGE_MODEL = "claude-opus-4-7"


class QaRefinementError(RuntimeError):
    """Refinement could not run (e.g. missing judge credentials or grounding)."""


@dataclass(frozen=True)
class JudgeRequest:
    """Everything the judge needs to rule on one field's placement."""

    field_id: str
    field_type: str
    label: str
    value: str
    page_index: int
    bbox: dict[str, int]
    page_width: int
    page_height: int
    crop_zoom: float
    crop_path: Path


@dataclass(frozen=True)
class JudgeVerdict:
    """Judge output for one field (positional only)."""

    verdict: str  # "ok" | "adjust"
    dx: int = 0
    dy: int = 0
    confidence: float = 1.0
    reason: str = ""
    usage: dict[str, int] | None = None


JudgeFn = Callable[[JudgeRequest], JudgeVerdict]


@dataclass
class _FieldOutcome:
    moved: bool = False
    last_verdict: str | None = None
    last_confidence: float = 1.0
    total_shift: tuple[int, int] = (0, 0)


@dataclass
class _CostAccumulator:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0

    def add(self, usage: dict[str, int] | None, *, latency: float) -> None:
        self.calls += 1
        self.latency_seconds += latency
        if usage:
            self.input_tokens += int(usage.get("input_tokens", 0))
            self.output_tokens += int(usage.get("output_tokens", 0))

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested directly)
# --------------------------------------------------------------------------- #


def clamp_delta(value: int, *, max_delta: int) -> int:
    """Clamp a single-axis delta to ``[-max_delta, +max_delta]``."""
    if value > max_delta:
        return max_delta
    if value < -max_delta:
        return -max_delta
    return value


def shift_bbox(bbox: dict[str, int], dx: int, dy: int, *, page_w: int, page_h: int) -> dict[str, int]:
    """Translate a bbox by ``(dx, dy)`` and clamp it inside the page."""
    w = int(bbox["w"])
    h = int(bbox["h"])
    x = int(bbox["x"]) + dx
    y = int(bbox["y"]) + dy
    x = max(0, min(x, max(0, page_w - w)))
    y = max(0, min(y, max(0, page_h - h)))
    return {"x": x, "y": y, "w": w, "h": h}


def merge_consensus_translation(
    deltas: dict[str, tuple[int, int]],
    *,
    min_fields: int,
    max_spread_px: int,
) -> tuple[int, int] | None:
    """Return a single page translation when enough per-field deltas agree.

    Fields "agree" when at least ``min_fields`` were flagged for adjustment and the spread
    (max-min) of their delta components stays within ``max_spread_px`` on both axes — the
    signature of a whole-page/scan offset rather than independent per-field errors.
    """
    if len(deltas) < min_fields:
        return None
    xs = [d[0] for d in deltas.values()]
    ys = [d[1] for d in deltas.values()]
    if (max(xs) - min(xs)) > max_spread_px or (max(ys) - min(ys)) > max_spread_px:
        return None
    return round(sum(xs) / len(xs)), round(sum(ys) / len(ys))


def bbox_center_error(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Euclidean distance between two bbox centers (px); QA before/after metric helper."""
    acx = float(a["x"]) + float(a["w"]) / 2.0
    acy = float(a["y"]) + float(a["h"]) / 2.0
    bcx = float(b["x"]) + float(b["w"]) / 2.0
    bcy = float(b["y"]) + float(b["h"]) / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def resolve_judge_provider_model(settings: Settings, *, grounder_provider: str) -> tuple[str, str]:
    """Pick the judge provider/model, defaulting to a provider different from the grounder."""
    prov = settings.grounding_qa_provider.strip().lower()
    if not prov:
        prov = "anthropic" if grounder_provider.strip().lower() == "openai" else "openai"
    if prov not in _SUPPORTED_PROVIDERS:
        raise QaRefinementError(f"Unsupported QA judge provider {prov!r}; use openai or anthropic.")
    model = settings.grounding_qa_model.strip()
    if not model:
        model = _DEFAULT_OPENAI_JUDGE_MODEL if prov == "openai" else _DEFAULT_ANTHROPIC_JUDGE_MODEL
    return prov, model


def field_has_visible_value(field: dict[str, Any], values: dict[str, str]) -> bool:
    """True when a field stamps something the judge can actually see on the preview."""
    field_id = field.get("field_id")
    ftype = field.get("type")
    if not isinstance(field_id, str) or not isinstance(ftype, str):
        return False
    value = values.get(field_id, "")
    if stamps_as_toggle(ftype):
        return is_toggle_value_truthy(value)
    if stamps_as_text(ftype):
        return value.strip() != ""
    return False


# --------------------------------------------------------------------------- #
# Crop building
# --------------------------------------------------------------------------- #


def build_field_crop(
    *,
    preview_png: Path,
    bbox: dict[str, int],
    zoom: float,
    padding_px: int,
    dst_png: Path,
) -> Path | None:
    """Cut a padded, zoomed crop around ``bbox`` from ``preview_png``; return its path."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is a hard dependency here
        return None
    try:
        with Image.open(preview_png) as base:
            rgb = base.convert("RGB")
        w, h = rgb.size
        x0 = max(0, int(bbox["x"]) - padding_px)
        y0 = max(0, int(bbox["y"]) - padding_px)
        x1 = min(w, int(bbox["x"]) + int(bbox["w"]) + padding_px)
        y1 = min(h, int(bbox["y"]) + int(bbox["h"]) + padding_px)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = rgb.crop((x0, y0, x1, y1))
        if zoom > 1.0:
            crop = crop.resize((max(1, int(crop.width * zoom)), max(1, int(crop.height * zoom))))
        dst_png.parent.mkdir(parents=True, exist_ok=True)
        crop.save(dst_png, format="PNG")
        return dst_png
    except Exception as exc:  # pragma: no cover - defensive; a bad crop skips that field
        LOG.warning("qa crop failed for %s: %s", dst_png.name, exc)
        return None


# --------------------------------------------------------------------------- #
# Provider judge (OpenAI / Anthropic structured outputs)
# --------------------------------------------------------------------------- #


def _image_data_url(path: Path) -> str:
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _field_context_json(req: JudgeRequest) -> str:
    return json.dumps(
        {
            "field_id": req.field_id,
            "type": req.field_type,
            "label": req.label,
            "value": req.value,
            "bbox": req.bbox,
            "page": {"width": req.page_width, "height": req.page_height},
            "crop_zoom": req.crop_zoom,
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _parse_verdict_payload(field_id: str, payload: dict[str, Any], usage: dict[str, int] | None) -> JudgeVerdict:
    verdict = str(payload.get("verdict", "ok")).strip().lower()
    if verdict not in ("ok", "adjust"):
        verdict = "ok"
    try:
        dx = int(round(float(payload.get("dx", 0) or 0)))
        dy = int(round(float(payload.get("dy", 0) or 0)))
    except (TypeError, ValueError):
        dx, dy = 0, 0
    try:
        confidence = float(payload.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))
    if verdict == "ok":
        dx, dy = 0, 0
    reason = payload.get("reason")
    return JudgeVerdict(
        verdict=verdict,
        dx=dx,
        dy=dy,
        confidence=confidence,
        reason=reason if isinstance(reason, str) else "",
        usage=usage,
    )


def build_openai_judge(*, client: Any, model: str, timeout_seconds: float) -> JudgeFn:
    system = load_qa_judge_system_prompt()
    user_instr = load_qa_judge_user_prompt()

    def _judge(req: JudgeRequest) -> JudgeVerdict:
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_instr + "\n\nfield_context_json:\n" + _field_context_json(req)},
                    {"type": "image_url", "image_url": {"url": _image_data_url(req.crop_path)}},
                ],
            },
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=openai_qa_response_format(),
            timeout=timeout_seconds,
        )
        raw = response.choices[0].message.content or "{}"
        payload = json.loads(raw)
        usage = _openai_usage(getattr(response, "usage", None))
        return _parse_verdict_payload(req.field_id, payload, usage)

    return _judge


def build_anthropic_judge(*, client: Any, model: str, timeout_seconds: float, max_tokens: int) -> JudgeFn:
    system = load_qa_judge_system_prompt()
    user_instr = load_qa_judge_user_prompt()

    def _judge(req: JudgeRequest) -> JudgeVerdict:
        content = [
            {"type": "text", "text": user_instr + "\n\nfield_context_json:\n" + _field_context_json(req)},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(req.crop_path.read_bytes()).decode("ascii"),
                },
            },
        ]
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            tools=[anthropic_qa_tool()],
            tool_choice=anthropic_qa_tool_choice(),
            timeout=timeout_seconds,
        )
        payload: dict[str, Any] | None = None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and isinstance(getattr(block, "input", None), dict):
                payload = block.input
                break
        if payload is None:
            raise QaRefinementError(f"Anthropic QA judge returned no tool payload for {req.field_id}.")
        usage = _anthropic_usage(getattr(response, "usage", None))
        return _parse_verdict_payload(req.field_id, payload, usage)

    return _judge


def _openai_usage(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    out: dict[str, int] = {}
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is not None:
        out["input_tokens"] = int(prompt)
    if completion is not None:
        out["output_tokens"] = int(completion)
    return out or None


def _anthropic_usage(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    out: dict[str, int] = {}
    inp = getattr(usage, "input_tokens", None)
    outp = getattr(usage, "output_tokens", None)
    if inp is not None:
        out["input_tokens"] = int(inp)
    if outp is not None:
        out["output_tokens"] = int(outp)
    return out or None


def _default_judge_from_settings(settings: Settings, *, grounder_provider: str) -> JudgeFn:
    prov, model = resolve_judge_provider_model(settings, grounder_provider=grounder_provider)
    if prov == "openai":
        if not settings.openai_api_key.strip():
            raise QaRefinementError("FORMIQO_OPENAI_API_KEY is missing for the QA judge (provider=openai).")
        from openai import OpenAI

        return build_openai_judge(
            client=OpenAI(api_key=settings.openai_api_key),
            model=model,
            timeout_seconds=settings.openai_timeout_seconds,
        )
    if not settings.anthropic_api_key.strip():
        raise QaRefinementError("FORMIQO_ANTHROPIC_API_KEY is missing for the QA judge (provider=anthropic).")
    from anthropic import Anthropic

    return build_anthropic_judge(
        client=Anthropic(api_key=settings.anthropic_api_key),
        model=model,
        timeout_seconds=settings.anthropic_timeout_seconds,
        max_tokens=settings.grounding_anthropic_max_tokens,
    )


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #


def _load_fields_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_fields_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #


def run_qa_refinement_for_job(
    *,
    job_id: str,
    output_dir: Path,
    settings: Settings,
    judge_fn: JudgeFn | None = None,
) -> dict[str, Any]:
    """Run the closed QA refinement loop over a job's grounded pages.

    ``judge_fn`` is injectable for tests; when omitted a provider judge is built from config
    (requires the relevant API key). Returns a summary dict also persisted to
    ``stages.qa_refine``.
    """
    from app.services.image_stamping import stamp_qa_preview_pages
    from app.services.jobs import job_grounding_provider_model, update_job_after_qa_refine, update_job_stage
    from app.services.stamping_config import (
        load_stamping_json_parsed,
        stamping_overrides,
        stamping_style,
    )

    configure_prompt_dir(settings.grounding_prompt_dir)
    job_root = output_dir.parent
    grounding_dir = output_dir / "field_grounding"
    if not grounding_dir.is_dir():
        raise QaRefinementError("field_grounding directory not found; run grounding first.")

    try:
        grounder_provider, _grounder_model = job_grounding_provider_model(job_root)
    except (FileNotFoundError, ValueError) as exc:
        raise QaRefinementError("Job grounding metadata not found; run grounding first.") from exc

    stamping = load_stamping_json_parsed(output_dir)
    base_style = stamping_style(stamping)
    style = StampStyle(
        font_size_pt=base_style.font_size_pt,
        text_color=base_style.text_color,
        draw_debug_boxes=True,
        debug_box_color="#ff0000",
    )
    overrides = stamping_overrides(stamping)
    values = dict(stamping.values)

    try:
        update_job_stage(job_root, "qa_refine", status="running", iterations=0, error=None)
    except FileNotFoundError:
        pass

    if judge_fn is None:
        judge_fn = _default_judge_from_settings(settings, grounder_provider=grounder_provider)

    max_iterations = settings.grounding_qa_max_iterations
    max_delta = settings.grounding_qa_max_bbox_delta_px
    qa_root = output_dir / "qa_refine"
    outcomes: dict[str, _FieldOutcome] = {}
    cost = _CostAccumulator()

    pages = discover_grounding_pages(grounding_dir)
    judged_pages = len(pages)
    iterations_run = 0
    converged = False

    for iteration in range(max_iterations):
        iterations_run = iteration + 1
        preview_run_dir = qa_root / f"iter_{iterations_run:02d}"
        stamp_qa_preview_pages(
            output_dir=output_dir,
            provider=grounder_provider,
            refined_grounding_dir=grounding_dir,
            preview_run_dir=preview_run_dir,
            values=values,
            style=style,
            overrides=overrides,
            require_all_values=False,
        )

        any_adjust = False
        for page_index, grounding_path in pages:
            preview_png = preview_run_dir / f"page_{page_index + 1:04d}.{grounder_provider}.stamped.png"
            payload = _load_fields_file(grounding_path)
            fields = payload.get("fields")
            if not isinstance(fields, list):
                continue
            page_w = int(payload.get("width_px") or payload.get("width") or 0)
            page_h = int(payload.get("height_px") or payload.get("height") or 0)

            page_deltas: dict[str, tuple[int, int]] = {}
            for field in fields:
                if not isinstance(field, dict) or not field_has_visible_value(field, values):
                    continue
                field_id = field["field_id"]
                bbox = {k: int(field["bbox"][k]) for k in ("x", "y", "w", "h")}
                crop_path = build_field_crop(
                    preview_png=preview_png,
                    bbox=bbox,
                    zoom=settings.grounding_qa_crop_zoom,
                    padding_px=settings.grounding_qa_crop_padding_px,
                    dst_png=preview_run_dir / "crops" / f"{field_id}.png",
                )
                if crop_path is None:
                    continue
                req = JudgeRequest(
                    field_id=field_id,
                    field_type=str(field.get("type", "")),
                    label=str(field.get("label", "")),
                    value=values.get(field_id, ""),
                    page_index=page_index,
                    bbox=bbox,
                    page_width=page_w,
                    page_height=page_h,
                    crop_zoom=settings.grounding_qa_crop_zoom,
                    crop_path=crop_path,
                )
                started = time.monotonic()
                verdict = judge_fn(req)
                cost.add(verdict.usage, latency=time.monotonic() - started)

                outcome = outcomes.setdefault(field_id, _FieldOutcome())
                outcome.last_verdict = verdict.verdict
                outcome.last_confidence = verdict.confidence
                if verdict.verdict == "adjust":
                    dx = clamp_delta(verdict.dx, max_delta=max_delta)
                    dy = clamp_delta(verdict.dy, max_delta=max_delta)
                    if dx != 0 or dy != 0:
                        page_deltas[field_id] = (dx, dy)

            translation: tuple[int, int] | None = None
            if settings.grounding_qa_consensus_translation_enabled and page_deltas:
                translation = merge_consensus_translation(
                    page_deltas,
                    min_fields=settings.grounding_qa_consensus_min_fields,
                    max_spread_px=settings.grounding_qa_consensus_max_spread_px,
                )

            if translation is not None and (translation[0] != 0 or translation[1] != 0):
                tdx = clamp_delta(translation[0], max_delta=max_delta)
                tdy = clamp_delta(translation[1], max_delta=max_delta)
                for field in fields:
                    if not isinstance(field, dict) or not isinstance(field.get("bbox"), dict):
                        continue
                    field["bbox"] = shift_bbox(field["bbox"], tdx, tdy, page_w=page_w, page_h=page_h)
                    fid = field.get("field_id")
                    if isinstance(fid, str):
                        oc = outcomes.setdefault(fid, _FieldOutcome())
                        oc.moved = True
                        oc.total_shift = (oc.total_shift[0] + tdx, oc.total_shift[1] + tdy)
                any_adjust = True
            elif page_deltas:
                for field in fields:
                    if not isinstance(field, dict):
                        continue
                    fid = field.get("field_id")
                    if fid not in page_deltas or not isinstance(field.get("bbox"), dict):
                        continue
                    dx, dy = page_deltas[fid]
                    field["bbox"] = shift_bbox(field["bbox"], dx, dy, page_w=page_w, page_h=page_h)
                    oc = outcomes.setdefault(fid, _FieldOutcome())
                    oc.moved = True
                    oc.total_shift = (oc.total_shift[0] + dx, oc.total_shift[1] + dy)
                any_adjust = True

            _write_fields_file(grounding_path, payload)

        if not any_adjust:
            converged = True
            break

    _finalize_qa_status(pages, outcomes, clean_confidence=settings.grounding_qa_clean_confidence)

    flagged = sum(1 for oc in outcomes.values() if _status_for(oc, settings.grounding_qa_clean_confidence) == "flagged")
    adjusted = sum(
        1 for oc in outcomes.values() if _status_for(oc, settings.grounding_qa_clean_confidence) == "adjusted"
    )
    confirmed = sum(
        1 for oc in outcomes.values() if _status_for(oc, settings.grounding_qa_clean_confidence) == "confirmed"
    )

    iterations_denom = max(1, iterations_run * max(1, judged_pages))
    tokens_per_page_iter = round(cost.total_tokens / iterations_denom, 2)
    summary = {
        "job_id": job_id,
        "provider": grounder_provider,
        "iterations": iterations_run,
        "max_iterations": max_iterations,
        "converged": converged,
        "pages": judged_pages,
        "fields_judged": len(outcomes),
        "confirmed": confirmed,
        "adjusted": adjusted,
        "flagged": flagged,
        "cost": {
            "judge_calls": cost.calls,
            "input_tokens": cost.input_tokens,
            "output_tokens": cost.output_tokens,
            "total_tokens": cost.total_tokens,
            "tokens_per_page_per_iteration": tokens_per_page_iter,
            "judge_latency_seconds": round(cost.latency_seconds, 3),
        },
    }

    try:
        update_job_after_qa_refine(
            job_root,
            iterations=iterations_run,
            converged=converged,
            confirmed=confirmed,
            adjusted=adjusted,
            flagged=flagged,
            cost=summary["cost"],
        )
    except FileNotFoundError:
        pass

    LOG.info(
        "qa_refine job_id=%s iterations=%d converged=%s judged=%d flagged=%d tokens=%d (%.2f/page/iter)",
        job_id,
        iterations_run,
        converged,
        len(outcomes),
        flagged,
        cost.total_tokens,
        tokens_per_page_iter,
    )
    return summary


def _status_for(outcome: _FieldOutcome, clean_confidence: float) -> str:
    if outcome.last_verdict == "adjust":
        return "flagged"
    if outcome.last_confidence < clean_confidence:
        return "flagged"
    return "adjusted" if outcome.moved else "confirmed"


def _finalize_qa_status(
    pages: list[tuple[int, Path]],
    outcomes: dict[str, _FieldOutcome],
    *,
    clean_confidence: float,
) -> None:
    """Write ``qa_status`` / ``qa_confidence`` back onto each judged field."""
    for _page_index, grounding_path in pages:
        payload = _load_fields_file(grounding_path)
        fields = payload.get("fields")
        if not isinstance(fields, list):
            continue
        changed = False
        for field in fields:
            if not isinstance(field, dict):
                continue
            field_id = field.get("field_id")
            if not isinstance(field_id, str) or field_id not in outcomes:
                continue
            outcome = outcomes[field_id]
            field["qa_status"] = _status_for(outcome, clean_confidence)
            field["qa_confidence"] = round(outcome.last_confidence, 4)
            changed = True
        if changed:
            _write_fields_file(grounding_path, payload)
