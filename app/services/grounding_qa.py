"""E5 vision QA refinement loop: stamp → judge → bounded delta → re-stamp.

Closed loop behind ``POST /api/v1/jobs/{id}/refine-grounding``:

1. Stamp a numbered debug preview of the current grounding (``stamp_qa_preview_pages``).
2. For every stampable field, crop a ~3x zoomed region around its stamped bbox and ask a
   vision **judge** (preferably a different provider than the grounder) a narrow question:
   is this value correctly positioned on its line / in its cell? If not, which direction and
   roughly how far?
3. Apply **bounded** per-axis deltas (``grounding_qa_max_bbox_delta_px``), merging a single
   consensus translation when many fields on a page agree (``grounding_qa_consensus_*``).
4. Re-stamp and iterate until the judge reports every field clean or
   ``grounding_qa_max_iterations`` is reached.

Per-field ``qa_status`` (``confirmed | adjusted | flagged``) and the final judge confidence
are persisted so the editor can surface low-trust fields. Judge token/latency cost is
accumulated for tuning the auto-run default (PRD E5 acceptance).
"""

from __future__ import annotations

import base64
import json
import logging
import statistics
import time
from dataclasses import dataclass, field as dataclass_field, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from app.config import Settings
from app.grounding_field_types import (
    is_supported_grounding_field_type,
    is_toggle_value_truthy,
    stamps_as_text,
    stamps_as_toggle,
)
from app.services.form_geometry import clamp_bbox_to_page
from app.services.grounding_prompt import (
    configure_prompt_dir,
    load_grounding_qa_system_prompt,
    load_grounding_qa_user_prompt,
)
from app.services.grounding_qa_schema import (
    anthropic_qa_tool,
    anthropic_qa_tool_choice,
    openai_qa_response_format,
)
from app.services.image_stamping import stamp_qa_preview_pages
from app.services.stamping_common import discover_grounding_pages
from app.services.stamping_config import (
    load_stamping_json_parsed,
    stamping_overrides,
    stamping_style,
)

LOG = logging.getLogger(__name__)

_SUPPORTED_PROVIDERS = frozenset({"openai", "anthropic"})
_DEFAULT_JUDGE_MODELS = {"openai": "gpt-5", "anthropic": "claude-opus-4-7"}

# Preview filename label; the QA preview run is isolated from user-facing stamp runs.
_QA_PREVIEW_PROVIDER = "qa"


class RefineGroundingError(RuntimeError):
    """Refinement cannot run (e.g. no fields, no judge configured)."""


@dataclass
class FieldVerdict:
    """A single-field placement verdict from the judge."""

    verdict: str  # "ok" | "shift"
    dx: int = 0
    dy: int = 0
    confidence: float | None = None
    reason: str | None = None


@dataclass
class JudgeUsage:
    """Accumulated judge cost for tuning the auto-run default (PRD E5)."""

    judge_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    def record(self, *, input_tokens: int, output_tokens: int, latency_s: float) -> None:
        self.judge_calls += 1
        self.input_tokens += max(0, int(input_tokens))
        self.output_tokens += max(0, int(output_tokens))
        self.latency_s += max(0.0, float(latency_s))

    def as_dict(self) -> dict[str, Any]:
        return {
            "judge_calls": self.judge_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": round(self.latency_s, 3),
        }


class FieldJudge(Protocol):
    """Judge interface; the real implementation calls a vision model per field crop."""

    usage: JudgeUsage

    def judge_field(
        self,
        *,
        page_index: int,
        field: dict[str, Any],
        value: str,
        crop: Image.Image,
        crop_offset: tuple[int, int],
    ) -> FieldVerdict: ...


# --------------------------------------------------------------------------- #
# Cropping + delta math (pure, unit-testable)
# --------------------------------------------------------------------------- #


def crop_field_region(
    image: Image.Image,
    bbox: dict[str, Any],
    *,
    zoom: float,
    padding: int,
    page_w: int,
    page_h: int,
) -> tuple[Image.Image, int, int]:
    """Crop a context region (~``zoom``x the bbox) centered on ``bbox``.

    Returns ``(crop_image, x0, y0)`` where ``(x0, y0)`` is the crop's top-left offset in
    full-page pixel coordinates, so judge deltas map back deterministically.
    """
    x = float(bbox["x"])
    y = float(bbox["y"])
    w = float(bbox["w"])
    h = float(bbox["h"])
    cx = x + w / 2.0
    cy = y + h / 2.0
    half_w = (w * zoom) / 2.0 + padding
    half_h = (h * zoom) / 2.0 + padding
    x0 = max(0, int(round(cx - half_w)))
    y0 = max(0, int(round(cy - half_h)))
    x1 = min(page_w, int(round(cx + half_w)))
    y1 = min(page_h, int(round(cy + half_h)))
    if x1 <= x0:
        x1 = min(page_w, x0 + 1)
    if y1 <= y0:
        y1 = min(page_h, y0 + 1)
    return image.crop((x0, y0, x1, y1)), x0, y0


def clamp_delta(value: int, bound: int) -> int:
    """Clamp a single-axis pixel delta to ``[-bound, bound]``."""
    return max(-bound, min(bound, int(value)))


def consensus_translation(
    deltas: list[tuple[int, int]],
    *,
    min_fields: int,
    max_spread: int,
) -> tuple[int, int] | None:
    """One translation for a page when enough per-field deltas agree tightly.

    Returns the median ``(dx, dy)`` when there are at least ``min_fields`` deltas and the
    spread (max-min) on each axis is within ``max_spread``; otherwise ``None`` (keep
    per-field deltas). Mirrors the ``grounding_qa_consensus_*`` config semantics.
    """
    if len(deltas) < min_fields:
        return None
    xs = [d[0] for d in deltas]
    ys = [d[1] for d in deltas]
    if (max(xs) - min(xs)) > max_spread or (max(ys) - min(ys)) > max_spread:
        return None
    return int(round(statistics.median(xs))), int(round(statistics.median(ys)))


def _is_stampable(field: dict[str, Any], value: str) -> bool:
    ftype = field.get("type")
    if not isinstance(ftype, str) or not is_supported_grounding_field_type(ftype):
        return False
    if stamps_as_text(ftype):
        return value != ""
    if stamps_as_toggle(ftype):
        return is_toggle_value_truthy(value)
    return False


def _field_bbox(field: dict[str, Any]) -> dict[str, int] | None:
    bbox = field.get("bbox")
    if not isinstance(bbox, dict):
        return None
    try:
        return {k: int(round(float(bbox[k]))) for k in ("x", "y", "w", "h")}
    except (KeyError, TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Vision judge (real provider-backed implementation)
# --------------------------------------------------------------------------- #


def _png_b64(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _field_metadata(
    *,
    page_index: int,
    field: dict[str, Any],
    value: str,
    bbox: dict[str, int],
    crop_offset: tuple[int, int],
) -> dict[str, Any]:
    return {
        "page_index": page_index,
        "field_id": field.get("field_id"),
        "type": field.get("type"),
        "label": field.get("label"),
        "value": value,
        "bbox": bbox,
        "crop_offset": {"x": crop_offset[0], "y": crop_offset[1]},
        "unit": "px",
        "origin": "top-left",
    }


def resolve_judge_provider_model(settings: Settings, *, grounder_provider: str) -> tuple[str, str]:
    """Resolve the judge ``(provider, model)``; empty config prefers a different provider."""
    prov = settings.grounding_qa_judge_provider.strip().lower()
    if prov and prov not in _SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported grounding_qa_judge_provider {prov!r}; use openai or anthropic.")
    grounder = grounder_provider.strip().lower()
    if not prov:
        other = "anthropic" if grounder == "openai" else "openai"
        if other == "anthropic" and settings.anthropic_api_key.strip():
            prov = other
        elif other == "openai" and settings.openai_api_key.strip():
            prov = other
        else:
            prov = grounder if grounder in _SUPPORTED_PROVIDERS else "openai"
    model = settings.grounding_qa_judge_model.strip() or _DEFAULT_JUDGE_MODELS[prov]
    return prov, model


class VisionJudge:
    """Provider-backed placement judge (OpenAI json_schema / Anthropic tool-use)."""

    def __init__(self, *, provider: str, model: str, settings: Settings) -> None:
        self.provider = provider
        self.model = model
        self.settings = settings
        self.usage = JudgeUsage()
        self._system = load_grounding_qa_system_prompt()
        self._user = load_grounding_qa_user_prompt()
        if provider == "openai":
            from openai import OpenAI

            self._client: Any = OpenAI(api_key=settings.openai_api_key)
        elif provider == "anthropic":
            from anthropic import Anthropic

            self._client = Anthropic(api_key=settings.anthropic_api_key)
        else:  # pragma: no cover - guarded by resolve_judge_provider_model
            raise ValueError(f"Unsupported judge provider {provider!r}.")

    def judge_field(
        self,
        *,
        page_index: int,
        field: dict[str, Any],
        value: str,
        crop: Image.Image,
        crop_offset: tuple[int, int],
    ) -> FieldVerdict:
        bbox = _field_bbox(field) or {"x": 0, "y": 0, "w": 0, "h": 0}
        meta = _field_metadata(
            page_index=page_index, field=field, value=value, bbox=bbox, crop_offset=crop_offset
        )
        user_text = self._user + "\n\nfield_metadata_json:\n" + json.dumps(meta, ensure_ascii=False)
        crop_b64 = _png_b64(crop)
        t0 = time.monotonic()
        if self.provider == "openai":
            verdict = self._judge_openai(user_text, crop_b64)
        else:
            verdict = self._judge_anthropic(user_text, crop_b64)
        return verdict

    # -- OpenAI ------------------------------------------------------------- #

    def _judge_openai(self, user_text: str, crop_b64: str) -> FieldVerdict:
        structured = self.settings.grounding_structured_outputs
        response_format = openai_qa_response_format() if structured else {"type": "json_object"}
        t0 = time.monotonic()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{crop_b64}"}},
                    ],
                },
            ],
            response_format=response_format,
            timeout=self.settings.openai_timeout_seconds,
            max_completion_tokens=self.settings.grounding_openai_max_output_tokens,
        )
        latency = time.monotonic() - t0
        raw = response.choices[0].message.content or "{}"
        usage = getattr(response, "usage", None)
        self.usage.record(
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_s=latency,
        )
        return _parse_verdict(json.loads(raw))

    # -- Anthropic ---------------------------------------------------------- #

    def _judge_anthropic(self, user_text: str, crop_b64: str) -> FieldVerdict:
        structured = self.settings.grounding_structured_outputs
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.settings.grounding_anthropic_max_tokens,
            "system": self._system,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": crop_b64},
                        },
                    ],
                }
            ],
            "timeout": self.settings.anthropic_timeout_seconds,
        }
        if structured:
            create_kwargs["tools"] = [anthropic_qa_tool()]
            create_kwargs["tool_choice"] = anthropic_qa_tool_choice()
        t0 = time.monotonic()
        response = self._client.messages.create(**create_kwargs)
        latency = time.monotonic() - t0
        usage = getattr(response, "usage", None)
        self.usage.record(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            latency_s=latency,
        )
        payload: dict[str, Any] | None = None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and isinstance(getattr(block, "input", None), dict):
                payload = block.input
                break
            if getattr(block, "type", None) == "text":
                try:
                    payload = json.loads(block.text)
                except (json.JSONDecodeError, TypeError):
                    payload = None
        if payload is None:
            raise ValueError("Anthropic judge returned no structured verdict.")
        return _parse_verdict(payload)


def _parse_verdict(data: dict[str, Any]) -> FieldVerdict:
    verdict = str(data.get("verdict", "ok")).strip().lower()
    if verdict not in ("ok", "shift"):
        verdict = "ok"
    dx = int(data.get("dx", 0) or 0)
    dy = int(data.get("dy", 0) or 0)
    conf_raw = data.get("confidence")
    confidence = float(conf_raw) if isinstance(conf_raw, (int, float)) else None
    reason = data.get("reason")
    return FieldVerdict(
        verdict=verdict,
        dx=dx,
        dy=dy,
        confidence=confidence,
        reason=reason if isinstance(reason, str) else None,
    )


def build_default_judge(settings: Settings, *, grounder_provider: str) -> FieldJudge | None:
    """Build the provider-backed judge, or ``None`` when no API key is configured."""
    provider, model = resolve_judge_provider_model(settings, grounder_provider=grounder_provider)
    key = settings.openai_api_key if provider == "openai" else settings.anthropic_api_key
    if not key.strip():
        LOG.warning("no API key for judge provider=%s; refine-grounding cannot run", provider)
        return None
    return VisionJudge(provider=provider, model=model, settings=settings)


# --------------------------------------------------------------------------- #
# The closed loop
# --------------------------------------------------------------------------- #


def _write_qa_working_dir(qa_dir: Path, page_payloads: dict[int, dict[str, Any]]) -> None:
    qa_dir.mkdir(parents=True, exist_ok=True)
    for page_index, payload in page_payloads.items():
        (qa_dir / f"page_{page_index + 1:04d}.fields.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def refine_grounding_for_job(
    *,
    job_id: str,
    output_dir: Path,
    settings: Settings,
    judge: FieldJudge | None = None,
) -> dict[str, Any]:
    """Run the closed QA refinement loop for a job and persist outcomes.

    ``judge`` may be injected (tests / custom pipelines); when omitted, a provider-backed
    judge is built from settings. Raises :class:`RefineGroundingError` when no fields exist
    or no judge is available.
    """
    configure_prompt_dir(settings.grounding_prompt_dir)
    job_root = output_dir.parent
    fg_dir = output_dir / "field_grounding"
    if not fg_dir.is_dir():
        raise RefineGroundingError("field_grounding directory not found")

    from app.services.jobs import job_grounding_provider_model, update_job_stage

    try:
        grounder_provider, _ = job_grounding_provider_model(job_root)
    except (FileNotFoundError, ValueError):
        grounder_provider = settings.grounding_provider

    if judge is None:
        judge = build_default_judge(settings, grounder_provider=grounder_provider)
    if judge is None:
        raise RefineGroundingError(
            "No QA judge available; set FORMIQO_OPENAI_API_KEY or FORMIQO_ANTHROPIC_API_KEY "
            "(or FORMIQO_GROUNDING_QA_JUDGE_PROVIDER/MODEL)."
        )

    pages = discover_grounding_pages(fg_dir)  # [(page_index, path)]
    page_paths = {pi: path for pi, path in pages}
    page_payloads: dict[int, dict[str, Any]] = {}
    for page_index, path in pages:
        page_payloads[page_index] = json.loads(path.read_text(encoding="utf-8"))

    stamping = load_stamping_json_parsed(output_dir)
    values = dict(stamping.values)
    style = replace(stamping_style(stamping), draw_debug_boxes=True)
    overrides = stamping_overrides(stamping)

    qa_dir = output_dir / "field_grounding_qa"
    qa_runs = output_dir / "qa_refine"

    max_iter = settings.grounding_qa_max_iterations
    bound = settings.grounding_qa_max_bbox_delta_px
    zoom = settings.grounding_qa_crop_zoom
    padding = settings.grounding_qa_crop_padding_px
    min_conf = settings.grounding_qa_min_confidence

    moved_ever: dict[tuple[int, str], bool] = {}
    last_verdict: dict[tuple[int, str], str] = {}
    last_confidence: dict[tuple[int, str], float | None] = {}

    converged = False
    iterations_run = 0
    last_preview_dir: Path | None = None

    for iteration in range(max_iter):
        iterations_run = iteration + 1
        _write_qa_working_dir(qa_dir, page_payloads)
        preview_run_dir = qa_runs / f"iter_{iteration + 1:02d}"
        stamp_qa_preview_pages(
            output_dir=output_dir,
            provider=_QA_PREVIEW_PROVIDER,
            refined_grounding_dir=qa_dir,
            preview_run_dir=preview_run_dir,
            values=values,
            style=style,
            require_all_values=False,
            overrides=overrides,
        )
        last_preview_dir = preview_run_dir
        any_adjusted = False

        for page_index, payload in page_payloads.items():
            stamped_png = preview_run_dir / f"page_{page_index + 1:04d}.{_QA_PREVIEW_PROVIDER}.stamped.png"
            if not stamped_png.is_file():
                continue
            try:
                page_w = int(payload["width_px"])
                page_h = int(payload["height_px"])
            except (KeyError, TypeError, ValueError):
                continue
            fields = payload.get("fields")
            if not isinstance(fields, list):
                continue

            with Image.open(stamped_png) as im:
                image = im.convert("RGB")

            shift_entries: list[tuple[dict[str, Any], int, int]] = []
            for field in fields:
                if not isinstance(field, dict):
                    continue
                field_id = field.get("field_id")
                if not isinstance(field_id, str) or not field_id:
                    continue
                bbox = _field_bbox(field)
                if bbox is None:
                    continue
                value = values.get(field_id, "")
                if not _is_stampable(field, value):
                    continue
                crop, ox, oy = crop_field_region(
                    image, bbox, zoom=zoom, padding=padding, page_w=page_w, page_h=page_h
                )
                key = (page_index, field_id)
                try:
                    verdict = judge.judge_field(
                        page_index=page_index,
                        field=field,
                        value=value,
                        crop=crop,
                        crop_offset=(ox, oy),
                    )
                except Exception as exc:  # noqa: BLE001 - per-field isolation
                    LOG.warning("judge failed page=%d field=%s: %s", page_index, field_id, exc)
                    continue
                last_verdict[key] = verdict.verdict
                last_confidence[key] = verdict.confidence
                if verdict.verdict == "shift" and (verdict.dx or verdict.dy):
                    shift_entries.append(
                        (field, clamp_delta(verdict.dx, bound), clamp_delta(verdict.dy, bound))
                    )

            consensus: tuple[int, int] | None = None
            if settings.grounding_qa_consensus_translation_enabled and shift_entries:
                consensus = consensus_translation(
                    [(dxc, dyc) for _, dxc, dyc in shift_entries],
                    min_fields=settings.grounding_qa_consensus_min_fields,
                    max_spread=settings.grounding_qa_consensus_max_spread_px,
                )

            for field, dxc, dyc in shift_entries:
                if consensus is not None:
                    dxc, dyc = consensus
                if dxc == 0 and dyc == 0:
                    continue
                bbox = _field_bbox(field)
                if bbox is None:
                    continue
                new_bbox = clamp_bbox_to_page(
                    {"x": bbox["x"] + dxc, "y": bbox["y"] + dyc, "w": bbox["w"], "h": bbox["h"]},
                    page_w,
                    page_h,
                )
                if new_bbox != bbox:
                    field["bbox"] = new_bbox
                    moved_ever[(page_index, field["field_id"])] = True
                    any_adjusted = True

        if not any_adjusted:
            converged = True
            break

    counts = {"confirmed": 0, "adjusted": 0, "flagged": 0}
    for page_index, payload in page_payloads.items():
        fields = payload.get("fields")
        if not isinstance(fields, list):
            continue
        for field in fields:
            if not isinstance(field, dict):
                continue
            field_id = field.get("field_id")
            if not isinstance(field_id, str):
                continue
            key = (page_index, field_id)
            lv = last_verdict.get(key)
            if lv is None:
                continue  # unevaluated (skipped or judge error) — leave qa_status unchanged
            conf = last_confidence.get(key)
            if lv == "shift":
                status = "flagged"
            elif conf is not None and conf < min_conf:
                status = "flagged"
            elif moved_ever.get(key):
                status = "adjusted"
            else:
                status = "confirmed"
            field["qa_status"] = status
            if conf is not None:
                field["confidence"] = round(conf, 4)
            counts[status] += 1

    for page_index, path in page_paths.items():
        path.write_text(json.dumps(page_payloads[page_index], indent=2) + "\n", encoding="utf-8")

    usage = judge.usage.as_dict()
    page_count = len(page_payloads)
    denom = max(1, page_count * iterations_run)
    cost = dict(usage)
    cost["input_tokens_per_page_per_iteration"] = round(usage["input_tokens"] / denom, 2)
    cost["output_tokens_per_page_per_iteration"] = round(usage["output_tokens"] / denom, 2)

    _prune_preview_runs(qa_runs, keep=last_preview_dir)

    update_job_stage(
        job_root,
        "qa_refine",
        status="done",
        error=None,
        iterations=iterations_run,
        converged=converged,
        page_count=page_count,
        counts=counts,
        cost=cost,
    )

    return {
        "job_id": job_id,
        "iterations": iterations_run,
        "converged": converged,
        "page_count": page_count,
        "counts": counts,
        "cost": cost,
    }


def run_refine_grounding_task(
    *,
    job_id: str,
    job_root: Path,
    output_dir: Path,
    settings: Settings,
    judge: FieldJudge | None = None,
) -> dict[str, Any] | None:
    """Run the refine loop with job.json stage bookkeeping and error isolation.

    Used both as the ``POST /refine-grounding`` background task and the optional auto-run
    final pipeline stage. Never raises; failures are recorded in ``stages.qa_refine``.
    """
    from app.services.jobs import update_job_stage

    try:
        update_job_stage(job_root, "qa_refine", status="running", error=None)
    except FileNotFoundError:
        pass
    try:
        return refine_grounding_for_job(
            job_id=job_id, output_dir=output_dir, settings=settings, judge=judge
        )
    except Exception as exc:  # noqa: BLE001 - background/auto-run must not crash the caller
        LOG.warning("refine-grounding failed job_id=%s: %s", job_id, exc)
        try:
            update_job_stage(job_root, "qa_refine", status="failed", error=str(exc))
        except FileNotFoundError:
            pass
        return None


def _prune_preview_runs(qa_runs: Path, *, keep: Path | None) -> None:
    """Keep only the final iteration's preview to bound disk usage."""
    if not qa_runs.is_dir():
        return
    for child in qa_runs.iterdir():
        if child.is_dir() and child != keep:
            import shutil

            shutil.rmtree(child, ignore_errors=True)
