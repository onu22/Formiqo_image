"""Vision semantic field grounding using line maps (OpenAI or Anthropic)."""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Literal

from anthropic import Anthropic
from openai import OpenAI

from app.config import Settings
from app.grounding_field_types import stamping_type_for_field
from app.services.form_geometry import (
    apply_anchor_grounding,
    build_geometry_index,
    load_detected_lines,
    normalize_page_grounding,
)
from app.services.grounding_prompt import (
    adapt_grounding_response,
    build_anthropic_messages,
    build_openai_messages,
    configure_prompt_dir,
)
from app.services.grounding_schema import (
    anthropic_grounding_tool,
    anthropic_tool_choice,
    openai_response_format,
)
from app.services.label_anchors import extract_label_anchors
from app.services.line_detection_job import list_converted_page_pngs
from app.services.stamping_config import write_stamping_json_sample

LOG = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_SUPPORTED_PROVIDERS = frozenset({"openai", "anthropic"})

OUTPUT_TRUNCATED_MESSAGE = (
    "Output truncated; increase FORMIQO_GROUNDING_OPENAI_MAX_OUTPUT_TOKENS or "
    "FORMIQO_GROUNDING_ANTHROPIC_MAX_TOKENS."
)


class OutputTruncatedError(ValueError):
    """Vision model hit the output token limit before completing JSON."""


class SemanticGroundingJobError(RuntimeError):
    """Entire job failed (no pages succeeded)."""

    def __init__(self, message: str, *, failed_pages: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.failed_pages: list[dict[str, Any]] = list(failed_pages or [])


@dataclass(frozen=True)
class GroundingLlmCallResult:
    raw_text: str
    finish_reason: str | None
    max_output_tokens: int
    usage: dict[str, int] | None
    # Populated when the provider returns a structured object directly (Anthropic tool-use),
    # so no text JSON parsing is needed.
    parsed: dict[str, Any] | None = dataclass_field(default=None)


def _openai_usage_dict(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)
    if prompt is None and completion is None and total is None:
        return None
    out: dict[str, int] = {}
    if prompt is not None:
        out["input_tokens"] = int(prompt)
    if completion is not None:
        out["output_tokens"] = int(completion)
    if total is not None:
        out["total_tokens"] = int(total)
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        reasoning = getattr(details, "reasoning_tokens", None)
        if reasoning is not None:
            out["reasoning_tokens"] = int(reasoning)
    return out or None


def _anthropic_usage_dict(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    inp = getattr(usage, "input_tokens", None)
    out_tok = getattr(usage, "output_tokens", None)
    if inp is None and out_tok is None:
        return None
    result: dict[str, int] = {}
    if inp is not None:
        result["input_tokens"] = int(inp)
    if out_tok is not None:
        result["output_tokens"] = int(out_tok)
    return result or None


def _grounding_attempt_label(*, compact_json: bool) -> Literal["initial", "compact_retry"]:
    if compact_json:
        return "compact_retry"
    return "initial"


def _log_grounding_llm_usage(
    *,
    job_id: str | None,
    page_index: int,
    provider: str,
    model: str,
    attempt: Literal["initial", "compact_retry"],
    call: GroundingLlmCallResult,
) -> None:
    truncated = _is_output_truncated(provider=provider, finish_reason=call.finish_reason)
    msg = (
        "grounding_llm job_id=%s page=%d provider=%s model=%s attempt=%s "
        "max_output_tokens=%d finish_reason=%s usage=%s raw_chars=%d"
    )
    args = (
        job_id,
        page_index,
        provider,
        model,
        attempt,
        call.max_output_tokens,
        call.finish_reason,
        call.usage,
        len(call.raw_text),
    )
    if truncated:
        LOG.warning(msg, *args)
    else:
        LOG.info(msg, *args)


_DEFAULT_ANTHROPIC_GROUNDING_MODEL = "claude-opus-4-7"


def resolve_grounding_model(*, provider: str, model: str | None, settings: Settings) -> tuple[str, str]:
    """Resolve the grounding model string, falling back to per-provider defaults.

    ``FORMIQO_GROUNDING_MODEL`` (``settings.grounding_model``) is the single configurable
    default, used for provider=openai; anthropic has no equivalent settings knob (its
    default is fixed here, matching the request-schema fallback in ``app.schemas``).
    """
    prov = provider.strip().lower()
    if prov not in _SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider {provider!r}; use openai or anthropic.")

    raw = (model or "").strip()
    if raw:
        return prov, raw

    if prov == "anthropic":
        return prov, _DEFAULT_ANTHROPIC_GROUNDING_MODEL

    resolved = settings.grounding_model.strip()
    if not resolved:
        raise ValueError(
            "Resolved OpenAI model is empty; set model in request or FORMIQO_GROUNDING_MODEL."
        )
    return prov, resolved


def _extract_json_text(content: str) -> str:
    text = content.strip()
    text = _JSON_FENCE_RE.sub("", text).strip()
    return text


def _is_output_truncated(*, provider: str, finish_reason: str | None) -> bool:
    if not finish_reason:
        return False
    reason = finish_reason.strip().lower()
    if provider == "openai":
        return reason == "length"
    return reason in {"max_tokens", "model_context_window_exceeded"}


def _call_openai_grounding_raw(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    timeout_seconds: float,
    max_output_tokens: int,
    structured: bool = False,
) -> GroundingLlmCallResult:
    response_format = openai_response_format() if structured else {"type": "json_object"}
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
        timeout=timeout_seconds,
        max_completion_tokens=max_output_tokens,
    )
    choice = response.choices[0]
    raw = choice.message.content or ""
    if not raw.strip():
        raise ValueError("OpenAI returned empty content.")
    finish_reason = getattr(choice, "finish_reason", None)
    return GroundingLlmCallResult(
        raw_text=raw,
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        max_output_tokens=max_output_tokens,
        usage=_openai_usage_dict(getattr(response, "usage", None)),
    )


def _extract_anthropic_text(response: Any) -> str:
    parts: list[str] = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts)


def _extract_anthropic_tool_input(response: Any) -> dict[str, Any] | None:
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            data = getattr(block, "input", None)
            if isinstance(data, dict):
                return data
    return None


def _call_anthropic_grounding_raw(
    *,
    client: Anthropic,
    model: str,
    system_text: str,
    user_content: list[dict[str, Any]],
    timeout_seconds: float,
    max_tokens: int,
    structured: bool = False,
) -> GroundingLlmCallResult:
    create_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_text,
        "messages": [{"role": "user", "content": user_content}],
        "timeout": timeout_seconds,
    }
    if structured:
        create_kwargs["tools"] = [anthropic_grounding_tool()]
        create_kwargs["tool_choice"] = anthropic_tool_choice()

    response = client.messages.create(**create_kwargs)
    stop_reason = getattr(response, "stop_reason", None)
    finish_reason = str(stop_reason) if stop_reason is not None else None
    usage = _anthropic_usage_dict(getattr(response, "usage", None))

    if structured:
        parsed = _extract_anthropic_tool_input(response)
        if parsed is None:
            raise ValueError("Anthropic tool-use returned no structured field payload.")
        return GroundingLlmCallResult(
            raw_text=json.dumps(parsed, separators=(",", ":")),
            finish_reason=finish_reason,
            max_output_tokens=max_tokens,
            usage=usage,
            parsed=parsed,
        )

    raw = _extract_anthropic_text(response)
    if not raw.strip():
        raise ValueError("Anthropic returned empty content.")
    return GroundingLlmCallResult(
        raw_text=raw,
        finish_reason=finish_reason,
        max_output_tokens=max_tokens,
        usage=usage,
    )


def _parse_grounding_json(*, provider: str, raw_text: str, finish_reason: str | None) -> dict[str, Any]:
    if _is_output_truncated(provider=provider, finish_reason=finish_reason):
        raise OutputTruncatedError(OUTPUT_TRUNCATED_MESSAGE)
    return json.loads(_extract_json_text(raw_text))


def _call_grounding_llm_raw(
    *,
    provider: str,
    model: str,
    settings: Settings,
    openai_client: OpenAI | None,
    anthropic_client: Anthropic | None,
    paths: dict[str, Path],
    detected_lines: dict[str, Any],
    page_manifest: dict[str, Any],
    compact_json: bool,
    include_attachment_manifest: bool,
    label_anchors: list[dict[str, Any]] | None = None,
    highlighted_png: Path | None = None,
) -> GroundingLlmCallResult:
    slim_lines = settings.grounding_slim_line_detection
    # Structured provider outputs are used on the initial attempt; the compact-JSON retry is
    # a thin text fallback that turns structured mode off.
    structured = settings.grounding_structured_outputs and not compact_json
    image_path = highlighted_png or paths["highlighted"]
    if provider == "openai":
        if openai_client is None:
            raise ValueError("OpenAI client not configured.")
        messages = build_openai_messages(
            highlighted_png=image_path,
            detected_lines=detected_lines,
            page_manifest=page_manifest,
            compact_json=compact_json,
            include_attachment_manifest=include_attachment_manifest,
            slim_line_detection=slim_lines,
            label_anchors=label_anchors,
        )
        return _call_openai_grounding_raw(
            client=openai_client,
            model=model,
            messages=messages,
            timeout_seconds=settings.openai_timeout_seconds,
            max_output_tokens=settings.grounding_openai_max_output_tokens,
            structured=structured,
        )

    if anthropic_client is None:
        raise ValueError("Anthropic client not configured.")
    system_text, user_content = build_anthropic_messages(
        highlighted_png=image_path,
        detected_lines=detected_lines,
        page_manifest=page_manifest,
        compact_json=compact_json,
        include_attachment_manifest=include_attachment_manifest,
        slim_line_detection=slim_lines,
        label_anchors=label_anchors,
    )
    return _call_anthropic_grounding_raw(
        client=anthropic_client,
        model=model,
        system_text=system_text,
        user_content=user_content,
        timeout_seconds=settings.anthropic_timeout_seconds,
        max_tokens=settings.grounding_anthropic_max_tokens,
        structured=structured,
    )


def _fetch_grounding_payload(
    *,
    job_id: str | None = None,
    provider: str,
    model: str,
    settings: Settings,
    openai_client: OpenAI | None,
    anthropic_client: Anthropic | None,
    page_index: int,
    paths: dict[str, Path],
    detected_lines: dict[str, Any],
    page_manifest: dict[str, Any],
    compact_json: bool = False,
    include_attachment_manifest: bool = True,
    label_anchors: list[dict[str, Any]] | None = None,
    highlighted_png: Path | None = None,
) -> dict[str, Any]:
    """Call the vision API and return an adapted payload.

    Structured provider outputs (OpenAI json_schema / Anthropic tool-use) make parse failures
    essentially impossible; the compact-JSON retry remains only as a thin fallback for the
    unstructured path.
    """
    attempt = _grounding_attempt_label(compact_json=compact_json)
    call = _call_grounding_llm_raw(
        provider=provider,
        model=model,
        settings=settings,
        openai_client=openai_client,
        anthropic_client=anthropic_client,
        paths=paths,
        detected_lines=detected_lines,
        page_manifest=page_manifest,
        compact_json=compact_json,
        include_attachment_manifest=include_attachment_manifest,
        label_anchors=label_anchors,
        highlighted_png=highlighted_png,
    )
    _log_grounding_llm_usage(
        job_id=job_id,
        page_index=page_index,
        provider=provider,
        model=model,
        attempt=attempt,
        call=call,
    )
    try:
        if call.parsed is not None:
            if _is_output_truncated(provider=provider, finish_reason=call.finish_reason):
                raise OutputTruncatedError(OUTPUT_TRUNCATED_MESSAGE)
            parsed = call.parsed
        else:
            parsed = _parse_grounding_json(
                provider=provider,
                raw_text=call.raw_text,
                finish_reason=call.finish_reason,
            )
        return adapt_grounding_response(parsed)
    except json.JSONDecodeError:
        if compact_json:
            raise
        LOG.info("grounding page %d JSONDecodeError; retrying with compact JSON instructions", page_index)
        return _fetch_grounding_payload(
            job_id=job_id,
            provider=provider,
            model=model,
            settings=settings,
            openai_client=openai_client,
            anthropic_client=anthropic_client,
            page_index=page_index,
            paths=paths,
            detected_lines=detected_lines,
            page_manifest=page_manifest,
            compact_json=True,
            include_attachment_manifest=include_attachment_manifest,
            label_anchors=label_anchors,
            highlighted_png=highlighted_png,
        )


def _page_paths(output_dir: Path, page_index: int) -> dict[str, Path]:
    stem = f"page_{page_index + 1:04d}"
    return {
        "highlighted": output_dir / "line_detection" / stem / "lines_highlighted.png",
        "lines_json": output_dir / "line_detection" / stem / "detected_lines.json",
        "page_manifest": output_dir / "converted_images" / "pages" / f"{stem}.json",
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _page_image_dimensions(page_manifest: dict[str, Any]) -> tuple[int, int]:
    from app.services.jobs import page_manifest_image_px

    return page_manifest_image_px(page_manifest)


def _apply_normalize(
    payload: dict[str, Any],
    *,
    geometry: dict[str, Any],
    page_manifest: dict[str, Any],
    settings: Settings,
    aggressive: bool = False,
) -> dict[str, Any]:
    img_w, img_h = _page_image_dimensions(page_manifest)
    return normalize_page_grounding(
        payload,
        geometry,
        stamp_inset_px=settings.grounding_stamp_inset_px,
        page_w=img_w,
        page_h=img_h,
        aggressive=aggressive,
    )


_PERSISTED_FIELD_DROP_KEYS = frozenset(
    {"nearby_label_text", "supporting_lines", "field_surface", "anchor", "_anchored"}
)


def _field_for_stamp_storage(field: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in field.items() if k not in _PERSISTED_FIELD_DROP_KEYS}
    evidence = out.get("evidence")
    if isinstance(evidence, dict):
        out["evidence"] = {k: v for k, v in evidence.items() if k != "label"}
    ftype = field.get("type")
    if isinstance(ftype, str):
        out["type"] = stamping_type_for_field(ftype)
        if ftype != out["type"]:
            out["grounding_type"] = ftype
    out.setdefault("grounding_source", None)
    out.setdefault("qa_status", None)
    out.setdefault("reviewed", False)
    out.setdefault("font_size_pt", None)
    return out


def _resolve_label_anchors(
    *,
    output_dir: Path,
    page_index: int,
    page_manifest: dict[str, Any],
    settings: Settings,
) -> list[dict[str, Any]]:
    """Best-effort label anchors from the digital PDF text layer (empty for scanned PDFs)."""
    if not settings.grounding_label_anchors:
        return []
    input_pdf = output_dir.parent / "input.pdf"
    try:
        return extract_label_anchors(
            input_pdf=input_pdf,
            page_index=page_index,
            page_manifest=page_manifest,
        )
    except Exception as exc:  # pragma: no cover - defensive; never fail grounding on anchors
        LOG.warning("label anchor extraction failed page=%d: %s", page_index, exc)
        return []


def _resolve_highlighted_image(
    *,
    paths: dict[str, Path],
    page_index: int,
    settings: Settings,
) -> Path:
    """Return the page image to send the model, optionally with a labeled coordinate grid."""
    base = paths["highlighted"]
    if not settings.grounding_grid_overlay_enabled:
        return base
    try:
        from app.services.grid_overlay import build_grid_overlay_image

        dst = base.with_name(base.stem + "_grid.png")
        return build_grid_overlay_image(
            base,
            dst,
            spacing_px=settings.grounding_grid_overlay_spacing_px,
        )
    except Exception as exc:  # pragma: no cover - overlay is an accuracy aid, never required
        LOG.warning("grid overlay failed page=%d: %s", page_index, exc)
        return base


def ground_one_page(
    *,
    job_id: str | None = None,
    output_dir: Path,
    page_index: int,
    provider: str,
    model: str,
    settings: Settings,
    openai_client: OpenAI | None,
    anthropic_client: Anthropic | None,
) -> dict[str, Any]:
    paths = _page_paths(output_dir, page_index)
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {key} for page {page_index}: {path}")

    detected_lines = load_detected_lines(paths["lines_json"])
    page_manifest = _read_json(paths["page_manifest"])
    geometry = build_geometry_index(
        detected_lines,
        line_padding_px=settings.grounding_line_padding_px,
    )

    label_anchors = _resolve_label_anchors(
        output_dir=output_dir,
        page_index=page_index,
        page_manifest=page_manifest,
        settings=settings,
    )
    highlighted_png = _resolve_highlighted_image(
        paths=paths,
        page_index=page_index,
        settings=settings,
    )

    raw_response = _fetch_grounding_payload(
        job_id=job_id,
        provider=provider,
        model=model,
        settings=settings,
        openai_client=openai_client,
        anthropic_client=anthropic_client,
        page_index=page_index,
        paths=paths,
        detected_lines=detected_lines,
        page_manifest=page_manifest,
        compact_json=False,
        include_attachment_manifest=True,
        label_anchors=label_anchors,
        highlighted_png=highlighted_png,
    )

    # Anchor-first: compute deterministic bboxes for fields that reference cells / lines /
    # labels before the geometry-snapping fallback runs on the remaining pixel estimates.
    img_w, img_h = _page_image_dimensions(page_manifest)
    raw_response = apply_anchor_grounding(
        raw_response,
        geometry,
        label_anchors,
        page_w=img_w,
        page_h=img_h,
        stamp_inset_px=settings.grounding_stamp_inset_px,
    )

    raw_response = _apply_normalize(
        raw_response,
        geometry=geometry,
        page_manifest=page_manifest,
        settings=settings,
    )
    return {
        "page_index": page_index,
        "grounding": raw_response,
    }


def write_field_grounding_outputs(
    *,
    job_id: str,
    output_dir: Path,
    provider: str,
    model: str,
    page_results: list[dict[str, Any]],
) -> dict[str, Any]:
    fg_dir = output_dir / "field_grounding"
    fg_dir.mkdir(parents=True, exist_ok=True)

    pages_meta: list[dict[str, Any]] = []
    for pr in page_results:
        page_index = int(pr["page_index"])
        grounding = pr["grounding"]
        stem = f"page_{page_index + 1:04d}"
        out_name = f"{stem}.fields.json"
        out_path = fg_dir / out_name

        stamp_fields = [_field_for_stamp_storage(f) for f in grounding.get("fields", []) if isinstance(f, dict)]
        width = grounding.get("width_px", grounding.get("width"))
        height = grounding.get("height_px", grounding.get("height"))
        stamp_payload = {
            "page_index": grounding.get("page_index", page_index),
            "width_px": width,
            "height_px": height,
            "unit": grounding.get("unit", "px"),
            "origin": grounding.get("origin", "top-left"),
            "fields": stamp_fields,
        }
        out_path.write_text(json.dumps(stamp_payload, indent=2) + "\n", encoding="utf-8")

        pages_meta.append(
            {
                "page_index": page_index,
                "status": "ok",
                "grounding_file": f"field_grounding/{out_name}",
            }
        )

    write_stamping_json_sample(fg_dir)

    from app.services.jobs import update_job_after_grounding

    job_root_dir = output_dir.parent
    update_job_after_grounding(
        job_root_dir,
        provider=provider,
        model=model,
        grounded_pages=len(pages_meta),
        total_pages=len(page_results),
        failed_pages=[],
    )

    return {
        "job_id": job_id,
        "provider": provider,
        "model": model,
        "run_dir": "field_grounding",
        "manifest_path": "job.json",
        "page_count": len(pages_meta),
        "succeeded_count": len(pages_meta),
        "failed_count": 0,
        "pages": pages_meta,
    }


def run_semantic_grounding_for_job(
    *,
    job_id: str,
    output_dir: Path,
    settings: Settings,
    provider: str = "openai",
    model: str | None = None,
) -> dict[str, Any]:
    prov, resolved_model = resolve_grounding_model(provider=provider, model=model, settings=settings)
    configure_prompt_dir(settings.grounding_prompt_dir)

    openai_client: OpenAI | None = None
    anthropic_client: Anthropic | None = None
    if prov == "openai":
        if not settings.openai_api_key.strip():
            raise ValueError("FORMIQO_OPENAI_API_KEY is missing for provider=openai.")
        openai_client = OpenAI(api_key=settings.openai_api_key)
    else:
        if not settings.anthropic_api_key.strip():
            raise ValueError("FORMIQO_ANTHROPIC_API_KEY is missing for provider=anthropic.")
        anthropic_client = Anthropic(api_key=settings.anthropic_api_key)

    page_entries = list_converted_page_pngs(output_dir)
    if not page_entries:
        raise ValueError("No converted page PNGs found; run conversion and line detection first.")

    targets = sorted(idx for idx, _ in page_entries)

    job_root_dir = output_dir.parent
    progress_lock = threading.Lock()
    try:
        from app.services.jobs import read_job_manifest, update_job_stage

        read_job_manifest(job_root_dir)
        update_job_stage(
            job_root_dir,
            "grounding",
            status="running",
            grounded_pages=0,
            total_pages=len(targets),
        )
    except FileNotFoundError:
        pass

    def _record_progress(done: int) -> None:
        """Thread-safe incremental grounded-page count for GET /jobs/{id} polling."""
        try:
            from app.services.jobs import update_job_stage as _update_stage

            with progress_lock:
                _update_stage(job_root_dir, "grounding", grounded_pages=done)
        except FileNotFoundError:
            pass

    def _ground(page_index: int) -> dict[str, Any]:
        return ground_one_page(
            job_id=job_id,
            output_dir=output_dir,
            page_index=page_index,
            provider=prov,
            model=resolved_model,
            settings=settings,
            openai_client=openai_client,
            anthropic_client=anthropic_client,
        )

    succeeded: list[dict[str, Any]] = []
    failed_pages: list[dict[str, Any]] = []

    # Bounded parallelism: a multi-page form grounds in roughly the time of its slowest
    # pages, not the sum. Each page is isolated so one failure never sinks the others.
    max_workers = max(1, min(settings.grounding_max_concurrency, len(targets)))
    results_by_page: dict[int, dict[str, Any]] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {executor.submit(_ground, idx): idx for idx in targets}
        for future in as_completed(future_to_page):
            page_index = future_to_page[future]
            try:
                results_by_page[page_index] = future.result()
                completed += 1
                _record_progress(completed)
            except Exception as exc:
                LOG.warning("semantic grounding failed page %d: %s", page_index, exc)
                failed_pages.append(
                    {
                        "page_index": page_index,
                        "status": "failed",
                        "error": type(exc).__name__,
                        "detail": str(exc),
                    }
                )

    succeeded = [results_by_page[idx] for idx in sorted(results_by_page)]
    failed_pages.sort(key=lambda fp: fp.get("page_index", 0))

    if not succeeded:
        raise SemanticGroundingJobError(
            f"Semantic grounding failed for all pages ({len(failed_pages)}).",
            failed_pages=failed_pages,
        )

    summary = write_field_grounding_outputs(
        job_id=job_id,
        output_dir=output_dir,
        provider=prov,
        model=resolved_model,
        page_results=succeeded,
    )
    summary["failed_count"] = len(failed_pages)
    summary["succeeded_count"] = len(succeeded)
    summary["page_count"] = len(succeeded) + len(failed_pages)
    summary["failed_pages"] = failed_pages

    if failed_pages:
        from app.services.jobs import read_job_manifest, update_job_stage, write_job_manifest

        job_root_dir = output_dir.parent
        manifest = read_job_manifest(job_root_dir)
        manifest["stages"]["grounding"]["failed_pages"] = failed_pages
        manifest["stages"]["grounding"]["grounded_pages"] = len(succeeded)
        manifest["stages"]["grounding"]["total_pages"] = len(succeeded) + len(failed_pages)
        write_job_manifest(job_root_dir, manifest)

    return summary
