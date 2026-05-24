"""Vision semantic field grounding using line maps (OpenAI or Anthropic)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from anthropic import Anthropic
from openai import OpenAI

from app.config import Settings
from app.grounding_field_types import stamping_type_for_field
from app.services.form_geometry import (
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


def resolve_grounding_model(*, provider: str, model: str | None, settings: Settings) -> tuple[str, str]:
    prov = provider.strip().lower()
    if prov not in _SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider {provider!r}; use openai or anthropic.")

    raw = (model or "").strip()
    if raw:
        return prov, raw

    if prov == "anthropic":
        resolved = settings.combined_default_anthropic_model.strip()
        if not resolved:
            raise ValueError(
                "Resolved Anthropic model is empty; set model in request or "
                "FORMIQO_COMBINED_DEFAULT_ANTHROPIC_MODEL."
            )
        return prov, resolved

    resolved = settings.grounding_model.strip() or settings.combined_default_openai_model.strip()
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
) -> GroundingLlmCallResult:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
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


def _call_anthropic_grounding_raw(
    *,
    client: Anthropic,
    model: str,
    system_text: str,
    user_content: list[dict[str, Any]],
    timeout_seconds: float,
    max_tokens: int,
) -> GroundingLlmCallResult:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_text,
        messages=[{"role": "user", "content": user_content}],
        timeout=timeout_seconds,
    )
    raw = _extract_anthropic_text(response)
    if not raw.strip():
        raise ValueError("Anthropic returned empty content.")
    stop_reason = getattr(response, "stop_reason", None)
    return GroundingLlmCallResult(
        raw_text=raw,
        finish_reason=str(stop_reason) if stop_reason is not None else None,
        max_output_tokens=max_tokens,
        usage=_anthropic_usage_dict(getattr(response, "usage", None)),
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
) -> GroundingLlmCallResult:
    slim_lines = settings.grounding_slim_line_detection
    if provider == "openai":
        if openai_client is None:
            raise ValueError("OpenAI client not configured.")
        messages = build_openai_messages(
            highlighted_png=paths["highlighted"],
            detected_lines=detected_lines,
            page_manifest=page_manifest,
            compact_json=compact_json,
            include_attachment_manifest=include_attachment_manifest,
            slim_line_detection=slim_lines,
        )
        return _call_openai_grounding_raw(
            client=openai_client,
            model=model,
            messages=messages,
            timeout_seconds=settings.openai_timeout_seconds,
            max_output_tokens=settings.grounding_openai_max_output_tokens,
        )

    if anthropic_client is None:
        raise ValueError("Anthropic client not configured.")
    system_text, user_content = build_anthropic_messages(
        highlighted_png=paths["highlighted"],
        detected_lines=detected_lines,
        page_manifest=page_manifest,
        compact_json=compact_json,
        include_attachment_manifest=include_attachment_manifest,
        slim_line_detection=slim_lines,
    )
    return _call_anthropic_grounding_raw(
        client=anthropic_client,
        model=model,
        system_text=system_text,
        user_content=user_content,
        timeout_seconds=settings.anthropic_timeout_seconds,
        max_tokens=settings.grounding_anthropic_max_tokens,
    )

    raise ValueError(f"Unsupported provider: {provider}")


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
) -> dict[str, Any]:
    """Call vision API and parse JSON; retry once with compact-json instructions on JSONDecodeError."""
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
    image_node = page_manifest.get("image")
    if not isinstance(image_node, dict):
        raise ValueError("Page manifest missing image object.")
    return int(image_node["saved_image_width_px"]), int(image_node["saved_image_height_px"])


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


def _field_for_stamp_storage(field: dict[str, Any]) -> dict[str, Any]:
    out = dict(field)
    ftype = field.get("type")
    if isinstance(ftype, str):
        out["type"] = stamping_type_for_field(ftype)
        if ftype != out["type"]:
            out["grounding_type"] = ftype
    return out


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
        stamp_payload = {
            "page_index": grounding.get("page_index", page_index),
            "width": grounding.get("width"),
            "height": grounding.get("height"),
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

    manifest = {
        "job_id": job_id,
        "provider": provider,
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(pages_meta),
        "pages": pages_meta,
    }
    manifest_path = fg_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    write_stamping_json_sample(fg_dir)

    return {
        "job_id": job_id,
        "provider": provider,
        "model": model,
        "run_dir": "field_grounding",
        "manifest_path": "field_grounding/manifest.json",
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

    succeeded: list[dict[str, Any]] = []
    failed_pages: list[dict[str, Any]] = []

    for page_index in targets:
        try:
            result = ground_one_page(
                job_id=job_id,
                output_dir=output_dir,
                page_index=page_index,
                provider=prov,
                model=resolved_model,
                settings=settings,
                openai_client=openai_client,
                anthropic_client=anthropic_client,
            )
            succeeded.append(result)
        except Exception as exc:
            LOG.warning("semantic grounding failed page %d: %s", page_index, exc)
            err_detail: Any = str(exc)
            failed_pages.append(
                {
                    "page_index": page_index,
                    "status": "failed",
                    "error": type(exc).__name__,
                    "detail": err_detail,
                }
            )

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
    return summary
