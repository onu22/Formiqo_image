"""Shared refine-grounding and stamp-pdf steps without FastAPI HTTP layer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.config import Settings
from app.schemas import StampingJson
from app.services.grounding_qa import run_grounding_qa_refinement_loop
from app.services.image_stamping import _assert_grounding_run_matches
from app.services.pdf_stamping import StampPdfStyle, run_pdf_stamping_for_job
from app.services.stamping_config import (
    load_field_grounding_manifest,
    load_stamping_json_parsed,
    manifest_provider_model,
    stamping_json_to_image_style,
)

LOG = logging.getLogger(__name__)


def load_stamping_or_raise(output_dir: Path) -> StampingJson:
    try:
        return load_stamping_json_parsed(output_dir)
    except FileNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    except ValidationError as exc:
        raise ValueError(f"Invalid stamping.json: {exc}") from exc


def manifest_provider_must_match(manifest: dict[str, Any], route_provider: str) -> tuple[str, str]:
    prov, model = manifest_provider_model(manifest)
    want = route_provider.strip().lower()
    if prov != want:
        raise ValueError(
            f"Grounding manifest provider is {prov!r}; use provider={prov!r} in stamp body, "
            f"not {want!r}.",
        )
    return prov, model


def run_refine_grounding_sync(*, job_id: str, output_dir: Path, settings: Settings) -> dict[str, Any]:
    grounding_dir = output_dir / "field_grounding"
    if not grounding_dir.is_dir():
        raise FileNotFoundError("field_grounding not found for this job.")

    manifest = load_field_grounding_manifest(output_dir)
    provider, model = manifest_provider_model(manifest)
    _assert_grounding_run_matches(grounding_dir, provider=provider, model=model)

    stamping = load_stamping_or_raise(output_dir)
    style = stamping_json_to_image_style(stamping)

    return run_grounding_qa_refinement_loop(
        job_id=job_id,
        output_dir=output_dir,
        provider=provider,
        model=model,
        values=stamping.values,
        style=style,
        require_all_values=stamping.require_all_values,
        openai_api_key=settings.openai_api_key,
        openai_timeout_seconds=settings.openai_timeout_seconds,
        openai_max_output_tokens=settings.grounding_openai_max_output_tokens,
        anthropic_api_key=settings.anthropic_api_key,
        anthropic_timeout_seconds=settings.anthropic_timeout_seconds,
        anthropic_max_tokens=settings.grounding_anthropic_max_tokens,
        max_iterations=settings.grounding_qa_max_iterations,
        max_bbox_delta_px=settings.grounding_qa_max_bbox_delta_px,
        consensus_translation_enabled=settings.grounding_qa_consensus_translation_enabled,
        consensus_min_fields=settings.grounding_qa_consensus_min_fields,
        consensus_max_spread_px=settings.grounding_qa_consensus_max_spread_px,
    )


def run_stamp_pdf_sync(
    *,
    job_id: str,
    input_pdf: Path,
    output_dir: Path,
    route_provider: str,
    settings: Settings,
) -> dict[str, Any]:
    if not input_pdf.is_file():
        raise FileNotFoundError(f"Input PDF not found for job: {input_pdf}")

    manifest = load_field_grounding_manifest(output_dir)
    provider, model = manifest_provider_must_match(manifest, route_provider)
    stamping = load_stamping_or_raise(output_dir)
    style = StampPdfStyle()

    result = run_pdf_stamping_for_job(
        job_id=job_id,
        input_pdf=input_pdf,
        output_dir=output_dir,
        provider=provider,
        model=model,
        values=stamping.values,
        style=style,
        require_all_values=stamping.require_all_values,
    )
    if result["succeeded_count"] == 0:
        raise RuntimeError("PDF stamping failed for all pages.")
    LOG.info(
        "stamp_pdf_sync job=%s provider=%s stamped_pages=%s",
        job_id,
        provider,
        result["succeeded_count"],
    )
    return result
