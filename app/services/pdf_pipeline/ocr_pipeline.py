"""Full OCR pipeline: convert-and-ground, refine-grounding, stamp-pdf."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.convert_and_ground_job import run_convert_and_ground_sync
from app.services.pdf_pipeline.errors import PdfPipelineError
from app.services.refine_and_stamp_jobs import run_refine_grounding_sync, run_stamp_pdf_sync

LOG = logging.getLogger(__name__)


class OcrPdfPipeline:
    """Chains existing vision OCR services without duplicating coordinate logic."""

    def run(
        self,
        *,
        job_id: str,
        input_pdf: Path,
        output_dir: Path,
        settings: Settings,
        dpi: float = 200.0,
        allow_rotated_pages: bool = False,
        provider: str = "anthropic",
        model: str | None = None,
    ) -> dict[str, Any]:
        provider_norm = provider.strip().lower()
        if provider_norm not in ("anthropic", "openai"):
            raise PdfPipelineError(f"Unsupported provider: {provider!r}")
        resolved_model = (model or "").strip()
        if not resolved_model:
            resolved_model = (
                settings.combined_default_anthropic_model.strip()
                if provider_norm == "anthropic"
                else settings.combined_default_openai_model.strip()
            )
        if not resolved_model:
            raise PdfPipelineError("Resolved vision model is empty; configure FORMIQO_COMBINED_DEFAULT_*.")

        LOG.info(
            "ocr_pipeline start job=%s provider=%s model=%s dpi=%s",
            job_id,
            provider_norm,
            resolved_model,
            dpi,
        )

        try:
            cg = run_convert_and_ground_sync(
                job_id=job_id,
                input_pdf=input_pdf,
                output_dir=output_dir,
                dpi=dpi,
                allow_rotated_pages=allow_rotated_pages,
                provider=provider_norm,
                model=resolved_model,
                settings=settings,
                source_filename=input_pdf.name,
            )
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            raise PdfPipelineError(str(exc)) from exc

        try:
            refine_result = run_refine_grounding_sync(job_id=job_id, output_dir=output_dir, settings=settings)
        except (FileNotFoundError, ValueError) as exc:
            raise PdfPipelineError(str(exc)) from exc

        try:
            stamp_result = run_stamp_pdf_sync(
                job_id=job_id,
                input_pdf=input_pdf,
                output_dir=output_dir,
                route_provider=provider_norm,
                settings=settings,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise PdfPipelineError(str(exc)) from exc

        LOG.info(
            "ocr_pipeline done job=%s pipeline=ocr promoted=%s stamped_pages=%s",
            job_id,
            refine_result.get("promoted"),
            stamp_result.get("succeeded_count"),
        )

        gr = cg["ground_result"]
        return {
            "pipeline": "ocr",
            "job_id": job_id,
            "provider": provider_norm,
            "model": resolved_model,
            "convert_and_ground": {
                "page_count": cg["page_count"],
                "grounding_succeeded_count": gr["succeeded_count"],
            },
            "refine_grounding": refine_result,
            "stamp_pdf": stamp_result,
        }
