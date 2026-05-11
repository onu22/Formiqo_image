"""Select OCR vs AcroForm pipeline implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.pdf_pipeline.acroform_pipeline import AcroFormPdfPipeline
from app.services.pdf_pipeline.errors import PdfPipelineError
from app.services.pdf_pipeline.ocr_pipeline import OcrPdfPipeline
from app.services.pdf_pipeline.types import PdfPipelineKind

LOG = logging.getLogger(__name__)

_XFA_UNSUPPORTED_MSG = (
    "This PDF uses XFA (Adobe LiveCycle) dynamic forms, which cannot be processed here—"
    "rasterizers only see the static fallback page, not the real form. "
    "Export or flatten to a standard PDF (for example with Adobe Acrobat) and upload again."
)


class PdfPipelineRouter:
    """Dispatches to the concrete pipeline for a detected PDF kind."""

    def run(
        self,
        *,
        kind: PdfPipelineKind,
        job_id: str,
        input_pdf: Path,
        output_dir: Path,
        settings: Settings,
        dpi: float = 200.0,
        allow_rotated_pages: bool = False,
    ) -> dict[str, Any]:
        LOG.info("PdfPipelineRouter job_id=%s dispatch=%s file=%s", job_id, kind.value, input_pdf.name)
        if kind == PdfPipelineKind.ACROFORM:
            return AcroFormPdfPipeline().run(job_id=job_id, input_pdf=input_pdf, output_dir=output_dir)
        if kind == PdfPipelineKind.XFA:
            LOG.warning("PdfPipelineRouter job_id=%s rejecting XFA file=%s", job_id, input_pdf.name)
            raise PdfPipelineError(_XFA_UNSUPPORTED_MSG)
        return OcrPdfPipeline().run(
            job_id=job_id,
            input_pdf=input_pdf,
            output_dir=output_dir,
            settings=settings,
            dpi=dpi,
            allow_rotated_pages=allow_rotated_pages,
        )
