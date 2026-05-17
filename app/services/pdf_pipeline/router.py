"""Select OCR vs AcroForm pipeline implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.pdf_pipeline.acroform_pipeline import AcroFormPdfPipeline
from app.services.pdf_pipeline.errors import PdfPipelineError, XFA_UNSUPPORTED_USER_MESSAGE
from app.services.pdf_pipeline.ocr_pipeline import OcrPdfPipeline
from app.services.pdf_pipeline.types import PdfPipelineKind

LOG = logging.getLogger(__name__)


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
            raise PdfPipelineError(XFA_UNSUPPORTED_USER_MESSAGE)
        return OcrPdfPipeline().run(
            job_id=job_id,
            input_pdf=input_pdf,
            output_dir=output_dir,
            settings=settings,
            dpi=dpi,
            allow_rotated_pages=allow_rotated_pages,
        )
