"""Detect whether a PDF uses interactive AcroForm widgets."""

from __future__ import annotations

from pathlib import Path

import fitz

from app.services.pdf_pipeline.types import PdfPipelineKind


class PdfTypeDetector:
    """Classifies PDFs as AcroForm-backed vs OCR (flat/scanned) pipeline."""

    @staticmethod
    def detect(pdf_path: Path) -> PdfPipelineKind:
        """
        If any page exposes form widgets, treat as AcroForm.

        PDFs with an AcroForm dictionary but zero widgets are classified as OCR.
        """
        path = pdf_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        with fitz.open(path) as doc:
            for page_index in range(doc.page_count):
                page = doc[page_index]
                widgets = page.widgets()
                if widgets is None:
                    continue
                first = next(widgets, None)
                if first is not None:
                    return PdfPipelineKind.ACROFORM
        return PdfPipelineKind.OCR
