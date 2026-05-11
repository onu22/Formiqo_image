"""Detect whether a PDF uses standard AcroForm widgets vs XFA vs flat content."""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from app.services.pdf_pipeline.types import PdfPipelineKind

_ACROFORM_REF_RE = re.compile(r"/AcroForm\s+(\d+)\s+\d+\s+R")


def _acroform_has_xfa(doc: fitz.Document) -> bool:
    """True if the catalog's AcroForm dictionary includes ``/XFA`` (LiveCycle / XML forms)."""
    try:
        catalog_xref = doc.pdf_catalog()
    except (AttributeError, RuntimeError, ValueError):
        return False
    catalog_obj = doc.xref_object(catalog_xref)
    m = _ACROFORM_REF_RE.search(catalog_obj)
    if not m:
        return False
    acro_xref = int(m.group(1))
    try:
        acro_obj = doc.xref_object(acro_xref)
    except (RuntimeError, ValueError):
        return False
    return "/XFA" in acro_obj


class PdfTypeDetector:
    """Classifies PDFs for routing: XFA and flat PDFs use OCR; classic widgets use AcroForm."""

    @staticmethod
    def detect(pdf_path: Path) -> PdfPipelineKind:
        """
        Routing order:

        1. **XFA** — AcroForm dict contains ``/XFA`` → ``PdfPipelineKind.XFA`` (OCR pipeline at runtime;
           fields are XML-driven, not reliable via classic widget APIs).

        2. **Standard AcroForm** — at least one page widget → ``ACROFORM``.

        3. **Flat / no widgets** — ``OCR``.
        """
        path = pdf_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        with fitz.open(path) as doc:
            if _acroform_has_xfa(doc):
                return PdfPipelineKind.XFA
            for page_index in range(doc.page_count):
                page = doc[page_index]
                widgets = page.widgets()
                if widgets is None:
                    continue
                first = next(widgets, None)
                if first is not None:
                    return PdfPipelineKind.ACROFORM
        return PdfPipelineKind.OCR
