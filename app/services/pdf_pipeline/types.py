"""Classification of PDF processing strategies."""

from __future__ import annotations

from enum import Enum


class PdfPipelineKind(str, Enum):
    """Detected handling strategy for a PDF."""

    ACROFORM = "acroform"
    OCR = "ocr"
