"""PDF intake routing: AcroForm vs OCR pipelines."""

from app.services.pdf_pipeline.intake import process_pdf_from_path, scan_and_process_user_uploads
from app.services.pdf_pipeline.router import PdfPipelineRouter
from app.services.pdf_pipeline.types import PdfPipelineKind

__all__ = [
    "PdfPipelineKind",
    "PdfPipelineRouter",
    "process_pdf_from_path",
    "scan_and_process_user_uploads",
]
