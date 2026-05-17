"""Errors raised by automated PDF pipelines (non-HTTP)."""

XFA_UNSUPPORTED_USER_MESSAGE = (
    "This PDF uses XFA (Adobe LiveCycle) dynamic forms, which cannot be processed here—"
    "rasterizers only see the static fallback page, not the real form. "
    "Export or flatten to a standard PDF (for example with Adobe Acrobat) and upload again."
)


class PdfPipelineError(Exception):
    """Recoverable pipeline failure with a message suitable for logs or API detail."""


class PdfIntakeArchiveError(PdfPipelineError):
    """Intake failed after a ``job_id`` was allocated (job folder was removed)."""

    def __init__(self, job_id: str, message: str | None = None):
        self.job_id = job_id
        super().__init__(message or "PDF intake failed after job allocation")
