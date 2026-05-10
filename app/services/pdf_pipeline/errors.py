"""Errors raised by automated PDF pipelines (non-HTTP)."""


class PdfPipelineError(Exception):
    """Recoverable pipeline failure with a message suitable for logs or API detail."""


class PdfIntakeArchiveError(PdfPipelineError):
    """Intake failed after a ``job_id`` was allocated (job folder was removed)."""

    def __init__(self, job_id: str, message: str | None = None):
        self.job_id = job_id
        super().__init__(message or "PDF intake failed after job allocation")
