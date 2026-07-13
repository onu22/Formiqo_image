"""OpenAPI tag names and order for Swagger / ReDoc (matches the PDF stamping pipeline)."""

from __future__ import annotations

TAG_PREPARE_PDF = "1. Prepare PDF"
TAG_LOCATE_FIELDS = "2. Locate form fields"
TAG_FILL_EXPORT = "3. Fill & export"
TAG_JOBS = "Jobs (UI API)"

OPENAPI_TAGS: list[dict[str, str]] = [
    {
        "name": TAG_JOBS,
        "description": (
            "Primary UI API: upload a PDF, poll job status, edit fields and values, "
            "preview stamped images, and export the final PDF."
        ),
    },
    {
        "name": TAG_PREPARE_PDF,
        "description": (
            "Place a PDF in the uploads folder, then run conversion: each page becomes an image "
            "and OpenCV detects printed lines and table structure. You receive a job_id."
        ),
    },
    {
        "name": TAG_LOCATE_FIELDS,
        "description": (
            "AI reviews each page image plus the line map to find fillable areas — text boxes, "
            "checkboxes, and radio buttons — and saves field positions plus a sample stamping.json."
        ),
    },
    {
        "name": TAG_FILL_EXPORT,
        "description": (
            "Edit stamping.json with the values you want on the form, then generate filled page "
            "previews (PNG) or a completed PDF."
        ),
    },
]
