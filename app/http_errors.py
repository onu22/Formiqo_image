"""Structured API error responses per E3 contract."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


class ApiHttpError(Exception):
    """Raise from route handlers for ``{error, message}`` JSON bodies."""

    def __init__(self, status_code: int, error: str, message: str) -> None:
        self.status_code = status_code
        self.error = error
        self.message = message
        super().__init__(message)


async def api_http_error_handler(_request: Request, exc: ApiHttpError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.error, "message": exc.message},
    )


def job_not_found(job_id: str) -> ApiHttpError:
    return ApiHttpError(404, "job_not_found", f"Job not found: {job_id}")


def job_not_ready(message: str) -> ApiHttpError:
    return ApiHttpError(400, "job_not_ready", message)
