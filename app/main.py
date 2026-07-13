"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api_tags import OPENAPI_TAGS
from app.dependencies import get_settings
from app.http_errors import ApiHttpError, api_http_error_handler
from app.routers import convert, grounding, jobs

LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Jobs directory ready at %s", settings.jobs_dir.resolve())
    settings.user_uploads_dir.mkdir(parents=True, exist_ok=True)
    (settings.user_uploads_dir / "processed").mkdir(parents=True, exist_ok=True)
    (settings.user_uploads_dir / "failed").mkdir(parents=True, exist_ok=True)
    LOG.info("User uploads directory ready at %s", settings.user_uploads_dir.resolve())
    yield


def create_app() -> FastAPI:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )

    settings = get_settings()
    application = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        lifespan=lifespan,
        openapi_tags=OPENAPI_TAGS,
        description=(
            "Formiqo MVP API.\n\n"
            "**Jobs (UI API)** — upload, poll, edit fields, export.\n\n"
            "Legacy three-step pipeline endpoints remain for batch/CLI use."
        ),
    )

    application.add_exception_handler(ApiHttpError, api_http_error_handler)

    origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    if origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    application.include_router(jobs.router, prefix="/api/v1")
    application.include_router(convert.ingest_router, prefix="/api/v1")
    application.include_router(grounding.router, prefix="/api/v1")

    if _FRONTEND_DIST.is_dir():
        application.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
    else:

        @application.get("/", include_in_schema=False)
        def root() -> dict[str, str]:
            return {
                "service": settings.api_title,
                "docs": "/docs",
                "redoc": "/redoc",
            }

    return application


app = create_app()
