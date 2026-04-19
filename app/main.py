"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.dependencies import get_settings
from app.routers import convert

LOG = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Jobs directory ready at %s", settings.jobs_dir.resolve())
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        lifespan=lifespan,
        description=(
            "Upload a PDF to produce per-page PNGs plus JSON manifests with exact "
            "image↔PDF coordinate mapping. OpenAPI / Swagger UI is served at `/docs`."
        ),
    )

    origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    if origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    application.include_router(convert.router, prefix="/api/v1")

    @application.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": settings.api_title,
            "docs": "/docs",
            "redoc": "/redoc",
            "convert": "/api/v1/convert",
        }

    return application


app = create_app()
