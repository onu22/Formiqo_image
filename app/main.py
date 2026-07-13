"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api_tags import OPENAPI_TAGS
from app.dependencies import get_settings
from app.http_errors import ApiHttpError, api_http_error_handler
from app.routers import convert, grounding, jobs

LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "dist"
# Client-side route prefixes the SPA owns (see frontend/src/App.tsx). Anything else
# that isn't a real static asset falls through to a plain 404 rather than index.html,
# so path-traversal-style probes don't get masked as a 200 (see tests/test_g3_security.py).
_SPA_ROUTE_PREFIXES = ("jobs", "upload")


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
        application.mount(
            "/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="frontend-assets"
        )

        @application.get("/{full_path:path}", include_in_schema=False, response_model=None)
        async def serve_frontend(full_path: str) -> FileResponse | Response:
            """Serve the built SPA.

            Known client-side route prefixes (``jobs``, ``upload``) fall back to
            ``index.html`` so deep links and refreshes work. Anything else that
            isn't a real static file 404s rather than masking as a 200 index page.
            """
            dist_root = _FRONTEND_DIST.resolve()
            candidate = (_FRONTEND_DIST / full_path).resolve()
            if full_path and candidate.is_file() and dist_root in candidate.parents:
                return FileResponse(candidate)

            first_segment = full_path.split("/", 1)[0] if full_path else ""
            if full_path == "" or first_segment in _SPA_ROUTE_PREFIXES:
                return FileResponse(_FRONTEND_DIST / "index.html")

            return Response(status_code=404, content="Not found.")

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
