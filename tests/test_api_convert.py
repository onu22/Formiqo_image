"""FastAPI integration tests (Swagger/OpenAPI served by the same app)."""

from __future__ import annotations

import io
import zipfile

import fitz
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_settings
from app.main import app


def _minimal_pdf_bytes() -> bytes:
    doc = fitz.open()
    try:
        doc.new_page(width=200, height=200)
        buf = io.BytesIO()
        doc.save(buf)
    finally:
        doc.close()
    return buf.getvalue()


@pytest.fixture
def api_client(tmp_path):
    get_settings.cache_clear()
    jobs = tmp_path / "jobs"
    app.dependency_overrides[get_settings] = lambda: Settings(
        jobs_dir=jobs,
        max_upload_bytes=8 * 1024 * 1024,
    )
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def test_openapi_docs_available(api_client: TestClient) -> None:
    r = api_client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert "openapi" in spec
    paths = spec["paths"]
    assert "/api/v1/convert" in paths


def test_convert_and_download_zip(api_client: TestClient) -> None:
    pdf = _minimal_pdf_bytes()
    files = {"file": ("sample.pdf", pdf, "application/pdf")}
    data = {"dpi": "72", "allow_rotated_pages": "false"}
    r = api_client.post("/api/v1/convert", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["page_count"] == 1
    assert body["dpi"] == 72.0
    job_id = body["job_id"]
    assert "archive_zip" in body["links"]

    zr = api_client.get(f"/api/v1/jobs/{job_id}/archive.zip")
    assert zr.status_code == 200, zr.text
    assert zr.headers.get("content-type") == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(zr.content))
    names = set(zf.namelist())
    assert "document_manifest.json" in names
    assert any(n.startswith("converted_images/page_") for n in names)

    mj = api_client.get(f"/api/v1/jobs/{job_id}/document_manifest.json")
    assert mj.status_code == 200
    assert len(mj.json()["pages"]) == 1


def test_convert_rejects_non_pdf_name(api_client: TestClient) -> None:
    pdf = _minimal_pdf_bytes()
    files = {"file": ("not-a.txt", pdf, "application/octet-stream")}
    r = api_client.post("/api/v1/convert", files=files, data={})
    assert r.status_code == 400
