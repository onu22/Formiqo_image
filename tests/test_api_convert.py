"""FastAPI integration tests (Swagger/OpenAPI served by the same app)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

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
        grounding_provider="openai",
        grounding_model="gpt-4o",
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
    assert "/api/v1/jobs/{job_id}/ground-fields" in paths


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


def _create_conversion_job(api_client: TestClient) -> str:
    pdf = _minimal_pdf_bytes()
    files = {"file": ("sample.pdf", pdf, "application/pdf")}
    data = {"dpi": "72", "allow_rotated_pages": "false"}
    r = api_client.post("/api/v1/convert", files=files, data=data)
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def test_ground_fields_job_not_found(api_client: TestClient) -> None:
    r = api_client.post("/api/v1/jobs/11111111-1111-4111-8111-111111111111/ground-fields")
    assert r.status_code == 404


def test_ground_fields_missing_converted_images(api_client: TestClient) -> None:
    job_id = _create_conversion_job(api_client)
    settings = app.dependency_overrides[get_settings]()
    output_dir = settings.jobs_dir / job_id / "output"
    converted = output_dir / "converted_images"
    backup = output_dir / "converted_images_backup"
    converted.rename(backup)
    try:
        r = api_client.post(f"/api/v1/jobs/{job_id}/ground-fields")
        assert r.status_code == 400
    finally:
        backup.rename(converted)


def test_ground_fields_mixed_success_returns_200(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = _create_conversion_job(api_client)

    def fake_run_field_grounding_for_job(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        provider = kwargs["provider"]
        model = kwargs["model"]
        run_dir = f"field_grounding/{provider}_{model}"
        field_dir = output_dir / run_dir
        field_dir.mkdir(parents=True, exist_ok=True)

        page_file = field_dir / "page_0001.fields.json"
        page_file.write_text(
            json.dumps(
                {
                    "page_index": 0,
                    "width": 200,
                    "height": 200,
                    "unit": "px",
                    "origin": "top-left",
                    "fields": [
                        {
                            "field_id": "field_1",
                            "type": "text",
                            "bbox": {"x": 10, "y": 20, "w": 50, "h": 12},
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        manifest = field_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "job_id": kwargs["job_id"],
                    "provider": provider,
                    "model": model,
                    "run_id": "run_abc",
                    "run_dir": run_dir,
                    "created_at": "2026-01-01T00:00:00Z",
                    "page_count": 2,
                    "output_dir": run_dir,
                    "files": [f"{run_dir}/page_0001.fields.json"],
                    "succeeded_count": 1,
                    "failed_count": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "job_id": kwargs["job_id"],
            "provider": provider,
            "model": model,
            "run_id": "run_abc",
            "run_dir": run_dir,
            "page_count": 2,
            "succeeded_count": 1,
            "failed_count": 1,
            "output_dir": run_dir,
            "manifest_path": f"{run_dir}/manifest.json",
            "pages": [
                {
                    "page_index": 0,
                    "image_path": "converted_images/page_0001.png",
                    "status": "succeeded",
                    "output_file": f"{run_dir}/page_0001.fields.json",
                },
                {
                    "page_index": 1,
                    "image_path": "converted_images/page_0002.png",
                    "status": "failed",
                    "error": "validation failed",
                },
            ],
        }

    monkeypatch.setattr("app.routers.convert.run_field_grounding_for_job", fake_run_field_grounding_for_job)
    r = api_client.post(f"/api/v1/jobs/{job_id}/ground-fields")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4o"
    assert len(body["pages"]) == 2
    assert body["pages"][0]["output_file"] == "field_grounding/openai_gpt-4o/page_0001.fields.json"


def test_ground_fields_all_failed_returns_422(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = _create_conversion_job(api_client)

    def fake_all_failed(**kwargs):
        return {
            "job_id": kwargs["job_id"],
            "provider": kwargs["provider"],
            "model": kwargs["model"],
            "run_id": "run_fail",
            "run_dir": "field_grounding/openai_gpt-4o",
            "page_count": 1,
            "succeeded_count": 0,
            "failed_count": 1,
            "output_dir": "field_grounding/openai_gpt-4o",
            "manifest_path": "field_grounding/openai_gpt-4o/manifest.json",
            "pages": [
                {
                    "page_index": 0,
                    "image_path": "converted_images/page_0001.png",
                    "status": "failed",
                    "error": "invalid json",
                }
            ],
        }

    monkeypatch.setattr("app.routers.convert.run_field_grounding_for_job", fake_all_failed)
    r = api_client.post(f"/api/v1/jobs/{job_id}/ground-fields")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["message"] == "Field grounding failed for all pages."


def test_ground_fields_accepts_provider_model_override(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = _create_conversion_job(api_client)
    calls: list[dict] = []

    def fake_run_field_grounding_for_job(**kwargs):
        calls.append(kwargs)
        provider = kwargs["provider"]
        model = kwargs["model"]
        run_dir = f"field_grounding/{provider}_{model}"
        return {
            "job_id": kwargs["job_id"],
            "provider": provider,
            "model": model,
            "run_id": "run_custom",
            "run_dir": run_dir,
            "page_count": 1,
            "succeeded_count": 1,
            "failed_count": 0,
            "output_dir": run_dir,
            "manifest_path": f"{run_dir}/manifest.json",
            "pages": [
                {
                    "page_index": 0,
                    "image_path": "converted_images/page_0001.png",
                    "status": "succeeded",
                    "output_file": f"{run_dir}/page_0001.fields.json",
                }
            ],
        }

    monkeypatch.setattr("app.routers.convert.run_field_grounding_for_job", fake_run_field_grounding_for_job)
    r = api_client.post(
        f"/api/v1/jobs/{job_id}/ground-fields",
        json={"provider": "anthropic", "model": "claude-opus-4-7"},
    )
    assert r.status_code == 200, r.text
    assert calls[0]["provider"] == "anthropic"
    assert calls[0]["model"] == "claude-opus-4-7"
    assert r.json()["run_dir"] == "field_grounding/anthropic_claude-opus-4-7"


def test_ground_fields_side_by_side_runs_isolated(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = _create_conversion_job(api_client)

    def fake_run_field_grounding_for_job(**kwargs):
        provider = kwargs["provider"]
        model = kwargs["model"]
        run_dir = f"field_grounding/{provider}_{model}"
        return {
            "job_id": kwargs["job_id"],
            "provider": provider,
            "model": model,
            "run_id": f"run_{provider}",
            "run_dir": run_dir,
            "page_count": 1,
            "succeeded_count": 1,
            "failed_count": 0,
            "output_dir": run_dir,
            "manifest_path": f"{run_dir}/manifest.json",
            "pages": [
                {
                    "page_index": 0,
                    "image_path": "converted_images/page_0001.png",
                    "status": "succeeded",
                    "output_file": f"{run_dir}/page_0001.fields.json",
                }
            ],
        }

    monkeypatch.setattr("app.routers.convert.run_field_grounding_for_job", fake_run_field_grounding_for_job)
    r_openai = api_client.post(
        f"/api/v1/jobs/{job_id}/ground-fields",
        json={"provider": "openai", "model": "gpt-4o"},
    )
    r_anthropic = api_client.post(
        f"/api/v1/jobs/{job_id}/ground-fields",
        json={"provider": "anthropic", "model": "claude-opus-4-7"},
    )
    assert r_openai.status_code == 200
    assert r_anthropic.status_code == 200
    assert r_openai.json()["run_dir"] != r_anthropic.json()["run_dir"]
