"""FastAPI integration tests (provider-specific endpoints)."""

from __future__ import annotations

import io
import json
import uuid
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_settings
from app.main import app
from app.services.conversion import run_convert_pdf_to_images
from app.services.jobs import job_paths


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
        combined_default_anthropic_model="claude-opus-4-7",
        combined_default_openai_model="gpt-5.5",
    )
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def test_openapi_docs_provider_split(api_client: TestClient) -> None:
    r = api_client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    paths = spec["paths"]
    assert "/api/v1/convert-and-ground/anthropic" in paths
    assert "/api/v1/convert-and-ground/openai" in paths
    assert "/api/v1/jobs/{job_id}/stamp-images/anthropic" in paths
    assert "/api/v1/jobs/{job_id}/stamp-images/openai" in paths
    assert "/api/v1/jobs/{job_id}/stamp-pdf/anthropic" in paths
    assert "/api/v1/jobs/{job_id}/stamp-pdf/openai" in paths
    assert "/api/v1/convert" not in paths
    assert "/api/v1/jobs/{job_id}/ground-fields" not in paths
    assert "/api/v1/convert-and-ground" not in paths
    anthropic_schema_ref = paths["/api/v1/convert-and-ground/anthropic"]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]["$ref"]
    openai_schema_ref = paths["/api/v1/convert-and-ground/openai"]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]["$ref"]
    anthropic_schema_name = anthropic_schema_ref.rsplit("/", 1)[-1]
    openai_schema_name = openai_schema_ref.rsplit("/", 1)[-1]
    assert "model" not in spec["components"]["schemas"][anthropic_schema_name]["properties"]
    assert "model" not in spec["components"]["schemas"][openai_schema_name]["properties"]
    stamp_anthropic_schema = paths["/api/v1/jobs/{job_id}/stamp-images/anthropic"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    stamp_openai_schema = paths["/api/v1/jobs/{job_id}/stamp-images/openai"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    stamp_anthropic_ref = stamp_anthropic_schema.get("$ref")
    stamp_openai_ref = stamp_openai_schema.get("$ref")
    if stamp_anthropic_ref:
        stamp_anthropic_name = stamp_anthropic_ref.rsplit("/", 1)[-1]
        anthropic_props = spec["components"]["schemas"][stamp_anthropic_name]["properties"]
    else:
        anthropic_props = stamp_anthropic_schema.get("properties", {})
    if stamp_openai_ref:
        stamp_openai_name = stamp_openai_ref.rsplit("/", 1)[-1]
        openai_props = spec["components"]["schemas"][stamp_openai_name]["properties"]
    else:
        openai_props = stamp_openai_schema.get("properties", {})
    assert "model" not in anthropic_props
    assert "model" not in openai_props
    stamp_pdf_anthropic_schema = paths["/api/v1/jobs/{job_id}/stamp-pdf/anthropic"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    stamp_pdf_openai_schema = paths["/api/v1/jobs/{job_id}/stamp-pdf/openai"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    stamp_pdf_anthropic_ref = stamp_pdf_anthropic_schema.get("$ref")
    stamp_pdf_openai_ref = stamp_pdf_openai_schema.get("$ref")
    if stamp_pdf_anthropic_ref:
        stamp_pdf_anthropic_name = stamp_pdf_anthropic_ref.rsplit("/", 1)[-1]
        stamp_pdf_anthropic_props = spec["components"]["schemas"][stamp_pdf_anthropic_name]["properties"]
    else:
        stamp_pdf_anthropic_props = stamp_pdf_anthropic_schema.get("properties", {})
    if stamp_pdf_openai_ref:
        stamp_pdf_openai_name = stamp_pdf_openai_ref.rsplit("/", 1)[-1]
        stamp_pdf_openai_props = spec["components"]["schemas"][stamp_pdf_openai_name]["properties"]
    else:
        stamp_pdf_openai_props = stamp_pdf_openai_schema.get("properties", {})
    assert "model" not in stamp_pdf_anthropic_props
    assert "model" not in stamp_pdf_openai_props
    assert "style" not in stamp_pdf_anthropic_props
    assert "style" not in stamp_pdf_openai_props


def _create_job_via_provider_endpoint(api_client: TestClient, provider: str) -> str:
    pdf = _minimal_pdf_bytes()
    files = {"file": ("sample.pdf", pdf, "application/pdf")}
    r = api_client.post(f"/api/v1/convert-and-ground/{provider}", files=files, data={"dpi": "72"})
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def _create_converted_job_only() -> str:
    settings = app.dependency_overrides[get_settings]()
    job_id = str(uuid.uuid4())
    root, input_pdf, output_dir = job_paths(settings.jobs_dir, job_id)
    root.mkdir(parents=True, exist_ok=True)
    input_pdf.write_bytes(_minimal_pdf_bytes())
    run_convert_pdf_to_images(
        str(input_pdf),
        str(output_dir),
        72.0,
        overwrite=True,
        allow_rotated_pages=False,
    )
    return job_id


def test_convert_and_ground_anthropic_happy_path(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_ground(**kwargs):
        calls.append((kwargs["provider"], kwargs["model"]))
        run_dir = f"field_grounding/{kwargs['provider']}_{kwargs['model']}"
        return {
            "job_id": kwargs["job_id"],
            "provider": kwargs["provider"],
            "model": kwargs["model"],
            "run_id": "run1",
            "run_dir": run_dir,
            "page_count": 1,
            "succeeded_count": 1,
            "failed_count": 0,
            "output_dir": run_dir,
            "manifest_path": f"{run_dir}/manifest.json",
            "stamping_sample_path": f"{run_dir}/stamping.json",
            "pages": [],
        }

    monkeypatch.setattr("app.routers.convert.run_field_grounding_for_job", fake_ground)
    pdf = _minimal_pdf_bytes()
    files = {"file": ("sample.pdf", pdf, "application/pdf")}
    r = api_client.post("/api/v1/convert-and-ground/anthropic", files=files, data={"dpi": "72"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert calls == [("anthropic", "claude-opus-4-7")]
    assert body["grounding"]["provider"] == "anthropic"
    assert body["grounding"]["model"] == "claude-opus-4-7"
    assert body["grounding"]["stamping_sample_path"] == "field_grounding/anthropic_claude-opus-4-7/stamping.json"


def test_convert_and_ground_openai_uses_default_model(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_ground(**kwargs):
        calls.append((kwargs["provider"], kwargs["model"]))
        run_dir = f"field_grounding/{kwargs['provider']}_{kwargs['model']}"
        return {
            "job_id": kwargs["job_id"],
            "provider": kwargs["provider"],
            "model": kwargs["model"],
            "run_id": "run2",
            "run_dir": run_dir,
            "page_count": 1,
            "succeeded_count": 1,
            "failed_count": 0,
            "output_dir": run_dir,
            "manifest_path": f"{run_dir}/manifest.json",
            "stamping_sample_path": f"{run_dir}/stamping.json",
            "pages": [],
        }

    monkeypatch.setattr("app.routers.convert.run_field_grounding_for_job", fake_ground)
    pdf = _minimal_pdf_bytes()
    files = {"file": ("sample.pdf", pdf, "application/pdf")}
    r = api_client.post("/api/v1/convert-and-ground/openai", files=files, data={"dpi": "72"})
    assert r.status_code == 200, r.text
    assert calls == [("openai", "gpt-5.5")]
    assert r.json()["grounding"]["model"] == "gpt-5.5"
    assert r.json()["grounding"]["stamping_sample_path"] == "field_grounding/openai_gpt-5.5/stamping.json"


def test_convert_and_ground_non_pdf_rejected(api_client: TestClient) -> None:
    files = {"file": ("not-a.txt", b"abc", "application/octet-stream")}
    r = api_client.post("/api/v1/convert-and-ground/openai", files=files, data={})
    assert r.status_code == 400


def test_convert_and_ground_all_failed_returns_422(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_failed(**kwargs):
        run_dir = f"field_grounding/{kwargs['provider']}_{kwargs['model']}"
        return {
            "job_id": kwargs["job_id"],
            "provider": kwargs["provider"],
            "model": kwargs["model"],
            "run_id": "run3",
            "run_dir": run_dir,
            "page_count": 1,
            "succeeded_count": 0,
            "failed_count": 1,
            "output_dir": run_dir,
            "manifest_path": f"{run_dir}/manifest.json",
            "stamping_sample_path": f"{run_dir}/stamping.json",
            "pages": [{"page_index": 0, "status": "failed", "error": "invalid json", "image_path": "x"}],
        }

    monkeypatch.setattr("app.routers.convert.run_field_grounding_for_job", fake_failed)
    pdf = _minimal_pdf_bytes()
    files = {"file": ("sample.pdf", pdf, "application/pdf")}
    r = api_client.post("/api/v1/convert-and-ground/anthropic", files=files, data={"dpi": "72"})
    assert r.status_code == 422
    assert r.json()["detail"]["message"] == "Convert succeeded but grounding failed for all pages."


def _converted_image_dimensions(output_dir: Path, page_index: int = 0) -> tuple[int, int]:
    page_manifest = output_dir / "converted_images" / "pages" / f"page_{page_index + 1:04d}.json"
    data = json.loads(page_manifest.read_text(encoding="utf-8"))
    image = data["image"]
    return int(image["saved_image_width_px"]), int(image["saved_image_height_px"])


def _write_field_grounding_file(
    output_dir: Path,
    *,
    provider: str,
    model: str,
    width: int,
    height: int,
) -> None:
    run_dir = output_dir / "field_grounding" / f"{provider}_{model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "page_0001.fields.json").write_text(
        json.dumps(
            {
                "page_index": 0,
                "width": width,
                "height": height,
                "unit": "px",
                "origin": "top-left",
                "fields": [
                    {
                        "field_id": "first_name",
                        "type": "text",
                        "bbox": {"x": 10, "y": 20, "w": 50, "h": 12},
                    },
                    {
                        "field_id": "last_name",
                        "type": "text",
                        "bbox": {"x": 70, "y": 20, "w": 50, "h": 12},
                    },
                ],
                "provider": provider,
                "model": model,
                "run_id": "run_test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "job_id": "test",
                "provider": provider,
                "model": model,
                "run_id": "run_test",
                "run_dir": f"field_grounding/{provider}_{model}",
                "created_at": "2026-01-01T00:00:00Z",
                "page_count": 1,
                "output_dir": f"field_grounding/{provider}_{model}",
                "files": [f"field_grounding/{provider}_{model}/page_0001.fields.json"],
                "succeeded_count": 1,
                "failed_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_stamp_images_anthropic_job_not_found(api_client: TestClient) -> None:
    r = api_client.post("/api/v1/jobs/11111111-1111-4111-8111-111111111111/stamp-images/anthropic")
    assert r.status_code == 404


def test_stamp_images_openai_missing_grounding_run(api_client: TestClient) -> None:
    job_id = _create_converted_job_only()
    r = api_client.post(f"/api/v1/jobs/{job_id}/stamp-images/openai", json={"values": {"first_name": "Jane"}})
    assert r.status_code == 400
    assert "Field grounding run not found" in r.json()["detail"]


def test_stamp_images_openai_happy_path(api_client: TestClient) -> None:
    job_id = _create_converted_job_only()
    settings = app.dependency_overrides[get_settings]()
    output_dir = settings.jobs_dir / job_id / "output"
    width, height = _converted_image_dimensions(output_dir)
    _write_field_grounding_file(output_dir, provider="openai", model="gpt-5.5", width=width, height=height)

    r = api_client.post(
        f"/api/v1/jobs/{job_id}/stamp-images/openai",
        json={"values": {"first_name": "Jane", "last_name": "Doe"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-5.5"
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 0
    assert body["pages"][0]["stamped_count"] == 2
    assert body["pages"][0]["missing_value_count"] == 0
    assert body["pages"][0]["output_image"].endswith(".openai.stamped.png")
    assert (output_dir / body["pages"][0]["output_image"]).is_file()
    assert (output_dir / body["manifest_path"]).is_file()


def test_stamp_images_anthropic_happy_path(api_client: TestClient) -> None:
    job_id = _create_converted_job_only()
    settings = app.dependency_overrides[get_settings]()
    output_dir = settings.jobs_dir / job_id / "output"
    width, height = _converted_image_dimensions(output_dir)
    _write_field_grounding_file(
        output_dir,
        provider="anthropic",
        model="claude-opus-4-7",
        width=width,
        height=height,
    )

    r = api_client.post(
        f"/api/v1/jobs/{job_id}/stamp-images/anthropic",
        json={
            "values": {"first_name": "Jane"},
            "style": {"draw_debug_boxes": True, "font_color": "#222222"},
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "anthropic"
    assert body["model"] == "claude-opus-4-7"
    assert body["run_dir"].startswith("stamped_images/anthropic_claude-opus-4-7/")
    assert body["pages"][0]["stamped_count"] == 1
    assert body["pages"][0]["missing_value_count"] == 1
    assert body["pages"][0]["output_image"].endswith(".anthropic.stamped.png")


def test_stamp_images_anthropic_uses_default_model(api_client: TestClient) -> None:
    job_id = _create_converted_job_only()
    settings = app.dependency_overrides[get_settings]()
    output_dir = settings.jobs_dir / job_id / "output"
    width, height = _converted_image_dimensions(output_dir)
    _write_field_grounding_file(
        output_dir,
        provider="anthropic",
        model="claude-3-5-sonnet",
        width=width,
        height=height,
    )

    r = api_client.post(
        f"/api/v1/jobs/{job_id}/stamp-images/anthropic",
        json={"values": {"first_name": "Jane"}},
    )
    assert r.status_code == 400
    assert "Field grounding run not found" in r.json()["detail"]


def test_stamp_images_openai_all_failed_returns_422(api_client: TestClient) -> None:
    job_id = _create_converted_job_only()
    settings = app.dependency_overrides[get_settings]()
    output_dir = settings.jobs_dir / job_id / "output"
    _write_field_grounding_file(output_dir, provider="openai", model="gpt-5.5", width=9999, height=9999)

    r = api_client.post(
        f"/api/v1/jobs/{job_id}/stamp-images/openai",
        json={"values": {"first_name": "Jane", "last_name": "Doe"}},
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["message"] == "Image stamping failed for all pages."
    assert detail["failed_count"] == 1


def test_stamp_pdf_anthropic_job_not_found(api_client: TestClient) -> None:
    r = api_client.post("/api/v1/jobs/11111111-1111-4111-8111-111111111111/stamp-pdf/anthropic")
    assert r.status_code == 404


def test_stamp_pdf_openai_missing_grounding_run(api_client: TestClient) -> None:
    job_id = _create_converted_job_only()
    r = api_client.post(f"/api/v1/jobs/{job_id}/stamp-pdf/openai", json={"values": {"first_name": "Jane"}})
    assert r.status_code == 400
    assert "Field grounding run not found" in r.json()["detail"]


def test_stamp_pdf_rejects_style_input(api_client: TestClient) -> None:
    job_id = _create_converted_job_only()
    r = api_client.post(
        f"/api/v1/jobs/{job_id}/stamp-pdf/openai",
        json={
            "values": {"first_name": "Jane"},
            "style": {"font_size_pt": 20},
        },
    )
    assert r.status_code == 422


def test_stamp_pdf_openai_happy_path(api_client: TestClient) -> None:
    job_id = _create_converted_job_only()
    settings = app.dependency_overrides[get_settings]()
    output_dir = settings.jobs_dir / job_id / "output"
    width, height = _converted_image_dimensions(output_dir)
    _write_field_grounding_file(output_dir, provider="openai", model="gpt-5.5", width=width, height=height)

    r = api_client.post(
        f"/api/v1/jobs/{job_id}/stamp-pdf/openai",
        json={"values": {"first_name": "Jane", "last_name": "Doe"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-5.5"
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 0
    assert body["output_pdf"].endswith(".openai.pdf")
    assert (output_dir / body["output_pdf"]).is_file()
    assert (output_dir / body["manifest_path"]).is_file()


def test_stamp_pdf_anthropic_happy_path(api_client: TestClient) -> None:
    job_id = _create_converted_job_only()
    settings = app.dependency_overrides[get_settings]()
    output_dir = settings.jobs_dir / job_id / "output"
    width, height = _converted_image_dimensions(output_dir)
    _write_field_grounding_file(output_dir, provider="anthropic", model="claude-opus-4-7", width=width, height=height)

    r = api_client.post(
        f"/api/v1/jobs/{job_id}/stamp-pdf/anthropic",
        json={"values": {"first_name": "Jane"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "anthropic"
    assert body["model"] == "claude-opus-4-7"
    assert body["run_dir"].startswith("stamped_pdfs/anthropic_claude-opus-4-7/")
    assert body["pages"][0]["stamped_count"] == 1
    assert body["pages"][0]["missing_value_count"] == 1


def test_stamp_pdf_openai_all_failed_returns_422(api_client: TestClient) -> None:
    job_id = _create_converted_job_only()
    settings = app.dependency_overrides[get_settings]()
    output_dir = settings.jobs_dir / job_id / "output"
    _write_field_grounding_file(output_dir, provider="openai", model="gpt-5.5", width=9999, height=9999)

    r = api_client.post(
        f"/api/v1/jobs/{job_id}/stamp-pdf/openai",
        json={"values": {"first_name": "Jane", "last_name": "Doe"}},
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["message"] == "PDF stamping failed for all pages."
    assert detail["failed_count"] == 1
