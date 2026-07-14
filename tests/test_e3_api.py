"""E3 API integration and security tests."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_settings
from app.main import create_app
from app.services.jobs import update_job_after_grounding

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "golden-job"


@pytest.fixture
def jobs_dir(tmp_path: Path) -> Path:
    path = tmp_path / "jobs"
    path.mkdir()
    return path


@pytest.fixture
def client(jobs_dir: Path) -> TestClient:
    settings = Settings(jobs_dir=jobs_dir, openai_api_key="test-key")

    def _settings_override() -> Settings:
        return settings

    app = create_app()
    app.dependency_overrides[get_settings] = _settings_override
    return TestClient(app)


def _minimal_pdf_bytes() -> bytes:
    doc = fitz.open()
    try:
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), "Sample form content for tests.")
        buf = doc.tobytes()
    finally:
        doc.close()
    return buf


def _large_pdf_bytes() -> bytes:
    doc = fitz.open()
    try:
        for _ in range(8):
            page = doc.new_page(width=612, height=792)
            page.insert_text((72, 72), "x" * 400)
        return doc.tobytes()
    finally:
        doc.close()


def _mock_grounding(job_id: str, output_dir: Path, **_: object) -> dict:
    fg = output_dir / "field_grounding"
    fg.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        FIXTURE_ROOT / "output" / "field_grounding" / "page_0001.fields.json",
        fg / "page_0001.fields.json",
    )
    shutil.copy2(
        FIXTURE_ROOT / "output" / "field_grounding" / "stamping.json",
        fg / "stamping.json",
    )
    update_job_after_grounding(
        output_dir.parent,
        provider="openai",
        model="gpt-4.1",
        grounded_pages=1,
        total_pages=1,
        failed_pages=[],
    )
    return {"job_id": job_id, "page_count": 1, "succeeded_count": 1, "failed_count": 0}


def _wait_for_status(client: TestClient, job_id: str, wanted: str, timeout_s: float = 30.0) -> dict:
    deadline = time.time() + timeout_s
    last: dict = {}
    while time.time() < deadline:
        resp = client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        last = resp.json()
        if last.get("status") == wanted:
            return last
        if last.get("status") == "failed":
            pytest.fail(f"Job failed: {last}")
        time.sleep(0.05)
    pytest.fail(f"Timed out waiting for status={wanted!r}, last={last}")


@patch("app.services.job_pipeline.run_semantic_grounding_for_job", side_effect=_mock_grounding)
def test_upload_poll_fields_patch_stamp_export(mock_grounding, client: TestClient) -> None:
    pdf = _minimal_pdf_bytes()
    upload = client.post(
        "/api/v1/jobs",
        files={"file": ("demo.pdf", pdf, "application/pdf")},
    )
    assert upload.status_code == 201
    body = upload.json()
    job_id = body["job_id"]
    assert body["status"] == "converting"

    detail = _wait_for_status(client, job_id, "ready")
    assert detail["page_count"] >= 1

    fields = client.get(f"/api/v1/jobs/{job_id}/fields")
    assert fields.status_code == 200
    fields_body = fields.json()
    assert fields_body["values"]["field_001"] == ""
    assert fields_body["pages"][0]["fields"][0]["field_id"] == "field_001"

    patch_fields = client.patch(
        f"/api/v1/jobs/{job_id}/fields",
        json={
            "fields": [
                {
                    "field_id": "field_001",
                    "page_number": 1,
                    "bbox": {"x": 110, "y": 210, "w": 300, "h": 24},
                    "reviewed": True,
                }
            ]
        },
    )
    assert patch_fields.status_code == 200

    patch_values = client.patch(
        f"/api/v1/jobs/{job_id}/values",
        json={"values": {"field_001": "Jane Doe"}},
    )
    assert patch_values.status_code == 200
    assert patch_values.json()["values"]["field_001"] == "Jane Doe"

    # E2 lands the global style PATCH (was a 501 stub pre-E2); font_size_pt round-trips.
    patch_style = client.patch(
        f"/api/v1/jobs/{job_id}/values",
        json={"style": {"font_size_pt": 14}},
    )
    assert patch_style.status_code == 200
    assert patch_style.json()["style"]["font_size_pt"] == 14

    fields_after_style = client.get(f"/api/v1/jobs/{job_id}/fields")
    assert fields_after_style.json()["style"]["font_size_pt"] == 14

    page_image = client.get(f"/api/v1/jobs/{job_id}/pages/1/image")
    assert page_image.status_code == 200
    assert page_image.headers["content-type"].startswith("image/png")

    stamp_images = client.post(f"/api/v1/jobs/{job_id}/stamp-images")
    assert stamp_images.status_code == 200
    stamp_body = stamp_images.json()
    assert stamp_body["pages"][0]["image_url"].endswith("variant=stamped")

    stamp_pdf = client.post(f"/api/v1/jobs/{job_id}/stamp-pdf")
    assert stamp_pdf.status_code == 200
    assert stamp_pdf.json()["download_url"] == f"/api/v1/jobs/{job_id}/export"

    export_resp = client.get(f"/api/v1/jobs/{job_id}/export")
    assert export_resp.status_code == 200
    assert export_resp.headers["content-type"] == "application/pdf"
    assert export_resp.content.startswith(b"%PDF")


def test_upload_rejects_non_pdf(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/jobs",
        files={"file": ("bad.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "invalid_pdf"


def test_upload_rejects_oversize(client: TestClient, jobs_dir: Path) -> None:
    settings = Settings(jobs_dir=jobs_dir, max_upload_bytes=1024)
    app = create_app()

    def _settings_override() -> Settings:
        return settings

    app.dependency_overrides[get_settings] = _settings_override
    small_client = TestClient(app)

    pdf = _large_pdf_bytes()
    assert len(pdf) > 1024
    resp = small_client.post(
        "/api/v1/jobs",
        files={"file": ("big.pdf", pdf, "application/pdf")},
    )
    assert resp.status_code == 413
    assert resp.json()["error"] == "file_too_large"


def test_path_traversal_job_id_rejected(client: TestClient) -> None:
    resp = client.get("/api/v1/jobs/not-a-uuid")
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_job_id"


def test_path_traversal_page_image(client: TestClient, jobs_dir: Path) -> None:
    job_id = "550e8400-e29b-41d4-a716-446655440000"
    root = jobs_dir / job_id
    root.mkdir()
    shutil.copy2(FIXTURE_ROOT / "job.json", root / "job.json")
    output = root / "output"
    shutil.copytree(FIXTURE_ROOT / "output", output)

    from scripts.convert_pdf_pages_for_grounding import convert_pdf_to_images
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "p.pdf"
        pdf_path.write_bytes(_minimal_pdf_bytes())
        convert_pdf_to_images(str(pdf_path), str(output), dpi=72.0, overwrite=True, job_id=job_id)

    missing = client.get(f"/api/v1/jobs/{job_id}/pages/99/image")
    assert missing.status_code == 404


def test_list_and_delete_job(client: TestClient, jobs_dir: Path) -> None:
    job_id = "550e8400-e29b-41d4-a716-446655440000"
    root = jobs_dir / job_id
    root.mkdir()
    manifest = json.loads((FIXTURE_ROOT / "job.json").read_text(encoding="utf-8"))
    manifest["job_id"] = job_id
    (root / "job.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    listed = client.get("/api/v1/jobs")
    assert listed.status_code == 200
    assert any(item["job_id"] == job_id for item in listed.json()["jobs"])

    deleted = client.delete(f"/api/v1/jobs/{job_id}")
    assert deleted.status_code == 204
    assert not root.exists()
