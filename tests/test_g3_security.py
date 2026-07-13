"""G3 security tests — path containment and upload validation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_settings
from app.main import create_app

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "golden-job"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    settings = Settings(jobs_dir=jobs_dir)

    def _settings_override() -> Settings:
        return settings

    app = create_app()
    app.dependency_overrides[get_settings] = _settings_override
    return TestClient(app), jobs_dir


def test_job_json_artifact_path_traversal_blocked(client: tuple[TestClient, Path]) -> None:
    test_client, jobs_dir = client
    job_id = "550e8400-e29b-41d4-a716-446655440000"
    root = jobs_dir / job_id
    root.mkdir()
    output = root / "output"
    shutil.copytree(FIXTURE_ROOT / "output", output)

    manifest = json.loads((FIXTURE_ROOT / "job.json").read_text(encoding="utf-8"))
    manifest["job_id"] = job_id
    manifest["artifacts"]["latest_stamped_pdf"] = "../../../etc/passwd"
    (root / "job.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    resp = test_client.get(f"/api/v1/jobs/{job_id}/export")
    assert resp.status_code in (400, 404)
    body = resp.json()
    assert "error" in body
    assert "/etc/passwd" not in body.get("message", "")
    assert str(jobs_dir) not in body.get("message", "")


def test_invalid_uuid_rejected(client: tuple[TestClient, Path]) -> None:
    test_client, _ = client
    resp = test_client.get("/api/v1/jobs/../../../../etc/passwd/export")
    assert resp.status_code in (400, 404, 405, 422)
