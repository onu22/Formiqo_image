"""E5 vision QA refinement loop: judge schema, crops, bounded deltas, convergence, API."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.services.grounding_qa as gqa
from app.config import Settings
from app.dependencies import get_settings
from app.main import create_app
from app.services.grounding_qa import (
    JudgeResult,
    JudgeVerdict,
    apply_verdicts_to_fields,
    build_judge_callable,
    crop_field_region,
    resolve_judge_config,
    run_grounding_qa_refinement,
)
from app.services.grounding_schema import (
    QA_JUDGE_TOOL_NAME,
    anthropic_qa_tool,
    openai_qa_response_format,
    qa_judge_response_schema,
)
from app.services.jobs import job_detail_projection, new_job_manifest, read_job_manifest, write_job_manifest

_JOB_ID = "22222222-2222-4222-8222-222222222222"


# --------------------------------------------------------------------------- schema


def test_openai_qa_schema_is_strict_and_closed() -> None:
    fmt = openai_qa_response_format()
    assert fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    item = schema["properties"]["fields"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"field_id", "verdict", "dx", "dy", "confidence"}
    assert item["properties"]["verdict"]["enum"] == ["ok", "shift", "unsure"]


def test_anthropic_qa_tool_is_relaxed() -> None:
    tool = anthropic_qa_tool()
    assert tool["name"] == QA_JUDGE_TOOL_NAME
    item = tool["input_schema"]["properties"]["fields"]["items"]
    assert item["required"] == ["field_id", "verdict"]


def test_qa_schema_shares_verdict_vocabulary() -> None:
    strict = qa_judge_response_schema(strict=True)
    relaxed = qa_judge_response_schema(strict=False)
    s = strict["properties"]["fields"]["items"]["properties"]["verdict"]["enum"]
    r = relaxed["properties"]["fields"]["items"]["properties"]["verdict"]["enum"]
    assert s == r == ["ok", "shift", "unsure"]


# --------------------------------------------------------------------------- judge config


def test_resolve_judge_config_picks_opposite_provider() -> None:
    settings = Settings(grounding_qa_judge_provider="", grounding_qa_judge_model="")
    assert resolve_judge_config(settings, grounder_provider="openai") == ("anthropic", "claude-opus-4-7")
    assert resolve_judge_config(settings, grounder_provider="anthropic") == ("openai", "gpt-5")


def test_resolve_judge_config_honors_explicit_override() -> None:
    settings = Settings(grounding_qa_judge_provider="openai", grounding_qa_judge_model="gpt-judge")
    assert resolve_judge_config(settings, grounder_provider="openai") == ("openai", "gpt-judge")


def test_build_judge_missing_key_raises() -> None:
    settings = Settings(grounding_qa_judge_provider="anthropic", anthropic_api_key="")
    with pytest.raises(ValueError):
        build_judge_callable(settings, grounder_provider="openai")


# --------------------------------------------------------------------------- crops


def test_crop_field_region_zooms_with_context() -> None:
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    png = crop_field_region(
        img,
        {"x": 100, "y": 100, "w": 50, "h": 20},
        zoom=3.0,
        context_px=10,
        page_w=400,
        page_h=300,
    )
    with Image.open(io.BytesIO(png)) as crop:
        # region (90,90)-(160,130) = 70x40, zoomed 3x
        assert crop.size == (210, 120)


def test_crop_field_region_clamps_to_page_edges() -> None:
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    png = crop_field_region(
        img,
        {"x": 0, "y": 0, "w": 10, "h": 10},
        zoom=1.0,
        context_px=50,
        page_w=100,
        page_h=100,
    )
    with Image.open(io.BytesIO(png)) as crop:
        assert crop.size == (60, 60)


# --------------------------------------------------------------------------- bounded deltas


def _text_field(field_id: str, bbox: dict[str, int]) -> dict[str, Any]:
    return {"field_id": field_id, "type": "text", "bbox": dict(bbox)}


def test_apply_verdicts_clamps_to_bound() -> None:
    fields = [_text_field("a", {"x": 100, "y": 100, "w": 50, "h": 20})]
    verdicts = [JudgeVerdict("a", "shift", dx=500, dy=-500, confidence=0.9)]
    applied = apply_verdicts_to_fields(
        fields,
        verdicts,
        max_delta=30,
        page_w=1000,
        page_h=1000,
        consensus_enabled=False,
        consensus_min_fields=3,
        consensus_max_spread=4,
    )
    assert applied["a"] == (30, -30)
    assert fields[0]["bbox"]["x"] == 130
    assert fields[0]["bbox"]["y"] == 70


def test_apply_verdicts_ok_verdict_does_not_move() -> None:
    fields = [_text_field("a", {"x": 100, "y": 100, "w": 50, "h": 20})]
    verdicts = [JudgeVerdict("a", "ok", dx=0, dy=0, confidence=0.95)]
    applied = apply_verdicts_to_fields(
        fields,
        verdicts,
        max_delta=30,
        page_w=1000,
        page_h=1000,
        consensus_enabled=False,
        consensus_min_fields=3,
        consensus_max_spread=4,
    )
    assert applied == {}
    assert fields[0]["bbox"]["x"] == 100


def test_consensus_translation_merges_agreeing_shifts() -> None:
    fields = [
        _text_field("a", {"x": 100, "y": 100, "w": 40, "h": 20}),
        _text_field("b", {"x": 100, "y": 200, "w": 40, "h": 20}),
        _text_field("c", {"x": 100, "y": 300, "w": 40, "h": 20}),
    ]
    verdicts = [
        JudgeVerdict("a", "shift", dx=10, dy=0, confidence=0.9),
        JudgeVerdict("b", "shift", dx=11, dy=0, confidence=0.9),
        JudgeVerdict("c", "shift", dx=9, dy=0, confidence=0.9),
    ]
    applied = apply_verdicts_to_fields(
        fields,
        verdicts,
        max_delta=30,
        page_w=1000,
        page_h=1000,
        consensus_enabled=True,
        consensus_min_fields=3,
        consensus_max_spread=4,
    )
    # All three merged to the mean (10, 0) page translation.
    assert applied["a"] == (10, 0)
    assert applied["b"] == (10, 0)
    assert applied["c"] == (10, 0)


def test_consensus_translation_skipped_when_shifts_disagree() -> None:
    fields = [
        _text_field("a", {"x": 100, "y": 100, "w": 40, "h": 20}),
        _text_field("b", {"x": 100, "y": 200, "w": 40, "h": 20}),
        _text_field("c", {"x": 100, "y": 300, "w": 40, "h": 20}),
    ]
    verdicts = [
        JudgeVerdict("a", "shift", dx=10, dy=0, confidence=0.9),
        JudgeVerdict("b", "shift", dx=10, dy=0, confidence=0.9),
        JudgeVerdict("c", "shift", dx=30, dy=0, confidence=0.9),
    ]
    applied = apply_verdicts_to_fields(
        fields,
        verdicts,
        max_delta=30,
        page_w=1000,
        page_h=1000,
        consensus_enabled=True,
        consensus_min_fields=3,
        consensus_max_spread=4,
    )
    assert applied["a"] == (10, 0)
    assert applied["c"] == (30, 0)


# --------------------------------------------------------------------------- job fixture


def _build_job(
    tmp_path: Path,
    *,
    field_bbox: dict[str, int],
    values: dict[str, str] | None = None,
    page_w: int = 400,
    page_h: int = 300,
) -> tuple[str, Path]:
    root = tmp_path / _JOB_ID
    output = root / "output"
    conv = output / "converted_images"
    (conv / "pages").mkdir(parents=True)
    Image.new("RGB", (page_w, page_h), (255, 255, 255)).save(conv / "page_0001.png")
    (conv / "pages" / "page_0001.json").write_text(
        json.dumps(
            {
                "page_index": 0,
                "pdf": {"width_pt": page_w * 0.36, "height_pt": page_h * 0.36, "origin": "bottom-left"},
                "image": {
                    "path": "converted_images/page_0001.png",
                    "width_px": page_w,
                    "height_px": page_h,
                    "origin": "top-left",
                },
                "mapping": {"image_to_pdf_scale_x": 0.36, "image_to_pdf_scale_y": 0.36},
            }
        ),
        encoding="utf-8",
    )
    fg = output / "field_grounding"
    fg.mkdir(parents=True)
    (fg / "page_0001.fields.json").write_text(
        json.dumps(
            {
                "page_index": 0,
                "width_px": page_w,
                "height_px": page_h,
                "unit": "px",
                "origin": "top-left",
                "fields": [
                    {
                        "field_id": "name",
                        "type": "text",
                        "bbox": dict(field_bbox),
                        "confidence": 0.9,
                        "label": "Name",
                        "evidence": {"line_ids": []},
                        "grounding_source": "pixel",
                        "qa_status": None,
                        "reviewed": False,
                        "font_size_pt": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (fg / "stamping.json").write_text(
        json.dumps(
            {
                "values": values if values is not None else {"name": "John"},
                "require_all_values": False,
                "style": {"font_size_pt": 11, "text_color": "#111111"},
                "overrides": {},
            }
        ),
        encoding="utf-8",
    )
    manifest = new_job_manifest(job_id=_JOB_ID, source_filename="form.pdf")
    manifest["page_count"] = 1
    manifest["status"] = "ready"
    manifest["grounding"] = {"provider": "openai", "model": "gpt-test", "run_id": "r"}
    write_job_manifest(root, manifest)
    return _JOB_ID, output


def _make_targeting_judge(targets: dict[str, tuple[int, int]], *, tol: int = 1):
    """Fake judge that steers each field toward a known-truth (x, y) top-left corner."""

    def _judge(page_index: int, crops: list[gqa.FieldCrop]) -> JudgeResult:
        verdicts: list[JudgeVerdict] = []
        for c in crops:
            tx, ty = targets[c.field_id]
            dx = tx - c.bbox["x"]
            dy = ty - c.bbox["y"]
            if abs(dx) <= tol and abs(dy) <= tol:
                verdicts.append(JudgeVerdict(c.field_id, "ok", 0, 0, 0.96))
            else:
                verdicts.append(JudgeVerdict(c.field_id, "shift", dx, dy, 0.9))
        return JudgeResult(verdicts, {"input_tokens": 120, "output_tokens": 24})

    return _judge


# --------------------------------------------------------------------------- convergence


def test_perturbed_bbox_is_measurably_corrected(tmp_path: Path) -> None:
    truth = (150, 110)
    perturbed = {"x": 230, "y": 190, "w": 100, "h": 30}
    job_id, output = _build_job(tmp_path, field_bbox=perturbed)

    before_err = math.hypot(perturbed["x"] - truth[0], perturbed["y"] - truth[1])

    settings = Settings(
        jobs_dir=tmp_path,
        grounding_qa_max_iterations=6,
        grounding_qa_max_bbox_delta_px=30,
        grounding_qa_consensus_translation_enabled=False,
    )
    summary = run_grounding_qa_refinement(
        job_id=job_id,
        output_dir=output,
        settings=settings,
        judge=_make_targeting_judge({"name": truth}),
    )

    fields = json.loads((output / "field_grounding" / "page_0001.fields.json").read_text())["fields"]
    final = fields[0]["bbox"]
    after_err = math.hypot(final["x"] - truth[0], final["y"] - truth[1])

    assert after_err < before_err
    assert (final["x"], final["y"]) == truth
    assert summary["converged"] is True
    assert 3 <= summary["iterations"] <= 6
    assert fields[0]["qa_status"] == "adjusted"

    # No single iteration ever exceeded the per-axis bound.
    for m in summary["iteration_metrics"]:
        assert m["total_shift_px"] <= m["fields_shifted"] * 2 * settings.grounding_qa_max_bbox_delta_px


def test_confirmed_field_not_moved(tmp_path: Path) -> None:
    bbox = {"x": 150, "y": 110, "w": 100, "h": 30}
    job_id, output = _build_job(tmp_path, field_bbox=bbox)

    def _all_ok(page_index: int, crops: list[gqa.FieldCrop]) -> JudgeResult:
        return JudgeResult([JudgeVerdict(c.field_id, "ok", 0, 0, 0.97) for c in crops], {})

    settings = Settings(jobs_dir=tmp_path)
    summary = run_grounding_qa_refinement(
        job_id=job_id, output_dir=output, settings=settings, judge=_all_ok
    )
    assert summary["converged"] is True
    assert summary["iterations"] == 1
    assert summary["fields_confirmed"] == 1
    fields = json.loads((output / "field_grounding" / "page_0001.fields.json").read_text())["fields"]
    assert fields[0]["bbox"] == bbox
    assert fields[0]["qa_status"] == "confirmed"


def test_unsure_verdict_flags_field(tmp_path: Path) -> None:
    job_id, output = _build_job(tmp_path, field_bbox={"x": 150, "y": 110, "w": 100, "h": 30})

    def _unsure(page_index: int, crops: list[gqa.FieldCrop]) -> JudgeResult:
        return JudgeResult([JudgeVerdict(c.field_id, "unsure", 0, 0, 0.5) for c in crops], {})

    settings = Settings(jobs_dir=tmp_path)
    summary = run_grounding_qa_refinement(
        job_id=job_id, output_dir=output, settings=settings, judge=_unsure
    )
    assert summary["fields_flagged"] == 1
    fields = json.loads((output / "field_grounding" / "page_0001.fields.json").read_text())["fields"]
    assert fields[0]["qa_status"] == "flagged"


def test_low_confidence_ok_flags_field(tmp_path: Path) -> None:
    job_id, output = _build_job(tmp_path, field_bbox={"x": 150, "y": 110, "w": 100, "h": 30})

    def _low_conf(page_index: int, crops: list[gqa.FieldCrop]) -> JudgeResult:
        return JudgeResult([JudgeVerdict(c.field_id, "ok", 0, 0, 0.1) for c in crops], {})

    settings = Settings(jobs_dir=tmp_path, grounding_qa_flag_low_confidence=0.35)
    summary = run_grounding_qa_refinement(
        job_id=job_id, output_dir=output, settings=settings, judge=_low_conf
    )
    assert summary["fields_flagged"] == 1


def test_job_manifest_qa_refine_stage_and_projection(tmp_path: Path) -> None:
    job_id, output = _build_job(tmp_path, field_bbox={"x": 150, "y": 110, "w": 100, "h": 30})

    def _all_ok(page_index: int, crops: list[gqa.FieldCrop]) -> JudgeResult:
        return JudgeResult(
            [JudgeVerdict(c.field_id, "ok", 0, 0, 0.95) for c in crops],
            {"input_tokens": 200, "output_tokens": 30},
        )

    settings = Settings(jobs_dir=tmp_path)
    run_grounding_qa_refinement(job_id=job_id, output_dir=output, settings=settings, judge=_all_ok)

    manifest = read_job_manifest(output.parent)
    qa = manifest["stages"]["qa_refine"]
    assert qa["status"] == "done"
    assert qa["converged"] is True
    assert qa["cost"]["judge_calls"] == 1
    assert qa["cost"]["input_tokens"] == 200

    projection = job_detail_projection(manifest)
    assert projection["stages"]["qa_refine"]["fields_confirmed"] == 1
    assert projection["stages"]["qa_refine"]["converged"] is True


# --------------------------------------------------------------------------- endpoint


@pytest.fixture
def api_client(tmp_path: Path) -> TestClient:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    settings = Settings(jobs_dir=jobs_dir, openai_api_key="test-key")

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_refine_grounding_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    _build_job(jobs_dir, field_bbox={"x": 200, "y": 160, "w": 100, "h": 30})

    monkeypatch.setattr(gqa, "build_judge_callable", lambda *a, **k: _make_targeting_judge({"name": (150, 110)}))

    settings = Settings(jobs_dir=jobs_dir, openai_api_key="test-key", grounding_qa_max_bbox_delta_px=30)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)

    resp = client.post(f"/api/v1/jobs/{_JOB_ID}/refine-grounding")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["converged"] is True
    assert body["fields_adjusted"] == 1
    assert body["judge_provider"] in ("openai", "anthropic")


def test_refine_grounding_missing_judge_key_returns_400(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    _build_job(jobs_dir, field_bbox={"x": 150, "y": 110, "w": 100, "h": 30})

    # Grounder is openai → judge defaults to anthropic; no anthropic key configured.
    settings = Settings(jobs_dir=jobs_dir, openai_api_key="test-key", anthropic_api_key="")
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)

    resp = client.post(f"/api/v1/jobs/{_JOB_ID}/refine-grounding")
    assert resp.status_code == 400
    assert resp.json()["error"] == "refine_failed"
