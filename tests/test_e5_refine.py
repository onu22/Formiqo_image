"""E5 vision QA refinement loop: crops, bounded deltas, consensus, convergence, persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_settings
from app.main import create_app
from app.services import qa_refinement as qa
from app.services.qa_refinement import (
    JudgeRequest,
    JudgeVerdict,
    bbox_center_error,
    clamp_delta,
    field_has_visible_value,
    merge_consensus_translation,
    resolve_judge_provider_model,
    run_qa_refinement_for_job,
    shift_bbox,
)
from app.services.qa_schema import (
    QA_JUDGE_TOOL_NAME,
    anthropic_qa_tool,
    openai_qa_response_format,
    qa_verdict_schema,
)
from tests.fixtures.parity.builder import build_parity_job

_JOB_ID = "22222222-2222-4222-8222-222222222222"


# --------------------------------------------------------------------------- schema


def test_openai_qa_schema_is_strict_and_closed() -> None:
    fmt = openai_qa_response_format()
    assert fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"].keys())
    assert schema["properties"]["verdict"]["enum"] == ["ok", "adjust"]


def test_anthropic_qa_tool_relaxed() -> None:
    tool = anthropic_qa_tool()
    assert tool["name"] == QA_JUDGE_TOOL_NAME
    schema = tool["input_schema"]
    assert schema["required"] == ["field_id", "verdict"]
    assert schema["properties"]["verdict"]["enum"] == ["ok", "adjust"]


def test_qa_schema_shares_verdict_vocabulary() -> None:
    strict = qa_verdict_schema(strict=True)
    relaxed = qa_verdict_schema(strict=False)
    assert (
        strict["properties"]["verdict"]["enum"]
        == relaxed["properties"]["verdict"]["enum"]
        == ["ok", "adjust"]
    )


# --------------------------------------------------------------------------- pure helpers


def test_clamp_delta_bounds() -> None:
    assert clamp_delta(500, max_delta=30) == 30
    assert clamp_delta(-500, max_delta=30) == -30
    assert clamp_delta(12, max_delta=30) == 12


def test_shift_bbox_clamps_to_page() -> None:
    bbox = {"x": 10, "y": 10, "w": 40, "h": 20}
    moved = shift_bbox(bbox, 5, -3, page_w=100, page_h=100)
    assert moved == {"x": 15, "y": 7, "w": 40, "h": 20}
    # Cannot leave the page.
    clamped = shift_bbox(bbox, 1000, 1000, page_w=100, page_h=100)
    assert clamped["x"] == 60 and clamped["y"] == 80


def test_merge_consensus_translation_agrees() -> None:
    deltas = {"a": (10, 4), "b": (11, 5), "c": (12, 4)}
    result = merge_consensus_translation(deltas, min_fields=3, max_spread_px=4)
    assert result == (11, 4)


def test_merge_consensus_translation_rejects_disagreement() -> None:
    deltas = {"a": (10, 4), "b": (-20, 40), "c": (12, 4)}
    assert merge_consensus_translation(deltas, min_fields=3, max_spread_px=4) is None


def test_merge_consensus_translation_needs_min_fields() -> None:
    deltas = {"a": (10, 4), "b": (11, 5)}
    assert merge_consensus_translation(deltas, min_fields=3, max_spread_px=4) is None


def test_bbox_center_error() -> None:
    a = {"x": 0, "y": 0, "w": 10, "h": 10}
    b = {"x": 3, "y": 4, "w": 10, "h": 10}
    assert bbox_center_error(a, b) == pytest.approx(5.0)


def test_resolve_judge_provider_defaults_to_opposite() -> None:
    settings = Settings(grounding_qa_provider="", grounding_qa_model="")
    prov, model = resolve_judge_provider_model(settings, grounder_provider="openai")
    assert prov == "anthropic"
    assert model
    prov2, _ = resolve_judge_provider_model(settings, grounder_provider="anthropic")
    assert prov2 == "openai"


def test_resolve_judge_provider_explicit_override() -> None:
    settings = Settings(grounding_qa_provider="openai", grounding_qa_model="judge-x")
    prov, model = resolve_judge_provider_model(settings, grounder_provider="openai")
    assert (prov, model) == ("openai", "judge-x")


def test_field_has_visible_value() -> None:
    values = {"t": "hello", "empty": "", "cb": "true", "cb0": "false"}
    assert field_has_visible_value({"field_id": "t", "type": "text"}, values)
    assert not field_has_visible_value({"field_id": "empty", "type": "text"}, values)
    assert field_has_visible_value({"field_id": "cb", "type": "checkbox"}, values)
    assert not field_has_visible_value({"field_id": "cb0", "type": "checkbox"}, values)


# --------------------------------------------------------------------------- loop


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = dict(
        jobs_dir=tmp_path,
        grounding_qa_crop_zoom=1.0,
        grounding_qa_crop_padding_px=10,
        grounding_qa_max_bbox_delta_px=30,
        grounding_qa_clean_confidence=0.6,
        grounding_qa_consensus_translation_enabled=False,
    )
    base.update(overrides)
    return Settings(**base)


def _read_field(output_dir: Path, field_id: str) -> dict[str, Any]:
    data = json.loads((output_dir / "field_grounding" / "page_0001.fields.json").read_text(encoding="utf-8"))
    for field in data["fields"]:
        if field["field_id"] == field_id:
            return field
    raise KeyError(field_id)


def _perturb(output_dir: Path, field_id: str, dx: int, dy: int) -> dict[str, int]:
    """Shift a field's bbox on disk; return the pre-perturbation ("truth") bbox."""
    path = output_dir / "field_grounding" / "page_0001.fields.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    truth: dict[str, int] = {}
    for field in data["fields"]:
        if field["field_id"] == field_id:
            truth = dict(field["bbox"])
            field["bbox"] = {**field["bbox"], "x": field["bbox"]["x"] + dx, "y": field["bbox"]["y"] + dy}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return truth


def _targeting_judge(truth_by_field: dict[str, dict[str, int]], *, tol: int = 1) -> qa.JudgeFn:
    """Judge that steers each field's bbox back toward a known truth bbox."""

    def _judge(req: JudgeRequest) -> JudgeVerdict:
        truth = truth_by_field.get(req.field_id)
        usage = {"input_tokens": 100, "output_tokens": 8}
        if truth is None:
            return JudgeVerdict(verdict="ok", confidence=0.95, usage=usage)
        dx = int(truth["x"] - req.bbox["x"])
        dy = int(truth["y"] - req.bbox["y"])
        if abs(dx) <= tol and abs(dy) <= tol:
            return JudgeVerdict(verdict="ok", confidence=0.95, usage=usage)
        return JudgeVerdict(verdict="adjust", dx=dx, dy=dy, confidence=0.9, usage=usage)

    return _judge


def test_perturbed_bbox_converges_and_marks_adjusted(tmp_path: Path) -> None:
    form = build_parity_job(tmp_path, variant="form_a", job_id=_JOB_ID)
    truth = _perturb(form.output_dir, form.field_ids["text"], dx=25, dy=20)

    before = _read_field(form.output_dir, form.field_ids["text"])["bbox"]
    before_err = bbox_center_error(before, truth)

    summary = run_qa_refinement_for_job(
        job_id=form.job_id,
        output_dir=form.output_dir,
        settings=_settings(tmp_path, grounding_qa_max_iterations=6),
        judge_fn=_targeting_judge({form.field_ids["text"]: truth}),
    )

    after = _read_field(form.output_dir, form.field_ids["text"])
    after_err = bbox_center_error(after["bbox"], truth)

    assert summary["converged"] is True
    assert after_err <= before_err
    assert after_err <= 1.5
    assert after["qa_status"] == "adjusted"
    assert summary["adjusted"] >= 1


def test_never_moves_more_than_bound_per_iteration(tmp_path: Path) -> None:
    form = build_parity_job(tmp_path, variant="form_a", job_id=_JOB_ID)
    before = _read_field(form.output_dir, form.field_ids["text"])["bbox"]

    def _greedy_judge(req: JudgeRequest) -> JudgeVerdict:
        return JudgeVerdict(verdict="adjust", dx=500, dy=500, confidence=0.9)

    run_qa_refinement_for_job(
        job_id=form.job_id,
        output_dir=form.output_dir,
        settings=_settings(tmp_path, grounding_qa_max_iterations=1),
        judge_fn=_greedy_judge,
    )

    after = _read_field(form.output_dir, form.field_ids["text"])["bbox"]
    assert after["x"] - before["x"] == 30
    assert after["y"] - before["y"] == 30


def test_unconverged_field_is_flagged(tmp_path: Path) -> None:
    form = build_parity_job(tmp_path, variant="form_a", job_id=_JOB_ID)

    def _stubborn_judge(req: JudgeRequest) -> JudgeVerdict:
        return JudgeVerdict(verdict="adjust", dx=40, dy=0, confidence=0.9)

    summary = run_qa_refinement_for_job(
        job_id=form.job_id,
        output_dir=form.output_dir,
        settings=_settings(tmp_path, grounding_qa_max_iterations=2),
        judge_fn=_stubborn_judge,
    )

    assert summary["converged"] is False
    assert summary["iterations"] == 2
    assert _read_field(form.output_dir, form.field_ids["text"])["qa_status"] == "flagged"
    assert summary["flagged"] >= 1


def test_low_confidence_ok_is_flagged(tmp_path: Path) -> None:
    form = build_parity_job(tmp_path, variant="form_a", job_id=_JOB_ID)

    def _unsure_judge(req: JudgeRequest) -> JudgeVerdict:
        return JudgeVerdict(verdict="ok", confidence=0.2)

    summary = run_qa_refinement_for_job(
        job_id=form.job_id,
        output_dir=form.output_dir,
        settings=_settings(tmp_path, grounding_qa_max_iterations=3),
        judge_fn=_unsure_judge,
    )
    assert summary["converged"] is True
    assert summary["flagged"] >= 1
    assert _read_field(form.output_dir, form.field_ids["text"])["qa_status"] == "flagged"


def test_well_placed_field_confirmed(tmp_path: Path) -> None:
    form = build_parity_job(tmp_path, variant="form_a", job_id=_JOB_ID)

    def _ok_judge(req: JudgeRequest) -> JudgeVerdict:
        return JudgeVerdict(verdict="ok", confidence=0.95)

    summary = run_qa_refinement_for_job(
        job_id=form.job_id,
        output_dir=form.output_dir,
        settings=_settings(tmp_path, grounding_qa_max_iterations=6),
        judge_fn=_ok_judge,
    )
    assert summary["converged"] is True
    assert summary["iterations"] == 1
    assert _read_field(form.output_dir, form.field_ids["text"])["qa_status"] == "confirmed"


def test_consensus_translation_shifts_whole_page(tmp_path: Path) -> None:
    form = build_parity_job(tmp_path, variant="form_a", job_id=_JOB_ID)
    text_before = _read_field(form.output_dir, form.field_ids["text"])["bbox"]
    ml_before = _read_field(form.output_dir, form.field_ids["multiline"])["bbox"]

    def _uniform_judge(req: JudgeRequest) -> JudgeVerdict:
        # Every visible field agrees on the same small shift (a scan offset).
        return JudgeVerdict(verdict="adjust", dx=8, dy=6, confidence=0.9)

    run_qa_refinement_for_job(
        job_id=form.job_id,
        output_dir=form.output_dir,
        settings=_settings(
            tmp_path,
            grounding_qa_max_iterations=1,
            grounding_qa_consensus_translation_enabled=True,
            grounding_qa_consensus_min_fields=2,
            grounding_qa_consensus_max_spread_px=4,
        ),
        judge_fn=_uniform_judge,
    )

    text_after = _read_field(form.output_dir, form.field_ids["text"])["bbox"]
    ml_after = _read_field(form.output_dir, form.field_ids["multiline"])["bbox"]
    checkbox_after = _read_field(form.output_dir, form.field_ids["checkbox"])["bbox"]
    checkbox_before = {"x": 200, "y": 560, "w": 30, "h": 30}
    # Consensus applies one translation to every field on the page (even the checkbox).
    assert (text_after["x"] - text_before["x"], text_after["y"] - text_before["y"]) == (8, 6)
    assert (ml_after["x"] - ml_before["x"], ml_after["y"] - ml_before["y"]) == (8, 6)
    assert (checkbox_after["x"] - checkbox_before["x"], checkbox_after["y"] - checkbox_before["y"]) == (8, 6)


def test_refine_persists_stage_and_cost(tmp_path: Path) -> None:
    form = build_parity_job(tmp_path, variant="form_a", job_id=_JOB_ID)
    truth = _perturb(form.output_dir, form.field_ids["text"], dx=10, dy=10)

    summary = run_qa_refinement_for_job(
        job_id=form.job_id,
        output_dir=form.output_dir,
        settings=_settings(tmp_path, grounding_qa_max_iterations=6),
        judge_fn=_targeting_judge({form.field_ids["text"]: truth}),
    )

    manifest = json.loads((form.root / "job.json").read_text(encoding="utf-8"))
    stage = manifest["stages"]["qa_refine"]
    assert stage["status"] == "done"
    assert stage["iterations"] == summary["iterations"]
    assert stage["converged"] is True
    assert stage["cost"]["judge_calls"] >= 1
    assert stage["cost"]["total_tokens"] > 0
    assert "tokens_per_page_per_iteration" in stage["cost"]


def test_refine_requires_grounding(tmp_path: Path) -> None:
    root = tmp_path / _JOB_ID
    (root / "output").mkdir(parents=True)
    from app.services.jobs import new_job_manifest, write_job_manifest

    write_job_manifest(root, new_job_manifest(job_id=_JOB_ID, source_filename="x.pdf"))
    with pytest.raises(qa.QaRefinementError):
        run_qa_refinement_for_job(
            job_id=_JOB_ID,
            output_dir=root / "output",
            settings=_settings(tmp_path),
        )


# --------------------------------------------------------------------------- endpoint


def test_refine_endpoint_returns_202(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    form = build_parity_job(jobs_dir, variant="form_a", job_id=_JOB_ID)
    assert form.root.is_dir()

    settings = _settings(jobs_dir)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as client:
        resp = client.post(f"/api/v1/jobs/{_JOB_ID}/refine-grounding")
    assert resp.status_code == 202
    assert resp.json()["status"] == "running"


def test_refine_endpoint_400_without_grounding(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    root = jobs_dir / _JOB_ID
    (root / "output").mkdir(parents=True)
    from app.services.jobs import new_job_manifest, write_job_manifest

    write_job_manifest(root, new_job_manifest(job_id=_JOB_ID, source_filename="x.pdf"))

    settings = _settings(jobs_dir)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as client:
        resp = client.post(f"/api/v1/jobs/{_JOB_ID}/refine-grounding")
    assert resp.status_code == 400
    assert resp.json()["error"] == "grounding_not_found"
