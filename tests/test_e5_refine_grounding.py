"""E5 vision QA refinement loop: closed loop, bounded deltas, qa_status, endpoint.

The judge is dependency-injected so the loop is exercised deterministically without any
network calls. An *oracle* judge that knows the un-perturbed ("truth") bboxes stands in for
the vision model: it returns the remaining shift toward truth, which the loop clamps to the
per-iteration bound. This lets us assert convergence and the never-exceed-bound guarantee.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.dependencies import get_settings
from app.main import create_app
from app.services import grounding_qa as gqa
from app.services.grounding_qa import (
    FieldVerdict,
    JudgeUsage,
    clamp_delta,
    consensus_translation,
    crop_field_region,
    refine_grounding_for_job,
)
from app.services.grounding_qa_schema import (
    QA_TOOL_NAME,
    anthropic_qa_tool,
    openai_qa_response_format,
)
from tests.fixtures.parity.builder import build_parity_job

_JOB_ID = "22222222-2222-4222-8222-222222222222"


# --------------------------------------------------------------------------- schema


def test_openai_qa_schema_is_closed_and_fully_required() -> None:
    fmt = openai_qa_response_format()
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"].keys())
    assert schema["properties"]["verdict"]["enum"] == ["ok", "shift"]


def test_anthropic_qa_tool_uses_relaxed_schema() -> None:
    tool = anthropic_qa_tool()
    assert tool["name"] == QA_TOOL_NAME
    schema = tool["input_schema"]
    assert schema["required"] == ["verdict", "dx", "dy"]


# --------------------------------------------------------------------------- pure units


def test_clamp_delta_bounds_both_directions() -> None:
    assert clamp_delta(100, 30) == 30
    assert clamp_delta(-100, 30) == -30
    assert clamp_delta(7, 30) == 7


def test_consensus_translation_merges_tight_agreement() -> None:
    merged = consensus_translation([(10, 4), (11, 5), (9, 6)], min_fields=3, max_spread=4)
    assert merged == (10, 5)


def test_consensus_translation_declines_when_spread_too_wide() -> None:
    assert consensus_translation([(10, 0), (30, 0), (-5, 0)], min_fields=3, max_spread=4) is None


def test_consensus_translation_declines_below_min_fields() -> None:
    assert consensus_translation([(10, 5), (10, 5)], min_fields=3, max_spread=4) is None


def test_crop_field_region_returns_offset_and_stays_in_bounds() -> None:
    img = Image.new("RGB", (400, 300), "white")
    crop, x0, y0 = crop_field_region(
        img, {"x": 100, "y": 100, "w": 40, "h": 20}, zoom=3.0, padding=10, page_w=400, page_h=300
    )
    assert x0 >= 0 and y0 >= 0
    assert crop.width <= 400 and crop.height <= 300
    # Region is centered on the bbox and wider than the bbox itself (context).
    assert crop.width > 40


# --------------------------------------------------------------------------- loop harness


class _OracleJudge:
    """Returns the remaining shift toward known-truth bboxes (loop clamps to the bound)."""

    def __init__(self, truth: dict[str, dict[str, int]], *, confidence: float = 0.9) -> None:
        self.truth = truth
        self.confidence = confidence
        self.usage = JudgeUsage()

    def judge_field(self, *, page_index: int, field: dict[str, Any], value: str, crop, crop_offset):  # noqa: ANN001
        self.usage.record(input_tokens=120, output_tokens=12, latency_s=0.001)
        fid = field["field_id"]
        cur = field["bbox"]
        t = self.truth[fid]
        dx = int(t["x"]) - int(cur["x"])
        dy = int(t["y"]) - int(cur["y"])
        if dx == 0 and dy == 0:
            return FieldVerdict("ok", 0, 0, confidence=0.96)
        return FieldVerdict("shift", dx, dy, confidence=self.confidence)


def _truth_bboxes(output_dir: Path) -> dict[str, dict[str, int]]:
    payload = json.loads((output_dir / "field_grounding" / "page_0001.fields.json").read_text())
    return {f["field_id"]: dict(f["bbox"]) for f in payload["fields"]}


def _perturb(output_dir: Path, offsets: dict[str, tuple[int, int]]) -> None:
    path = output_dir / "field_grounding" / "page_0001.fields.json"
    payload = json.loads(path.read_text())
    for field in payload["fields"]:
        off = offsets.get(field["field_id"])
        if off:
            field["bbox"]["x"] += off[0]
            field["bbox"]["y"] += off[1]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_fields(output_dir: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((output_dir / "field_grounding" / "page_0001.fields.json").read_text())
    return {f["field_id"]: f for f in payload["fields"]}


def _settings(jobs_dir: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = dict(jobs_dir=jobs_dir, openai_api_key="test-key")
    base.update(overrides)
    return Settings(**base)


def test_loop_corrects_perturbed_bboxes_and_marks_adjusted(tmp_path: Path) -> None:
    form = build_parity_job(tmp_path, variant="form_a", job_id=_JOB_ID)
    truth = _truth_bboxes(form.output_dir)
    # Different per-field offsets (spread > consensus threshold → per-field correction path).
    _perturb(form.output_dir, {
        form.field_ids["text"]: (18, -12),
        form.field_ids["multiline"]: (-20, 9),
        form.field_ids["checkbox"]: (7, 15),
    })

    judge = _OracleJudge(truth)
    summary = refine_grounding_for_job(
        job_id=form.job_id,
        output_dir=form.output_dir,
        settings=_settings(tmp_path, grounding_qa_max_bbox_delta_px=30),
        judge=judge,
    )

    assert summary["converged"] is True
    fields = _read_fields(form.output_dir)
    for fid, t in truth.items():
        assert fields[fid]["bbox"]["x"] == t["x"], fid
        assert fields[fid]["bbox"]["y"] == t["y"], fid
        assert fields[fid]["qa_status"] == "adjusted"
    assert summary["counts"]["adjusted"] == 3


def test_loop_never_exceeds_per_iteration_bound_and_flags_nonconverged(tmp_path: Path) -> None:
    form = build_parity_job(tmp_path, variant="form_b", job_id=_JOB_ID)
    truth = _truth_bboxes(form.output_dir)
    text_id = form.field_ids["text"]
    start = dict(truth[text_id])
    # Perturb far beyond one step; a single iteration must move by at most the bound.
    _perturb(form.output_dir, {text_id: (100, 0)})

    judge = _OracleJudge(truth)
    summary = refine_grounding_for_job(
        job_id=form.job_id,
        output_dir=form.output_dir,
        settings=_settings(
            tmp_path,
            grounding_qa_max_bbox_delta_px=30,
            grounding_qa_max_iterations=1,
            grounding_qa_consensus_translation_enabled=False,
        ),
        judge=judge,
    )

    fields = _read_fields(form.output_dir)
    moved = fields[text_id]["bbox"]["x"]
    perturbed_x = start["x"] + 100
    # Moved toward truth by exactly the bound (30), never more.
    assert perturbed_x - moved == 30
    assert summary["converged"] is False
    assert fields[text_id]["qa_status"] == "flagged"


def test_loop_confirms_correct_fields_without_moving(tmp_path: Path) -> None:
    form = build_parity_job(tmp_path, variant="form_a", job_id=_JOB_ID)
    truth = _truth_bboxes(form.output_dir)  # no perturbation → already correct
    judge = _OracleJudge(truth)
    summary = refine_grounding_for_job(
        job_id=form.job_id, output_dir=form.output_dir, settings=_settings(tmp_path), judge=judge
    )
    assert summary["converged"] is True
    assert summary["iterations"] == 1
    fields = _read_fields(form.output_dir)
    for fid, t in truth.items():
        assert fields[fid]["bbox"] == t
        assert fields[fid]["qa_status"] == "confirmed"


def test_low_confidence_ok_is_flagged(tmp_path: Path) -> None:
    form = build_parity_job(tmp_path, variant="form_a", job_id=_JOB_ID)
    truth = _truth_bboxes(form.output_dir)

    class _LowConf:
        usage = JudgeUsage()

        def judge_field(self, **_: Any) -> FieldVerdict:
            return FieldVerdict("ok", 0, 0, confidence=0.1)

    summary = refine_grounding_for_job(
        job_id=form.job_id,
        output_dir=form.output_dir,
        settings=_settings(tmp_path, grounding_qa_min_confidence=0.4),
        judge=_LowConf(),
    )
    assert summary["counts"]["flagged"] == 3


def test_per_field_judge_error_isolated(tmp_path: Path) -> None:
    form = build_parity_job(tmp_path, variant="form_a", job_id=_JOB_ID)
    truth = _truth_bboxes(form.output_dir)
    bad_id = form.field_ids["multiline"]

    class _FlakyJudge(_OracleJudge):
        def judge_field(self, *, page_index, field, value, crop, crop_offset):  # noqa: ANN001
            if field["field_id"] == bad_id:
                raise RuntimeError("simulated judge error")
            return super().judge_field(
                page_index=page_index, field=field, value=value, crop=crop, crop_offset=crop_offset
            )

    refine_grounding_for_job(
        job_id=form.job_id, output_dir=form.output_dir, settings=_settings(tmp_path), judge=_FlakyJudge(truth)
    )
    fields = _read_fields(form.output_dir)
    # The erroring field stays unevaluated; the others are confirmed.
    assert fields[bad_id]["qa_status"] is None
    assert fields[form.field_ids["text"]]["qa_status"] == "confirmed"


def test_empty_value_text_field_is_skipped(tmp_path: Path) -> None:
    form = build_parity_job(tmp_path, variant="form_a", job_id=_JOB_ID)
    # Blank out the text value; it should not be judged (nothing stamped to assess).
    stamping_path = form.output_dir / "field_grounding" / "stamping.json"
    stamping = json.loads(stamping_path.read_text())
    stamping["values"][form.field_ids["text"]] = ""
    stamping_path.write_text(json.dumps(stamping, indent=2) + "\n", encoding="utf-8")

    truth = _truth_bboxes(form.output_dir)
    refine_grounding_for_job(
        job_id=form.job_id, output_dir=form.output_dir, settings=_settings(tmp_path), judge=_OracleJudge(truth)
    )
    fields = _read_fields(form.output_dir)
    assert fields[form.field_ids["text"]]["qa_status"] is None


def test_qa_refine_stage_recorded_with_cost(tmp_path: Path) -> None:
    from app.services.jobs import read_job_manifest

    form = build_parity_job(tmp_path, variant="form_a", job_id=_JOB_ID)
    truth = _truth_bboxes(form.output_dir)
    refine_grounding_for_job(
        job_id=form.job_id, output_dir=form.output_dir, settings=_settings(tmp_path), judge=_OracleJudge(truth)
    )
    manifest = read_job_manifest(form.root)
    qa = manifest["stages"]["qa_refine"]
    assert qa["status"] == "done"
    assert qa["iterations"] >= 1
    assert "cost" in qa and qa["cost"]["judge_calls"] > 0
    assert "input_tokens_per_page_per_iteration" in qa["cost"]


# --------------------------------------------------------------------------- endpoint


def test_refine_endpoint_schedules_loop_and_updates_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs_dir = tmp_path
    form = build_parity_job(jobs_dir, variant="form_a", job_id=_JOB_ID)
    truth = _truth_bboxes(form.output_dir)
    _perturb(form.output_dir, {form.field_ids["text"]: (10, 8)})

    # Inject a deterministic judge in place of the provider-backed one.
    monkeypatch.setattr(gqa, "build_default_judge", lambda settings, **_: _OracleJudge(truth))

    settings = Settings(jobs_dir=jobs_dir, openai_api_key="test-key")
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)

    resp = client.post(f"/api/v1/jobs/{form.job_id}/refine-grounding")
    assert resp.status_code == 202
    assert resp.json()["status"] == "running"

    detail = client.get(f"/api/v1/jobs/{form.job_id}").json()
    assert detail["stages"]["qa_refine"]["status"] == "done"
    assert detail["stages"]["qa_refine"]["converged"] is True

    fields = _read_fields(form.output_dir)
    assert fields[form.field_ids["text"]]["bbox"] == truth[form.field_ids["text"]]


def test_refine_endpoint_missing_fields_returns_400(tmp_path: Path) -> None:
    import shutil

    jobs_dir = tmp_path
    form = build_parity_job(jobs_dir, variant="form_a", job_id=_JOB_ID)
    shutil.rmtree(form.output_dir / "field_grounding")

    settings = Settings(jobs_dir=jobs_dir, openai_api_key="test-key")
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)

    resp = client.post(f"/api/v1/jobs/{form.job_id}/refine-grounding")
    assert resp.status_code == 400
