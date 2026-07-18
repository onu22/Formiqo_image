"""Tests for dev auto-stamp fixture mapping and selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.services.dev_auto_stamp import (
    filter_to_known_fields,
    fixture_to_values,
    map_imm5645e,
    resolve_fixture_path,
    run_dev_auto_stamp_after_grounding,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "applicant-data" / "imm5645e-okafor.json"
)


def test_map_imm5645e_from_committed_fixture() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    values = fixture_to_values(payload)

    assert values["section_a_applicant_name"] == "Chukwudi Emmanuel Okafor"
    assert values["section_a_spouse_or_common_law_partner_present_occupation"] == "Registered Nurse"
    assert values["section_a_mother_will_accompany_yes"] == ""
    assert values["section_a_mother_will_accompany_no"] == "true"
    assert values["section_b_child_1_relationship"] == "Son"
    assert values["section_b_child_2_name"] == "Chinaza Esther Okafor"
    assert values["sibling_4_present_occupation"] == "University Student"
    assert values["certification_signature"] == "Chukwudi Emmanuel Okafor"
    assert values["certification_date"] == "2026-07-17"


def test_fixture_flat_values_format() -> None:
    values = fixture_to_values({"values": {"a_name": "Ada", "a_yes": True}})
    assert values == {"a_name": "Ada", "a_yes": "true"}


def test_fixture_bare_flat_map() -> None:
    values = fixture_to_values({"field_one": "x", "field_two": 3})
    assert values == {"field_one": "x", "field_two": "3"}


def test_unknown_mapper_raises() -> None:
    with pytest.raises(ValueError, match="Unknown fixture mapper"):
        fixture_to_values({"mapper": "nope", "data": {}})


def test_filter_to_known_fields() -> None:
    mapped = map_imm5645e(
        {
            "sectionA": {"applicant": {"name": "A", "dateOfBirth": "2000-01-01"}},
            "sectionD": {"certification": {"signature": "A", "date": "2026-01-01"}},
        }
    )
    known = {"section_a_applicant_name", "certification_date", "unrelated"}
    filtered = filter_to_known_fields(mapped, known)
    assert filtered == {
        "section_a_applicant_name": "A",
        "certification_date": "2026-01-01",
    }


def test_resolve_fixture_by_source_filename(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "other.json").write_text(
        json.dumps({"id": "other", "match": ["xyz"], "values": {"a": "1"}}),
        encoding="utf-8",
    )
    target = fixture_dir / "imm.json"
    target.write_text(
        json.dumps({"id": "imm", "match": ["imm5645e"], "values": {"a": "2"}}),
        encoding="utf-8",
    )
    settings = Settings(
        dev_auto_stamp_fixture="",
        dev_auto_stamp_fixture_dir=fixture_dir,
    )
    resolved = resolve_fixture_path(settings, source_filename="imm5645e-1.pdf")
    assert resolved == target


def test_resolve_explicit_fixture_overrides_match(tmp_path: Path) -> None:
    explicit = tmp_path / "forced.json"
    explicit.write_text(json.dumps({"values": {"x": "1"}}), encoding="utf-8")
    settings = Settings(
        dev_auto_stamp_fixture=str(explicit),
        dev_auto_stamp_fixture_dir=tmp_path / "unused",
    )
    assert resolve_fixture_path(settings, source_filename="anything.pdf") == explicit


def test_run_dev_auto_stamp_disabled_returns_none(tmp_path: Path) -> None:
    settings = Settings(dev_auto_stamp_after_grounding=False)
    result = run_dev_auto_stamp_after_grounding(
        job_id="j",
        job_root=tmp_path,
        input_pdf=tmp_path / "in.pdf",
        output_dir=tmp_path / "out",
        settings=settings,
        source_filename="imm5645e.pdf",
    )
    assert result is None


def test_run_dev_auto_stamp_applies_values_and_calls_stampers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_root = tmp_path / "job"
    output_dir = job_root / "output"
    fg = output_dir / "field_grounding"
    fg.mkdir(parents=True)
    (job_root / "job.json").write_text(
        json.dumps(
            {
                "job_id": "j1",
                "grounding": {"provider": "openai", "model": "gpt-test"},
            }
        ),
        encoding="utf-8",
    )
    (fg / "stamping.json").write_text(
        json.dumps(
            {
                "values": {
                    "section_a_applicant_name": "",
                    "certification_date": "",
                    "other_field": "",
                },
                "require_all_values": False,
                "style": {"font_size_pt": 11.0, "text_color": "#111111"},
                "overrides": {},
            }
        ),
        encoding="utf-8",
    )
    fixture = tmp_path / "sample.json"
    fixture.write_text(
        json.dumps(
            {
                "id": "t",
                "mapper": "imm5645e",
                "data": {
                    "sectionA": {"applicant": {"name": "Pat"}},
                    "sectionD": {"certification": {"date": "2026-07-17"}},
                },
            }
        ),
        encoding="utf-8",
    )
    input_pdf = tmp_path / "in.pdf"
    input_pdf.write_bytes(b"%PDF-1.4")

    calls: list[str] = []

    def fake_images(**kwargs):  # type: ignore[no-untyped-def]
        calls.append("images")
        assert kwargs["values"]["section_a_applicant_name"] == "Pat"
        assert kwargs["values"]["certification_date"] == "2026-07-17"
        assert kwargs["values"]["other_field"] == ""
        return {"stamp_run_id": "img-run"}

    def fake_pdf(**kwargs):  # type: ignore[no-untyped-def]
        calls.append("pdf")
        return {"stamp_run_id": "pdf-run"}

    monkeypatch.setattr("app.services.image_stamping.run_image_stamping_for_job", fake_images)
    monkeypatch.setattr("app.services.pdf_stamping.run_pdf_stamping_for_job", fake_pdf)

    settings = Settings(
        dev_auto_stamp_after_grounding=True,
        dev_auto_stamp_fixture=str(fixture),
        dev_auto_stamp_mode="both",
    )
    summary = run_dev_auto_stamp_after_grounding(
        job_id="j1",
        job_root=job_root,
        input_pdf=input_pdf,
        output_dir=output_dir,
        settings=settings,
        source_filename="imm5645e.pdf",
    )
    assert summary is not None
    assert summary["applied_count"] == 2
    assert calls == ["images", "pdf"]
    persisted = json.loads((fg / "stamping.json").read_text(encoding="utf-8"))
    assert persisted["values"]["section_a_applicant_name"] == "Pat"
