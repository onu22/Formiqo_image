"""Service tests for provider/model-aware grounding output layout."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.field_grounding import run_field_grounding_for_job


def _write_min_png(path: Path) -> None:
    # 1x1 transparent PNG
    path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A"
            "0000000D49484452000000010000000108060000001F15C489"
            "0000000A49444154789C6360000000020001E221BC33"
            "0000000049454E44AE426082"
        )
    )


def test_run_field_grounding_writes_provider_model_run_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "output"
    images_dir = output_dir / "converted_images"
    images_dir.mkdir(parents=True)
    _write_min_png(images_dir / "page_0001.png")

    monkeypatch.setattr("app.services.field_grounding._create_provider_client", lambda **kwargs: object())
    monkeypatch.setattr(
        "app.services.field_grounding._call_provider_for_page",
        lambda **kwargs: {
            "page_index": kwargs["page_index"],
            "width": kwargs["width"],
            "height": kwargs["height"],
            "unit": "px",
            "origin": "top-left",
            "fields": [
                {"field_id": "field_1", "type": "text", "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}}
            ],
        },
    )

    result = run_field_grounding_for_job(
        job_id="11111111-1111-4111-8111-111111111111",
        output_dir=output_dir,
        provider="openai",
        model="gpt-4o",
        openai_api_key="dummy",
        openai_timeout_seconds=30,
        anthropic_api_key="",
        anthropic_timeout_seconds=30,
    )

    assert result["provider"] == "openai"
    assert result["model"] == "gpt-4o"
    assert result["run_dir"] == "field_grounding/openai_gpt-4o"

    page_path = output_dir / "field_grounding" / "openai_gpt-4o" / "page_0001.fields.json"
    data = json.loads(page_path.read_text(encoding="utf-8"))
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-4o"
    assert isinstance(data["run_id"], str) and data["run_id"]

    manifest_path = output_dir / "field_grounding" / "openai_gpt-4o" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["provider"] == "openai"
    assert manifest["model"] == "gpt-4o"
    assert manifest["run_dir"] == "field_grounding/openai_gpt-4o"
    assert result["stamping_sample_path"] == "field_grounding/openai_gpt-4o/stamping.json"
    assert manifest["stamping_sample_path"] == "field_grounding/openai_gpt-4o/stamping.json"
    sample_path = output_dir / result["stamping_sample_path"]
    assert sample_path.is_file()
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    assert sample["require_all_values"] is False
    assert sample["values"]["field_1"] == "field_1"


def test_stamping_sample_uses_first_10_chars_and_dedupes(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "output"
    images_dir = output_dir / "converted_images"
    images_dir.mkdir(parents=True)
    _write_min_png(images_dir / "page_0001.png")
    _write_min_png(images_dir / "page_0002.png")

    monkeypatch.setattr("app.services.field_grounding._create_provider_client", lambda **kwargs: object())

    def fake_call(**kwargs):
        if kwargs["page_index"] == 0:
            return {
                "page_index": 0,
                "width": kwargs["width"],
                "height": kwargs["height"],
                "unit": "px",
                "origin": "top-left",
                "fields": [
                    {
                        "field_id": "your_full_legal_name.given_name",
                        "type": "text",
                        "bbox": {"x": 0, "y": 0, "w": 1, "h": 1},
                    }
                ],
            }
        return {
            "page_index": 1,
            "width": kwargs["width"],
            "height": kwargs["height"],
            "unit": "px",
            "origin": "top-left",
            "fields": [
                {
                    "field_id": "your_full_legal_name.given_name",
                    "type": "text",
                    "bbox": {"x": 0, "y": 0, "w": 1, "h": 1},
                },
                {
                    "field_id": "another_field_id",
                    "type": "text",
                    "bbox": {"x": 0, "y": 0, "w": 1, "h": 1},
                },
            ],
        }

    monkeypatch.setattr("app.services.field_grounding._call_provider_for_page", fake_call)

    result = run_field_grounding_for_job(
        job_id="22222222-2222-4222-8222-222222222222",
        output_dir=output_dir,
        provider="openai",
        model="gpt-4o",
        openai_api_key="dummy",
        openai_timeout_seconds=30,
        anthropic_api_key="",
        anthropic_timeout_seconds=30,
    )

    sample_path = output_dir / result["stamping_sample_path"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    assert sample["values"]["your_full_legal_name.given_name"] == "your_full_"
    assert sample["values"]["another_field_id"] == "another_fi"
