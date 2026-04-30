"""Service tests for provider/model-aware grounding output layout."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.field_grounding import GroundingJsonParseError, run_field_grounding_for_job


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
        openai_max_output_tokens=9600,
        anthropic_api_key="",
        anthropic_timeout_seconds=30,
        anthropic_max_tokens=4800,
    )

    assert result["provider"] == "openai"
    assert result["model"] == "gpt-4o"
    assert result["run_dir"] == "field_grounding"
    assert result["pages"][0]["resolution"] == "direct"

    page_path = output_dir / "field_grounding" / "page_0001.fields.json"
    data = json.loads(page_path.read_text(encoding="utf-8"))
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-4o"
    assert isinstance(data["run_id"], str) and data["run_id"]

    manifest_path = output_dir / "field_grounding" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["provider"] == "openai"
    assert manifest["model"] == "gpt-4o"
    assert manifest["run_dir"] == "field_grounding"
    assert result["stamping_sample_path"] == "field_grounding/stamping.json"
    assert manifest["stamping_sample_path"] == "field_grounding/stamping.json"
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
        openai_max_output_tokens=9600,
        anthropic_api_key="",
        anthropic_timeout_seconds=30,
        anthropic_max_tokens=4800,
    )

    sample_path = output_dir / result["stamping_sample_path"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    assert sample["values"]["your_full_legal_name.given_name"] == "your_full_"
    assert sample["values"]["another_field_id"] == "another_fi"


def test_page_index_filters_to_single_page(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "output"
    images_dir = output_dir / "converted_images"
    images_dir.mkdir(parents=True)
    _write_min_png(images_dir / "page_0001.png")
    _write_min_png(images_dir / "page_0002.png")

    monkeypatch.setattr("app.services.field_grounding._create_provider_client", lambda **kwargs: object())
    called_pages: list[int] = []

    def fake_call(**kwargs):
        called_pages.append(kwargs["page_index"])
        return {
            "page_index": kwargs["page_index"],
            "width": kwargs["width"],
            "height": kwargs["height"],
            "unit": "px",
            "origin": "top-left",
            "fields": [{"field_id": f"field_{kwargs['page_index']}", "type": "text", "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}}],
        }

    monkeypatch.setattr("app.services.field_grounding._call_provider_for_page", fake_call)
    result = run_field_grounding_for_job(
        job_id="33333333-3333-4333-8333-333333333333",
        output_dir=output_dir,
        provider="openai",
        model="gpt-4o",
        page_index=1,
        openai_api_key="dummy",
        openai_timeout_seconds=30,
        openai_max_output_tokens=9600,
        anthropic_api_key="",
        anthropic_timeout_seconds=30,
        anthropic_max_tokens=4800,
    )

    assert called_pages == [1]
    assert result["page_count"] == 1
    assert (output_dir / "field_grounding" / "page_0002.fields.json").is_file()
    assert not (output_dir / "field_grounding" / "page_0001.fields.json").exists()


def test_page_index_out_of_range_raises_value_error(tmp_path: Path, monkeypatch) -> None:
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
            "fields": [],
        },
    )

    try:
        run_field_grounding_for_job(
            job_id="44444444-4444-4444-8444-444444444444",
            output_dir=output_dir,
            provider="openai",
            model="gpt-4o",
            page_index=5,
            openai_api_key="dummy",
            openai_timeout_seconds=30,
            openai_max_output_tokens=9600,
            anthropic_api_key="",
            anthropic_timeout_seconds=30,
            anthropic_max_tokens=4800,
        )
    except ValueError as exc:
        assert "page_index 5 not found" in str(exc)
    else:
        raise AssertionError("Expected ValueError for out-of-range page_index")


def test_invalid_json_saves_raw_model_output_file(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "output"
    images_dir = output_dir / "converted_images"
    images_dir.mkdir(parents=True)
    _write_min_png(images_dir / "page_0001.png")

    monkeypatch.setattr("app.services.field_grounding._create_provider_client", lambda **kwargs: object())
    monkeypatch.setattr(
        "app.services.field_grounding._call_provider_for_page",
        lambda **kwargs: (_ for _ in ()).throw(
            GroundingJsonParseError("Invalid JSON from model response", "{\"broken\": \"json")
        ),
    )

    result = run_field_grounding_for_job(
        job_id="55555555-5555-4555-8555-555555555555",
        output_dir=output_dir,
        provider="openai",
        model="gpt-4o",
        openai_api_key="dummy",
        openai_timeout_seconds=30,
        openai_max_output_tokens=9600,
        anthropic_api_key="",
        anthropic_timeout_seconds=30,
        anthropic_max_tokens=4800,
    )

    assert result["succeeded_count"] == 0
    assert result["failed_count"] == 1
    page = result["pages"][0]
    assert page["status"] == "failed"
    assert page["raw_model_output_file"] == "field_grounding/page_0001.failed.raw.txt"
    raw_path = output_dir / page["raw_model_output_file"]
    assert raw_path.is_file()
    assert raw_path.read_text(encoding="utf-8") == "{\"broken\": \"json"


def test_json_parse_failure_retries_and_succeeds(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "output"
    images_dir = output_dir / "converted_images"
    images_dir.mkdir(parents=True)
    _write_min_png(images_dir / "page_0001.png")

    monkeypatch.setattr("app.services.field_grounding._create_provider_client", lambda **kwargs: object())
    attempts = {"count": 0}

    def fake_call(**kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise GroundingJsonParseError("first parse fail", "{\"bad\":")
        return {
            "page_index": kwargs["page_index"],
            "width": kwargs["width"],
            "height": kwargs["height"],
            "unit": "px",
            "origin": "top-left",
            "fields": [{"field_id": "field_1", "type": "text", "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}}],
        }

    monkeypatch.setattr("app.services.field_grounding._call_provider_for_page", fake_call)
    result = run_field_grounding_for_job(
        job_id="66666666-6666-4666-8666-666666666666",
        output_dir=output_dir,
        provider="openai",
        model="gpt-4o",
        openai_api_key="dummy",
        openai_timeout_seconds=30,
        openai_max_output_tokens=9600,
        anthropic_api_key="",
        anthropic_timeout_seconds=30,
        anthropic_max_tokens=4800,
    )

    assert attempts["count"] == 2
    assert result["succeeded_count"] == 1
    assert result["failed_count"] == 0
    assert result["pages"][0]["resolution"] == "retry"


def test_json_parse_failure_uses_repair_pass(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "output"
    images_dir = output_dir / "converted_images"
    images_dir.mkdir(parents=True)
    _write_min_png(images_dir / "page_0001.png")

    monkeypatch.setattr("app.services.field_grounding._create_provider_client", lambda **kwargs: object())
    monkeypatch.setattr(
        "app.services.field_grounding._call_provider_for_page",
        lambda **kwargs: (_ for _ in ()).throw(GroundingJsonParseError("parse fail", "{\"oops\":")),
    )
    repaired = {"called": False}

    def fake_repair(**kwargs):
        repaired["called"] = True
        return {
            "page_index": kwargs["page_index"],
            "width": kwargs["width"],
            "height": kwargs["height"],
            "unit": "px",
            "origin": "top-left",
            "fields": [{"field_id": "field_1", "type": "text", "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}}],
        }

    monkeypatch.setattr("app.services.field_grounding._repair_with_provider", fake_repair)
    result = run_field_grounding_for_job(
        job_id="77777777-7777-4777-8777-777777777777",
        output_dir=output_dir,
        provider="openai",
        model="gpt-4o",
        openai_api_key="dummy",
        openai_timeout_seconds=30,
        openai_max_output_tokens=9600,
        anthropic_api_key="",
        anthropic_timeout_seconds=30,
        anthropic_max_tokens=4800,
    )

    assert repaired["called"] is True
    assert result["succeeded_count"] == 1
    assert result["failed_count"] == 0
    assert result["pages"][0]["resolution"] == "repair"
