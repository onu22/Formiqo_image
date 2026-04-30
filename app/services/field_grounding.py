"""Field grounding service for existing conversion jobs."""

from __future__ import annotations

import base64
import json
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from openai import OpenAI

_PAGE_IMAGE_RE = re.compile(r"^page_(\d{4})\.png$")
_SUPPORTED_PROVIDERS = {"openai", "anthropic"}
_MAX_PARSE_RETRIES = 1

_PROMPT_TEMPLATE = """You are a document field-grounding assistant.

Analyze the provided form image and return ONLY valid JSON in the exact format below.

IMPORTANT:
The true source image dimensions are:
- IMAGE_WIDTH: {{IMAGE_WIDTH}}
- IMAGE_HEIGHT: {{IMAGE_HEIGHT}}

The source image dimensions are authoritative and provided externally by the calling system:
width={{IMAGE_WIDTH}}, height={{IMAGE_HEIGHT}}.

These values are ground truth.
Do not substitute any dimensions derived from your own image preprocessing pipeline.
If your internal working copy has different dimensions, rescale your detected boxes back into the authoritative source coordinate space before returning JSON.

You MUST use these exact dimensions in the output.
Do NOT infer, estimate, resize, normalize, or replace them with internal processing dimensions, viewport dimensions, or downscaled dimensions.

Required output format:

{
  "page_index": 0,
  "width": {{IMAGE_WIDTH}},
  "height": {{IMAGE_HEIGHT}},
  "unit": "px",
  "origin": "top-left",
  "fields": [
    {
      "field_id": "<field_id>",
      "type": "text",
      "bbox": { "x": <int>, "y": <int>, "w": <int>, "h": <int> }
    }
  ]
}

Rules:
1. Return JSON only. No markdown. No prose. No explanation.
2. The values of "width" and "height" MUST be exactly {{IMAGE_WIDTH}} and {{IMAGE_HEIGHT}}.
3. All bounding boxes MUST use the same coordinate space as the true source image dimensions above.
4. Do not return coordinates based on resized, processed, internal, or normalized copies of the image.
5. Detect only user-writable input regions.
6. Ignore titles, labels, instructions, paragraph text, and decorative lines unless they are part of a fillable area.
7. Bounding boxes must cover the writable/stamping area only, not the label area.
8. Use integer pixel values only.
9. Keep fields ordered top-to-bottom, then left-to-right.
10. field_id must be lowercase and deterministic.
11. Use snake_case names derived from nearby printed labels where possible.
12. For repeated groups or columns, use consistent names such as:
    - qualification_1.qualification_name
    - qualification_2.qualification_name
    - qualification_3.qualification_name
13. If no semantic name is obvious, use deterministic fallback names like:
    - field_1
    - field_2
14. Each field object must contain exactly:
    - field_id
    - type
    - bbox
15. Each bbox must contain exactly:
    - x
    - y
    - w
    - h
16. Do not include extra keys such as confidence, notes, labels, sections, metadata, or reasoning.
17. All bbox values must lie within the source image bounds.
18. Do not output zero-size or negative-size boxes.
19. If uncertain, still return the best deterministic field boxes in the required JSON format.

Before finalizing your answer, verify:
- width == {{IMAGE_WIDTH}}
- height == {{IMAGE_HEIGHT}}
- every bbox is in the source image coordinate system

Now analyze the provided form image and return the JSON only.
"""

_REPAIR_PROMPT_TEMPLATE = """You are a JSON repair assistant.

You receive malformed text that was intended to be a single JSON object for document field grounding.
Return ONLY a valid JSON object that matches this exact schema and constraints:

{
  "page_index": <int exactly {{PAGE_INDEX}}>,
  "width": <int exactly {{IMAGE_WIDTH}}>,
  "height": <int exactly {{IMAGE_HEIGHT}}>,
  "unit": "px",
  "origin": "top-left",
  "fields": [
    {
      "field_id": "<non-empty string>",
      "type": "<non-empty string>",
      "bbox": { "x": <int>, "y": <int>, "w": <int>, "h": <int> }
    }
  ]
}

Rules:
1. Return JSON only. No markdown. No prose.
2. Root keys must be exactly: page_index, width, height, unit, origin, fields.
3. page_index must be {{PAGE_INDEX}}.
4. width must be {{IMAGE_WIDTH}} and height must be {{IMAGE_HEIGHT}}.
5. unit must be "px" and origin must be "top-left".
6. Every field must contain exactly: field_id, type, bbox.
7. Every bbox must contain exactly: x, y, w, h.
8. bbox values must be integers.
9. Do not add any extra keys anywhere.
10. Preserve as much valid field content as possible from the malformed input.

Malformed input:
{{BROKEN_JSON}}
"""


class GroundingJsonParseError(ValueError):
    """Raised when model text cannot be parsed as valid JSON."""

    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


def _read_png_dimensions(image_path: Path) -> tuple[int, int]:
    """Return authoritative PNG dimensions from IHDR."""
    with image_path.open("rb") as fh:
        header = fh.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {image_path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid PNG dimensions in {image_path}")
    return int(width), int(height)


def _discover_page_images(output_dir: Path) -> list[tuple[int, Path]]:
    images_dir = output_dir / "converted_images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"converted_images directory not found: {images_dir}")

    pages: list[tuple[int, Path]] = []
    for path in sorted(images_dir.glob("page_*.png")):
        m = _PAGE_IMAGE_RE.match(path.name)
        if not m:
            continue
        page_index = int(m.group(1)) - 1
        pages.append((page_index, path))
    if not pages:
        raise FileNotFoundError(f"No page PNG files found under: {images_dir}")
    return pages


def _prompt_for_dimensions(width: int, height: int) -> str:
    return _PROMPT_TEMPLATE.replace("{{IMAGE_WIDTH}}", str(width)).replace("{{IMAGE_HEIGHT}}", str(height))


def _repair_prompt_for_dimensions(*, page_index: int, width: int, height: int, broken_json: str) -> str:
    return (
        _REPAIR_PROMPT_TEMPLATE.replace("{{PAGE_INDEX}}", str(page_index))
        .replace("{{IMAGE_WIDTH}}", str(width))
        .replace("{{IMAGE_HEIGHT}}", str(height))
        .replace("{{BROKEN_JSON}}", broken_json)
    )


def _extract_json_text(response: Any) -> str:
    text = getattr(response, "output_text", "")
    if isinstance(text, str) and text.strip():
        return text
    raise ValueError("OpenAI response did not include output_text")


def _extract_json_text_anthropic(response: Any) -> str:
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        raise ValueError("Anthropic response did not include content blocks")
    texts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            texts.append(text)
    joined = "\n".join(texts).strip()
    if not joined:
        raise ValueError("Anthropic response did not include text content")
    return joined


def _validate_field_grounding_json(data: Any, *, page_index: int, width: int, height: int) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Root must be a JSON object")

    expected_root_keys = {"page_index", "width", "height", "unit", "origin", "fields"}
    got_root_keys = set(data.keys())
    if got_root_keys != expected_root_keys:
        raise ValueError(f"Root keys must be exactly {sorted(expected_root_keys)}")

    if data["page_index"] != page_index:
        raise ValueError(f"page_index must equal {page_index}")
    if data["width"] != width or data["height"] != height:
        raise ValueError(f"width/height must exactly match authoritative image dimensions ({width}x{height})")
    if data["unit"] != "px":
        raise ValueError('unit must be "px"')
    if data["origin"] != "top-left":
        raise ValueError('origin must be "top-left"')
    if not isinstance(data["fields"], list):
        raise ValueError("fields must be a list")

    for i, field in enumerate(data["fields"]):
        if not isinstance(field, dict):
            raise ValueError(f"fields[{i}] must be an object")
        expected_field_keys = {"field_id", "type", "bbox"}
        if set(field.keys()) != expected_field_keys:
            raise ValueError(f"fields[{i}] keys must be exactly {sorted(expected_field_keys)}")

        if not isinstance(field["field_id"], str) or not field["field_id"].strip():
            raise ValueError(f"fields[{i}].field_id must be a non-empty string")
        if not isinstance(field["type"], str) or not field["type"].strip():
            raise ValueError(f"fields[{i}].type must be a non-empty string")

        bbox = field["bbox"]
        if not isinstance(bbox, dict):
            raise ValueError(f"fields[{i}].bbox must be an object")
        expected_bbox_keys = {"x", "y", "w", "h"}
        if set(bbox.keys()) != expected_bbox_keys:
            raise ValueError(f"fields[{i}].bbox keys must be exactly {sorted(expected_bbox_keys)}")

        x = bbox["x"]
        y = bbox["y"]
        w = bbox["w"]
        h = bbox["h"]
        if not all(isinstance(v, int) for v in (x, y, w, h)):
            raise ValueError(f"fields[{i}].bbox values must all be integers")
        if w <= 0 or h <= 0:
            raise ValueError(f"fields[{i}].bbox requires w > 0 and h > 0")
        if x < 0 or y < 0:
            raise ValueError(f"fields[{i}].bbox requires x >= 0 and y >= 0")
        if x + w > width or y + h > height:
            raise ValueError(f"fields[{i}].bbox exceeds image bounds ({width}x{height})")

    return data


def _call_openai_for_page(
    client: OpenAI,
    *,
    model: str,
    timeout_seconds: float,
    max_output_tokens: int,
    page_index: int,
    width: int,
    height: int,
    image_path: Path,
) -> dict[str, Any]:
    prompt = _prompt_for_dimensions(width, height)
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = client.responses.create(
        model=model,
        timeout=timeout_seconds,
        max_output_tokens=max_output_tokens,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"},
                ],
            }
        ],
        text={"format": {"type": "json_object"}},
    )
    raw_json = _extract_json_text(response)
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise GroundingJsonParseError(f"Invalid JSON from OpenAI model response: {exc}", raw_json) from exc
    return _validate_field_grounding_json(parsed, page_index=page_index, width=width, height=height)


def _call_anthropic_for_page(
    client: Anthropic,
    *,
    model: str,
    timeout_seconds: float,
    max_tokens: int,
    page_index: int,
    width: int,
    height: int,
    image_path: Path,
) -> dict[str, Any]:
    prompt = _prompt_for_dimensions(width, height)
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        timeout=timeout_seconds,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                ],
            }
        ],
    )
    raw_json = _extract_json_text_anthropic(response)
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise GroundingJsonParseError(f"Invalid JSON from Anthropic model response: {exc}", raw_json) from exc
    return _validate_field_grounding_json(parsed, page_index=page_index, width=width, height=height)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _build_stamping_sample(field_dir: Path) -> tuple[str, dict[str, Any]]:
    values: dict[str, str] = {}
    for page_path in sorted(field_dir.glob("page_*.fields.json")):
        payload = _load_json(page_path)
        fields = payload.get("fields")
        if not isinstance(fields, list):
            continue
        for field in fields:
            if not isinstance(field, dict):
                continue
            field_id = field.get("field_id")
            if not isinstance(field_id, str) or not field_id:
                continue
            if field_id not in values:
                values[field_id] = field_id[:10]
    stamping_payload: dict[str, Any] = {
        "values": values,
        "require_all_values": False,
    }
    stamping_name = "stamping.json"
    (field_dir / stamping_name).write_text(json.dumps(stamping_payload, indent=2) + "\n", encoding="utf-8")
    return stamping_name, stamping_payload


def _create_provider_client(
    *,
    provider: str,
    openai_api_key: str,
    anthropic_api_key: str,
) -> Any:
    if provider == "openai":
        if not openai_api_key.strip():
            raise ValueError("FORMIQO_OPENAI_API_KEY is missing for provider=openai.")
        return OpenAI(api_key=openai_api_key)
    if provider == "anthropic":
        if not anthropic_api_key.strip():
            raise ValueError("FORMIQO_ANTHROPIC_API_KEY is missing for provider=anthropic.")
        return Anthropic(api_key=anthropic_api_key)
    raise ValueError(f"Unsupported provider: {provider}")


def _call_openai_repair_json(
    client: OpenAI,
    *,
    model: str,
    timeout_seconds: float,
    max_output_tokens: int,
    page_index: int,
    width: int,
    height: int,
    broken_json: str,
) -> dict[str, Any]:
    prompt = _repair_prompt_for_dimensions(
        page_index=page_index,
        width=width,
        height=height,
        broken_json=broken_json,
    )
    response = client.responses.create(
        model=model,
        timeout=timeout_seconds,
        max_output_tokens=max_output_tokens,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        text={"format": {"type": "json_object"}},
    )
    raw_json = _extract_json_text(response)
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise GroundingJsonParseError(f"Invalid JSON from OpenAI repair response: {exc}", raw_json) from exc
    return _validate_field_grounding_json(parsed, page_index=page_index, width=width, height=height)


def _call_anthropic_repair_json(
    client: Anthropic,
    *,
    model: str,
    timeout_seconds: float,
    max_tokens: int,
    page_index: int,
    width: int,
    height: int,
    broken_json: str,
) -> dict[str, Any]:
    prompt = _repair_prompt_for_dimensions(
        page_index=page_index,
        width=width,
        height=height,
        broken_json=broken_json,
    )
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        timeout=timeout_seconds,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    raw_json = _extract_json_text_anthropic(response)
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise GroundingJsonParseError(f"Invalid JSON from Anthropic repair response: {exc}", raw_json) from exc
    return _validate_field_grounding_json(parsed, page_index=page_index, width=width, height=height)


def _call_provider_for_page(
    *,
    provider: str,
    client: Any,
    model: str,
    timeout_seconds: float,
    openai_max_output_tokens: int,
    anthropic_max_tokens: int,
    page_index: int,
    width: int,
    height: int,
    image_path: Path,
) -> dict[str, Any]:
    if provider == "openai":
        return _call_openai_for_page(
            client,
            model=model,
            timeout_seconds=timeout_seconds,
            max_output_tokens=openai_max_output_tokens,
            page_index=page_index,
            width=width,
            height=height,
            image_path=image_path,
        )
    if provider == "anthropic":
        return _call_anthropic_for_page(
            client,
            model=model,
            timeout_seconds=timeout_seconds,
            max_tokens=anthropic_max_tokens,
            page_index=page_index,
            width=width,
            height=height,
            image_path=image_path,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def _repair_with_provider(
    *,
    provider: str,
    openai_client: OpenAI | None,
    provider_client: Any,
    model: str,
    openai_timeout_seconds: float,
    anthropic_timeout_seconds: float,
    openai_max_output_tokens: int,
    anthropic_max_tokens: int,
    page_index: int,
    width: int,
    height: int,
    broken_json: str,
) -> dict[str, Any]:
    if openai_client is not None:
        return _call_openai_repair_json(
            openai_client,
            model="gpt-5.5",
            timeout_seconds=openai_timeout_seconds,
            max_output_tokens=openai_max_output_tokens,
            page_index=page_index,
            width=width,
            height=height,
            broken_json=broken_json,
        )
    if provider == "openai":
        return _call_openai_repair_json(
            provider_client,
            model=model,
            timeout_seconds=openai_timeout_seconds,
            max_output_tokens=openai_max_output_tokens,
            page_index=page_index,
            width=width,
            height=height,
            broken_json=broken_json,
        )
    return _call_anthropic_repair_json(
        provider_client,
        model=model,
        timeout_seconds=anthropic_timeout_seconds,
        max_tokens=anthropic_max_tokens,
        page_index=page_index,
        width=width,
        height=height,
        broken_json=broken_json,
    )


def run_field_grounding_for_job(
    *,
    job_id: str,
    output_dir: Path,
    provider: str,
    model: str,
    page_index: int | None = None,
    openai_api_key: str,
    openai_timeout_seconds: float,
    openai_max_output_tokens: int,
    anthropic_api_key: str,
    anthropic_timeout_seconds: float,
    anthropic_max_tokens: int,
) -> dict[str, Any]:
    """Ground all converted page images for a completed conversion job."""
    provider = provider.strip().lower()
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider '{provider}'. Supported: {sorted(_SUPPORTED_PROVIDERS)}")
    if not model.strip():
        raise ValueError("model must be provided.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    pages = _discover_page_images(output_dir)
    if page_index is not None:
        pages = [(idx, path) for idx, path in pages if idx == page_index]
        if not pages:
            raise ValueError(f"page_index {page_index} not found in converted_images")
    field_dir = output_dir / "field_grounding"
    field_dir.mkdir(parents=True, exist_ok=True)

    client = _create_provider_client(
        provider=provider,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
    )
    repair_openai_client: OpenAI | None = None
    if openai_api_key.strip():
        repair_openai_client = OpenAI(api_key=openai_api_key)
    timeout_seconds = openai_timeout_seconds if provider == "openai" else anthropic_timeout_seconds

    page_results: list[dict[str, Any]] = []
    output_files: list[str] = []

    for page_index, image_path in pages:
        image_rel = str(image_path.relative_to(output_dir)).replace("\\", "/")
        parse_errors: list[GroundingJsonParseError] = []
        try:
            width, height = _read_png_dimensions(image_path)
            validated: dict[str, Any] | None = None
            resolution = "direct"
            for _attempt in range(_MAX_PARSE_RETRIES + 1):
                try:
                    validated = _call_provider_for_page(
                        provider=provider,
                        client=client,
                        model=model,
                        timeout_seconds=timeout_seconds,
                        openai_max_output_tokens=openai_max_output_tokens,
                        anthropic_max_tokens=anthropic_max_tokens,
                        page_index=page_index,
                        width=width,
                        height=height,
                        image_path=image_path,
                    )
                    resolution = "direct" if _attempt == 0 else "retry"
                    break
                except GroundingJsonParseError as exc:
                    parse_errors.append(exc)
            if validated is None:
                repair_source = parse_errors[-1]
                validated = _repair_with_provider(
                    provider=provider,
                    openai_client=repair_openai_client,
                    provider_client=client,
                    model=model,
                    openai_timeout_seconds=openai_timeout_seconds,
                    anthropic_timeout_seconds=anthropic_timeout_seconds,
                    openai_max_output_tokens=openai_max_output_tokens,
                    anthropic_max_tokens=anthropic_max_tokens,
                    page_index=page_index,
                    width=width,
                    height=height,
                    broken_json=repair_source.raw_text,
                )
                resolution = "repair"
            validated["provider"] = provider
            validated["model"] = model
            validated["run_id"] = run_id
            out_name = f"page_{page_index + 1:04d}.fields.json"
            out_path = field_dir / out_name
            out_path.write_text(json.dumps(validated, indent=2) + "\n", encoding="utf-8")
            out_rel = f"field_grounding/{out_name}"
            output_files.append(out_rel)
            page_results.append(
                {
                    "page_index": page_index,
                    "image_path": image_rel,
                    "status": "succeeded",
                    "resolution": resolution,
                    "output_file": out_rel,
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep POC error handling simple
            raw_output_rel: str | None = None
            if parse_errors:
                raw_name = f"page_{page_index + 1:04d}.failed.raw.txt"
                raw_path = field_dir / raw_name
                raw_path.write_text(parse_errors[-1].raw_text, encoding="utf-8")
                raw_output_rel = f"field_grounding/{raw_name}"
                output_files.append(raw_output_rel)
            elif isinstance(exc, GroundingJsonParseError):
                raw_name = f"page_{page_index + 1:04d}.failed.raw.txt"
                raw_path = field_dir / raw_name
                raw_path.write_text(exc.raw_text, encoding="utf-8")
                raw_output_rel = f"field_grounding/{raw_name}"
                output_files.append(raw_output_rel)
            page_results.append(
                {
                    "page_index": page_index,
                    "image_path": image_rel,
                    "status": "failed",
                    "error": str(exc),
                    **({"raw_model_output_file": raw_output_rel} if raw_output_rel else {}),
                }
            )

    succeeded_count = sum(1 for p in page_results if p["status"] == "succeeded")
    failed_count = len(page_results) - succeeded_count
    stamping_name, _ = _build_stamping_sample(field_dir)
    stamping_rel = f"field_grounding/{stamping_name}"
    output_files.append(stamping_rel)

    manifest_rel = "field_grounding/manifest.json"
    manifest = {
        "job_id": job_id,
        "provider": provider,
        "model": model,
        "run_id": run_id,
        "run_dir": "field_grounding",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(page_results),
        "output_dir": "field_grounding",
        "files": output_files,
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "stamping_sample_path": stamping_rel,
    }
    (field_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "job_id": job_id,
        "provider": provider,
        "model": model,
        "run_id": run_id,
        "run_dir": "field_grounding",
        "page_count": len(page_results),
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "output_dir": "field_grounding",
        "manifest_path": manifest_rel,
        "stamping_sample_path": stamping_rel,
        "pages": page_results,
    }
