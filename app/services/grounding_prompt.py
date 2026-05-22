"""Prompt templates and response adaptation for hybrid line-map semantic grounding."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROMPT_DIR = _REPO_ROOT / "prompts"

_COMPACT_JSON_SUFFIX = (
    "\nIMPORTANT: Return compact JSON only. Omit long evidence labels. "
    "Include at most 40 fields. No markdown fences.\n"
)

_ATTACHMENT_MANIFEST = """Attachments for this page (in order):
1. highlighted_line_image — PNG with line overlay
2. line_detection_json — slim line index (line_id, orientation, bbox per line)
3. page_metadata_json — authoritative page_index, width, height, unit, origin
"""


def _prompt_dir() -> Path:
    raw = os.environ.get("FORMIQO_GROUNDING_PROMPT_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_PROMPT_DIR


def configure_prompt_dir(path: Path) -> None:
    """Set prompt directory (also respects FORMIQO_GROUNDING_PROMPT_DIR env)."""
    os.environ["FORMIQO_GROUNDING_PROMPT_DIR"] = str(path.expanduser().resolve())
    _load_prompt_file.cache_clear()


@lru_cache(maxsize=8)
def _load_prompt_file(name: str) -> str:
    path = _prompt_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"Grounding prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_grounding_system_prompt() -> str:
    return _load_prompt_file("grounding_system.md")


def load_grounding_developer_prompt() -> str:
    return _load_prompt_file("grounding_developer.md")


def load_grounding_user_prompt() -> str:
    return _load_prompt_file("grounding_user.md")


def _compact_json(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def slim_line_detection_for_prompt(detected_lines: dict[str, Any]) -> dict[str, Any]:
    """
    Reduce detected_lines to what the LLM needs: page bounds plus line_id, orientation, bbox.
    Full detected_lines.json on disk is unchanged for geometry and validation.
    """
    image = detected_lines.get("image")
    out: dict[str, Any] = {"lines": []}
    if isinstance(image, dict):
        width = image.get("width")
        height = image.get("height")
        if isinstance(width, int) and isinstance(height, int):
            out["width"] = width
            out["height"] = height

    raw_lines = detected_lines.get("lines")
    if not isinstance(raw_lines, list):
        return out

    slim_lines: list[dict[str, Any]] = []
    for ln in raw_lines:
        if not isinstance(ln, dict):
            continue
        line_id = ln.get("line_id")
        bbox = ln.get("bbox")
        if not isinstance(line_id, str) or not line_id.strip():
            continue
        if not isinstance(bbox, dict):
            continue
        try:
            bb = {k: int(bbox[k]) for k in ("x", "y", "w", "h")}
        except (KeyError, TypeError, ValueError):
            continue
        entry: dict[str, Any] = {
            "line_id": line_id,
            "bbox": bb,
        }
        orientation = ln.get("orientation")
        if isinstance(orientation, str) and orientation:
            entry["orientation"] = orientation
        slim_lines.append(entry)

    out["lines"] = slim_lines
    return out


def line_detection_payload_for_prompt(
    detected_lines: dict[str, Any],
    *,
    slim: bool = True,
) -> dict[str, Any]:
    if slim:
        return slim_line_detection_for_prompt(detected_lines)
    return detected_lines


def build_attachment_manifest(*, include_manifest: bool = True) -> str:
    if not include_manifest:
        return ""
    return _ATTACHMENT_MANIFEST


def build_user_text_bundle(
    *,
    detected_lines: dict[str, Any],
    page_manifest: dict[str, Any],
    compact_json: bool = False,
    include_attachment_manifest: bool = True,
    slim_line_detection: bool = True,
) -> str:
    parts: list[str] = []
    manifest = build_attachment_manifest(include_manifest=include_attachment_manifest)
    if manifest:
        parts.append(manifest)
    parts.append(load_grounding_user_prompt())
    parts.append("highlighted_line_image:\n[attached PNG below]")
    line_payload = line_detection_payload_for_prompt(detected_lines, slim=slim_line_detection)
    parts.append("line_detection_json:\n" + _compact_json(line_payload))
    parts.append("page_metadata_json:\n" + _compact_json(page_manifest))
    if compact_json:
        parts.append(_COMPACT_JSON_SUFFIX)
    return "\n\n".join(parts)


def _image_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return "image/png" if suffix == ".png" else "image/jpeg"


def _image_base64(path: Path) -> str:
    import base64

    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _openai_image_part(path: Path) -> dict[str, Any]:
    mime = _image_media_type(path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{_image_base64(path)}"},
    }


def _anthropic_image_block(path: Path) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": _image_media_type(path),
            "data": _image_base64(path),
        },
    }


def _openai_user_content(
    *,
    highlighted_png: Path,
    detected_lines: dict[str, Any],
    page_manifest: dict[str, Any],
    compact_json: bool = False,
    include_attachment_manifest: bool = True,
    slim_line_detection: bool = True,
) -> list[dict[str, Any]]:
    user_text = build_user_text_bundle(
        detected_lines=detected_lines,
        page_manifest=page_manifest,
        compact_json=compact_json,
        include_attachment_manifest=include_attachment_manifest,
        slim_line_detection=slim_line_detection,
    )
    return [
        {"type": "text", "text": user_text},
        _openai_image_part(highlighted_png),
    ]


def build_openai_messages(
    *,
    highlighted_png: Path,
    detected_lines: dict[str, Any],
    page_manifest: dict[str, Any],
    compact_json: bool = False,
    include_attachment_manifest: bool = True,
    slim_line_detection: bool = True,
) -> list[dict[str, Any]]:
    """Chat Completions messages with system + developer + multimodal user."""
    return [
        {"role": "system", "content": load_grounding_system_prompt()},
        {"role": "developer", "content": load_grounding_developer_prompt()},
        {
            "role": "user",
            "content": _openai_user_content(
                highlighted_png=highlighted_png,
                detected_lines=detected_lines,
                page_manifest=page_manifest,
                compact_json=compact_json,
                include_attachment_manifest=include_attachment_manifest,
                slim_line_detection=slim_line_detection,
            ),
        },
    ]


def build_anthropic_messages(
    *,
    highlighted_png: Path,
    detected_lines: dict[str, Any],
    page_manifest: dict[str, Any],
    compact_json: bool = False,
    include_attachment_manifest: bool = True,
    slim_line_detection: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Return ``(system_text, user_content_blocks)`` for Anthropic Messages API."""
    system_text = (
        load_grounding_system_prompt()
        + "\n\n"
        + load_grounding_developer_prompt()
    )
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": build_user_text_bundle(
                detected_lines=detected_lines,
                page_manifest=page_manifest,
                compact_json=compact_json,
                include_attachment_manifest=include_attachment_manifest,
                slim_line_detection=slim_line_detection,
            ),
        },
        _anthropic_image_block(highlighted_png),
    ]
    return system_text, user_content


def _field_surface_for_llm_type(llm_type: str, existing: str | None) -> str:
    if isinstance(existing, str) and existing.strip() and existing != "unknown":
        return existing
    if llm_type == "table_cell":
        return "solid_box"
    if llm_type in ("checkbox", "radio"):
        return "checkbox" if llm_type == "checkbox" else "radio_circle"
    if llm_type == "signature":
        return "signature_line"
    if llm_type == "date":
        return "underline"
    return "unknown"


def adapt_grounding_field(field: dict[str, Any]) -> dict[str, Any]:
    """Map LLM grounding field shape to internal validator/stamping shape."""
    from app.grounding_field_types import normalize_llm_field_type

    out = dict(field)
    raw_type = field.get("type")
    llm_type = raw_type if isinstance(raw_type, str) else "unknown"
    internal_type = normalize_llm_field_type(llm_type)
    out["type"] = internal_type
    if llm_type != internal_type:
        out["grounding_type"] = llm_type

    evidence = field.get("evidence")
    if isinstance(evidence, dict):
        label = evidence.get("label")
        if isinstance(label, str) and label.strip():
            out["label"] = label.strip()
            out["nearby_label_text"] = label.strip()
        line_ids = evidence.get("line_ids")
        if isinstance(line_ids, list):
            out["supporting_lines"] = [lid for lid in line_ids if isinstance(lid, str) and lid]

    existing_surface = field.get("field_surface")
    surface_val = existing_surface if isinstance(existing_surface, str) else None
    out["field_surface"] = _field_surface_for_llm_type(llm_type, surface_val)

    return out


def adapt_grounding_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize top-level LLM JSON to internal grounding payload."""
    out = dict(raw)
    fields = raw.get("fields")
    if isinstance(fields, list):
        out["fields"] = [
            adapt_grounding_field(f) if isinstance(f, dict) else f for f in fields
        ]
    if out.get("unit") is None:
        out["unit"] = "px"
    if out.get("origin") is None:
        out["origin"] = "top-left"
    return out
