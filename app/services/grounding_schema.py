"""Structured-output contracts for grounding (OpenAI json_schema + Anthropic tool-use).

The model emits the anchor-first grounding payload described in ``prompts/``. These
schemas make the provider itself enforce the shape, replacing regex fence-stripping and
compact-JSON retries with a thin fallback.
"""

from __future__ import annotations

from typing import Any

# LLM-facing field type vocabulary (mapped to canonical types in grounding_field_types).
_LLM_FIELD_TYPES: tuple[str, ...] = (
    "text",
    "textarea",
    "multiline_text",
    "checkbox",
    "radio",
    "signature",
    "date",
    "numeric",
    "table_cell",
    "dropdown",
    "list_box",
    "unknown",
)

# Anchor kinds the model may reference; resolved deterministically in form_geometry.
_ANCHOR_KINDS: tuple[str, ...] = ("cell", "line_anchor", "label_anchor", "none")

# Where a field sits relative to its label anchor (used for label_anchor resolution).
_ANCHOR_RELATIONS: tuple[str, ...] = ("right_of", "below", "left_of", "above", "on_line")

GROUNDING_TOOL_NAME = "emit_grounded_fields"
GROUNDING_SCHEMA_NAME = "grounded_fields"


def _bbox_schema(*, strict: bool) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "w": {"type": "integer"},
            "h": {"type": "integer"},
        },
        "required": ["x", "y", "w", "h"],
    }
    if strict:
        schema["additionalProperties"] = False
    return schema


def _anchor_schema(*, strict: bool) -> dict[str, Any]:
    # Optional fields are expressed as nullable unions under strict mode so every property
    # can stay in ``required`` (OpenAI strict json_schema requirement).
    line_ids = {"type": "array", "items": {"type": "string"}}
    if strict:
        properties = {
            "kind": {"type": "string", "enum": list(_ANCHOR_KINDS)},
            "line_ids": line_ids,
            "label": {"type": ["string", "null"]},
            "relation": {"type": ["string", "null"], "enum": [*_ANCHOR_RELATIONS, None]},
        }
        return {
            "type": ["object", "null"],
            "properties": properties,
            "required": ["kind", "line_ids", "label", "relation"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(_ANCHOR_KINDS)},
            "line_ids": line_ids,
            "label": {"type": "string"},
            "relation": {"type": "string", "enum": list(_ANCHOR_RELATIONS)},
        },
        "required": ["kind"],
    }


def _evidence_schema(*, strict: bool) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "label": {"type": "string"} if not strict else {"type": ["string", "null"]},
            "line_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["label", "line_ids"] if strict else [],
    }
    if strict:
        schema["additionalProperties"] = False
    return schema


def _field_schema(*, strict: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "field_id": {"type": "string"},
        "type": {"type": "string", "enum": list(_LLM_FIELD_TYPES)},
        "bbox": _bbox_schema(strict=strict),
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "anchor": _anchor_schema(strict=strict),
        "evidence": _evidence_schema(strict=strict),
    }
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if strict:
        schema["required"] = list(properties.keys())
        schema["additionalProperties"] = False
    else:
        schema["required"] = ["field_id", "type", "bbox"]
    return schema


def grounding_response_schema(*, strict: bool) -> dict[str, Any]:
    """JSON schema for a single page grounding response."""
    properties: dict[str, Any] = {
        "page_index": {"type": "integer"},
        "width": {"type": "integer"},
        "height": {"type": "integer"},
        "unit": {"type": "string", "enum": ["px"]},
        "origin": {"type": "string", "enum": ["top-left"]},
        "fields": {"type": "array", "items": _field_schema(strict=strict)},
    }
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if strict:
        schema["required"] = list(properties.keys())
        schema["additionalProperties"] = False
    else:
        schema["required"] = ["page_index", "fields"]
    return schema


def openai_response_format() -> dict[str, Any]:
    """``response_format`` for OpenAI Chat Completions strict json_schema mode."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": GROUNDING_SCHEMA_NAME,
            "strict": True,
            "schema": grounding_response_schema(strict=True),
        },
    }


def anthropic_grounding_tool() -> dict[str, Any]:
    """Anthropic tool definition; the model returns fields via forced tool-use."""
    return {
        "name": GROUNDING_TOOL_NAME,
        "description": (
            "Return the grounded writable form fields for this page. Reference detected "
            "line_ids / label anchors so bounding boxes can be computed deterministically."
        ),
        "input_schema": grounding_response_schema(strict=False),
    }


def anthropic_tool_choice() -> dict[str, Any]:
    return {"type": "tool", "name": GROUNDING_TOOL_NAME}
