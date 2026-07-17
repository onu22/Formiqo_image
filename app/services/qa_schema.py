"""Structured-output contract for the E5 QA judge (OpenAI json_schema + Anthropic tool-use).

The judge inspects a single per-field zoom crop and returns a narrow positional verdict.
Provider-native structured outputs make parse failures essentially impossible, matching the
E4 grounding approach in ``grounding_schema.py``.
"""

from __future__ import annotations

from typing import Any

_VERDICTS: tuple[str, ...] = ("ok", "adjust")

QA_JUDGE_TOOL_NAME = "emit_field_verdict"
QA_JUDGE_SCHEMA_NAME = "field_placement_verdict"


def qa_verdict_schema(*, strict: bool) -> dict[str, Any]:
    """JSON schema for a single field placement verdict."""
    properties: dict[str, Any] = {
        "field_id": {"type": "string"},
        "verdict": {"type": "string", "enum": list(_VERDICTS)},
        "dx": {"type": "integer"},
        "dy": {"type": "integer"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"} if not strict else {"type": ["string", "null"]},
    }
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if strict:
        schema["required"] = list(properties.keys())
        schema["additionalProperties"] = False
    else:
        schema["required"] = ["field_id", "verdict"]
    return schema


def openai_qa_response_format() -> dict[str, Any]:
    """``response_format`` for OpenAI Chat Completions strict json_schema mode."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": QA_JUDGE_SCHEMA_NAME,
            "strict": True,
            "schema": qa_verdict_schema(strict=True),
        },
    }


def anthropic_qa_tool() -> dict[str, Any]:
    """Anthropic tool definition; the judge returns its verdict via forced tool-use."""
    return {
        "name": QA_JUDGE_TOOL_NAME,
        "description": (
            "Return the placement verdict for the single form field shown in the attached "
            "zoom crop. Judge position only; report page-pixel dx/dy corrections."
        ),
        "input_schema": qa_verdict_schema(strict=False),
    }


def anthropic_qa_tool_choice() -> dict[str, Any]:
    return {"type": "tool", "name": QA_JUDGE_TOOL_NAME}
