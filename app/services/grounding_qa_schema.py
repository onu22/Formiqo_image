"""Structured-output contracts for the E5 vision QA judge.

The judge inspects one stamped field crop and returns a narrow verdict: is the value
correctly positioned on its writing line / in its cell, and if not, the direction and rough
pixel magnitude of the needed shift. Provider-native structured outputs (OpenAI strict
``json_schema`` / Anthropic tool-use) keep the response machine-parseable.
"""

from __future__ import annotations

from typing import Any

QA_TOOL_NAME = "emit_field_verdict"
QA_SCHEMA_NAME = "field_verdict"

# Judge verdicts. ``ok`` = correctly placed; ``shift`` = translate by (dx, dy) pixels.
QA_VERDICTS: tuple[str, ...] = ("ok", "shift")


def qa_verdict_schema(*, strict: bool) -> dict[str, Any]:
    """JSON schema for a single-field judge verdict.

    ``dx``/``dy`` are pixel translations in top-left image space: positive ``dx`` moves the
    field right, positive ``dy`` moves it down. Magnitudes are advisory; the loop clamps
    them to ``grounding_qa_max_bbox_delta_px`` per axis per iteration.
    """
    properties: dict[str, Any] = {
        "verdict": {"type": "string", "enum": list(QA_VERDICTS)},
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
        schema["required"] = ["verdict", "dx", "dy"]
    return schema


def openai_qa_response_format() -> dict[str, Any]:
    """``response_format`` for OpenAI Chat Completions strict json_schema mode."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": QA_SCHEMA_NAME,
            "strict": True,
            "schema": qa_verdict_schema(strict=True),
        },
    }


def anthropic_qa_tool() -> dict[str, Any]:
    """Anthropic tool definition; the judge returns its verdict via forced tool-use."""
    return {
        "name": QA_TOOL_NAME,
        "description": (
            "Report whether the stamped value in the crop sits correctly on its writing "
            "line / in its cell. If not, give the pixel shift (dx right+, dy down+) to fix it."
        ),
        "input_schema": qa_verdict_schema(strict=False),
    }


def anthropic_qa_tool_choice() -> dict[str, Any]:
    return {"type": "tool", "name": QA_TOOL_NAME}
