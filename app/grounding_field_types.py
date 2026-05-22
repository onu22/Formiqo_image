"""Canonical field types for LLM grounding and stamping."""

from __future__ import annotations

ALLOWED_AI_GROUNDING_FIELD_TYPES: frozenset[str] = frozenset(
    {
        "text",
        "multiline_text",
        "checkbox",
        "radio",
        "date",
        "signature",
        "character_boxes",
        "numeric",
        "unknown",
        "dropdown",
        "list_box",
    }
)

ALLOWED_FIELD_SURFACES: frozenset[str] = frozenset(
    {
        "solid_box",
        "dotted_line",
        "underline",
        "checkbox",
        "radio_circle",
        "character_boxes",
        "open_area",
        "signature_line",
        "unknown",
    }
)

ALLOWED_STAMP_FIELD_TYPES: frozenset[str] = frozenset(
    {
        "text",
        "multiline_text",
        "checkbox",
        "radio",
        "dropdown",
        "list_box",
    }
)

TEXT_LIKE_GROUNDING_TYPES: frozenset[str] = frozenset(
    {"text", "multiline_text", "dropdown", "list_box", "date", "signature", "character_boxes", "numeric", "unknown"}
)

TOGGLE_GROUNDING_TYPES: frozenset[str] = frozenset({"checkbox", "radio"})

# Types emitted by the LLM grounding prompt (mapped before validation).
LLM_GROUNDING_TYPE_ALIASES: dict[str, str] = {
    "textarea": "multiline_text",
    "table_cell": "text",
}

_CHECK_MARK_CHARS = frozenset({"\u2713", "\u2714"})


def normalize_llm_field_type(field_type: str) -> str:
    """Map LLM field type strings to canonical grounding types."""
    if field_type in ALLOWED_AI_GROUNDING_FIELD_TYPES:
        return field_type
    return LLM_GROUNDING_TYPE_ALIASES.get(field_type, field_type)


def allowed_ai_types_sorted_join(*, sep: str = ", ") -> str:
    return sep.join(sorted(ALLOWED_AI_GROUNDING_FIELD_TYPES - {"dropdown", "list_box"}))


def allowed_surfaces_sorted_join(*, sep: str = ", ") -> str:
    return sep.join(sorted(ALLOWED_FIELD_SURFACES))


def stamps_as_text(field_type: str) -> bool:
    return field_type in TEXT_LIKE_GROUNDING_TYPES or stamping_type_for_field(field_type) == "text"


def stamps_as_toggle(field_type: str) -> bool:
    return field_type in TOGGLE_GROUNDING_TYPES


def is_supported_ai_field_type(field_type: object) -> bool:
    return isinstance(field_type, str) and field_type in ALLOWED_AI_GROUNDING_FIELD_TYPES


def is_supported_stamp_field_type(field_type: object) -> bool:
    return isinstance(field_type, str) and field_type in ALLOWED_STAMP_FIELD_TYPES


def stamping_type_for_field(field_type: str) -> str:
    """Map AI grounding types to types accepted by image/pdf stamping."""
    if field_type in TOGGLE_GROUNDING_TYPES:
        return field_type
    if field_type in {"text", "multiline_text", "dropdown", "list_box"}:
        return field_type
    return "text"


def is_toggle_value_truthy(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if lowered in {"1", "true", "yes", "y", "x"}:
        return True
    return stripped in _CHECK_MARK_CHARS


# Back-compat aliases used by stamping modules
ALLOWED_GROUNDING_FIELD_TYPES = ALLOWED_STAMP_FIELD_TYPES


def is_supported_grounding_field_type(field_type: object) -> bool:
    return is_supported_stamp_field_type(field_type)


def allowed_types_sorted_join(*, sep: str = ", ") -> str:
    return sep.join(sorted(ALLOWED_STAMP_FIELD_TYPES))
