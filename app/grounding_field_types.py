"""Canonical field `type` values for LLM grounding and stamping."""

from __future__ import annotations

ALLOWED_GROUNDING_FIELD_TYPES: frozenset[str] = frozenset(
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
    {"text", "multiline_text", "dropdown", "list_box"}
)

TOGGLE_GROUNDING_TYPES: frozenset[str] = frozenset({"checkbox", "radio"})

_CHECK_MARK_CHARS = frozenset({"\u2713", "\u2714"})


def allowed_types_sorted_join(*, sep: str = ", ") -> str:
    """Comma-separated sorted literals for prompts and errors."""
    return sep.join(sorted(ALLOWED_GROUNDING_FIELD_TYPES))


def stamps_as_text(field_type: str) -> bool:
    return field_type in TEXT_LIKE_GROUNDING_TYPES


def stamps_as_toggle(field_type: str) -> bool:
    return field_type in TOGGLE_GROUNDING_TYPES


def is_supported_grounding_field_type(field_type: object) -> bool:
    return isinstance(field_type, str) and field_type in ALLOWED_GROUNDING_FIELD_TYPES


def is_toggle_value_truthy(value: str) -> bool:
    """Whether stamping should draw a mark for checkbox/radio controls."""
    stripped = value.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if lowered in {"1", "true", "yes", "y", "x"}:
        return True
    return stripped in _CHECK_MARK_CHARS
