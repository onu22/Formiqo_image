"""Prompt templates for hybrid line-map semantic grounding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FORMIQO_GROUNDING_SYSTEM_PROMPT = """You are a production-grade Document Field Grounding Engine for Formiqo.

Your responsibility is to identify the precise writable regions of fields on structured forms using:
1. The clean rendered page image
2. Deterministic line-detection results
3. Page metadata/layout metadata

You are NOT performing OCR transcription.
You are NOT extracting business values.
You are ONLY grounding field locations.

The deterministic line-detection JSON is authoritative structural guidance produced by OpenCV and should be heavily trusted for:
- horizontal rules
- vertical rules
- table cells
- input boxes
- checkbox borders
- section dividers

Use the image primarily for semantic understanding and visual confirmation.

Your objective is to return highly accurate field bounding boxes aligned to the REAL writable areas on the page.

Coordinate System:
- Origin: top-left
- Unit: pixels
- Use EXACT page dimensions from the provided manifest JSON
- NEVER invent a different coordinate space
- NEVER normalize coordinates
- NEVER resize mentally
- All returned bbox values must map directly to the provided image

Do NOT explain reasoning.
Return JSON only."""

FORMIQO_GROUNDING_DEVELOPER_PROMPT = """You are working inside the Formiqo hybrid CV + LLM grounding pipeline.

The grounding stage receives exactly these page inputs:
1. clean converted page PNG
2. page manifest/layout metadata JSON
3. deterministic detected_lines.json from OpenCV

Do not expect or reference a highlighted line overlay image during grounding.
Do not return label bounding boxes. A field bbox must represent the safe writable/stamping area.

Valid field types:
- text
- multiline_text
- checkbox
- radio
- date
- signature
- character_boxes
- numeric
- unknown

Valid field surfaces:
- solid_box
- dotted_line
- underline
- checkbox
- radio_circle
- character_boxes
- open_area
- signature_line
- unknown

Rules:
1. Use exact page_index, width, and height from the manifest.
2. Coordinates must use unit px and origin top-left.
3. Treat detected_lines.json as authoritative structural evidence.
4. Use the clean page image for semantic understanding and visual confirmation.
5. Do not invent line coordinates.
6. Do not include logos, watermarks, stamps, graphics, headings, page numbers, instructions, or body paragraphs as fields.
7. Only create a field when there is a visible writable target:
   - blank box
   - checkbox or radio button
   - character boxes
   - dotted line
   - underline
   - signature line
   - clearly writable open area
8. Prefer smaller accurate writable regions over oversized boxes.
9. If a writable region is bounded by detected lines, target the white interior (not the stroke pixels). The pipeline snaps bboxes to detected cell interiors when the field center falls inside a cell.
10. When you know the enclosing lines, list their line_id values in supporting_lines (e.g. line_h_001, line_v_002).
11. If a writable region is bounded by detected lines, keep bbox inside the borders with small padding.
12. For checkboxes/radios, bbox should tightly cover only the square/circle, not the label text.
13. For character boxes, one logical run of boxes should become one field.
14. For table fields, place the field center in the target cell; bbox will align to that cell interior.
15. For multi-line regions, return the full writable region.
16. Maintain geometric consistency across aligned fields.
17. For dotted/signature lines, do not ignore them.
18. If uncertain, lower confidence and explain briefly in notes.

Return ONLY valid JSON using this schema:

{
  "page_index": 0,
  "width": 0,
  "height": 0,
  "unit": "px",
  "origin": "top-left",
  "fields": [
    {
      "field_id": "string_snake_case",
      "label": "Visible field label",
      "type": "text | multiline_text | checkbox | radio | date | signature | character_boxes | numeric | unknown",
      "field_surface": "solid_box | dotted_line | underline | checkbox | radio_circle | character_boxes | open_area | signature_line | unknown",
      "bbox": { "x": 0, "y": 0, "w": 0, "h": 0 },
      "section": "Visible section name if available",
      "nearby_label_text": "Text visually associated with this field",
      "supporting_lines": [],
      "confidence": 0.0,
      "notes": "Short explanation of why this bbox is the writable region"
    }
  ],
  "warnings": []
}

Field ID rules:
- Use stable snake_case names.
- Good: surname, forenames_in_full, date_of_birth, identity_number, representative_signature.
- Avoid generic names unless the label is unclear.

Before returning, verify:
- every bbox is within page bounds
- every bbox is writable space, not label text
- no bbox crosses unrelated grid lines (bounding lines of the target cell are OK)
- checkbox/radio bboxes are tight
- repeated fields have unique field_ids

Do not return markdown.
Do not explain anything outside the JSON."""

USER_PROMPT_TEMPLATE = """Ground all writable form fields visible on this page.

You are provided:
- page_index: {page_index}
1. A clean rendered page image: {original_png_path}
2. Page metadata defining the authoritative coordinate space: {page_manifest_json_path}
3. Deterministic line-detection results from OpenCV: {detected_lines_json_path}

Important:
Use the line-detection results as strong structural guidance, but visually verify against the page image.
Use integer pixel coordinates only.
Preserve exact image coordinate space.
Do not normalize coordinates.
Do not return markdown.
Do not explain anything.

Return only valid JSON."""

COMPACT_JSON_INSTRUCTION = (
    "\nIMPORTANT: Return compact JSON only. Omit long notes (max 80 characters each). "
    "Include at most 40 fields. No markdown fences.\n"
)


def build_user_prompt_text(
    *,
    page_index: int,
    original_png_path: str,
    detected_lines_json_path: str,
    page_manifest_json_path: str,
    geometry_summary: str | None = None,
    validator_errors: list[dict[str, Any]] | None = None,
    compact_json: bool = False,
) -> str:
    body = USER_PROMPT_TEMPLATE.format(
        page_index=page_index,
        original_png_path=original_png_path,
        detected_lines_json_path=detected_lines_json_path,
        page_manifest_json_path=page_manifest_json_path,
    )
    parts = [body]
    if geometry_summary:
        parts.append("\n" + geometry_summary)
    if validator_errors:
        parts.append("\nPrevious attempt failed validation. Fix bboxes only; do not invent line coordinates.\n")
        parts.append(json.dumps({"validator_errors": validator_errors}, indent=2))
    if compact_json:
        parts.append(COMPACT_JSON_INSTRUCTION)
    return "\n".join(parts)


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


def build_user_text_bundle(
    *,
    page_index: int,
    original_png: Path,
    detected_lines: dict[str, Any],
    page_manifest: dict[str, Any],
    rel_original: str,
    rel_lines_json: str,
    rel_page_manifest: str,
    geometry_summary: str | None = None,
    validator_errors: list[dict[str, Any]] | None = None,
    compact_json: bool = False,
) -> str:
    user_text = build_user_prompt_text(
        page_index=page_index,
        original_png_path=rel_original,
        detected_lines_json_path=rel_lines_json,
        page_manifest_json_path=rel_page_manifest,
        geometry_summary=geometry_summary,
        validator_errors=validator_errors,
        compact_json=compact_json,
    )
    user_text += "\n\ndetected_lines.json:\n" + json.dumps(detected_lines, indent=2)
    user_text += "\n\npage_manifest.json:\n" + json.dumps(page_manifest, indent=2)
    return user_text


def build_openai_messages(
    *,
    page_index: int,
    original_png: Path,
    detected_lines: dict[str, Any],
    page_manifest: dict[str, Any],
    rel_original: str,
    rel_lines_json: str,
    rel_page_manifest: str,
    geometry_summary: str | None = None,
    validator_errors: list[dict[str, Any]] | None = None,
    compact_json: bool = False,
) -> list[dict[str, Any]]:
    """Chat Completions messages with system + developer + multimodal user."""
    user_text = build_user_text_bundle(
        page_index=page_index,
        original_png=original_png,
        detected_lines=detected_lines,
        page_manifest=page_manifest,
        rel_original=rel_original,
        rel_lines_json=rel_lines_json,
        rel_page_manifest=rel_page_manifest,
        geometry_summary=geometry_summary,
        validator_errors=validator_errors,
        compact_json=compact_json,
    )

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": user_text},
        _openai_image_part(original_png),
    ]

    return [
        {"role": "system", "content": FORMIQO_GROUNDING_SYSTEM_PROMPT},
        {"role": "developer", "content": FORMIQO_GROUNDING_DEVELOPER_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_anthropic_messages(
    *,
    page_index: int,
    original_png: Path,
    detected_lines: dict[str, Any],
    page_manifest: dict[str, Any],
    rel_original: str,
    rel_lines_json: str,
    rel_page_manifest: str,
    geometry_summary: str | None = None,
    validator_errors: list[dict[str, Any]] | None = None,
    compact_json: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Return ``(system_text, user_content_blocks)`` for Anthropic Messages API.

    Developer instructions are folded into system (Anthropic has no developer role).
    """
    user_text = build_user_text_bundle(
        page_index=page_index,
        original_png=original_png,
        detected_lines=detected_lines,
        page_manifest=page_manifest,
        rel_original=rel_original,
        rel_lines_json=rel_lines_json,
        rel_page_manifest=rel_page_manifest,
        geometry_summary=geometry_summary,
        validator_errors=validator_errors,
        compact_json=compact_json,
    )
    system_text = (
        FORMIQO_GROUNDING_SYSTEM_PROMPT
        + "\n\n"
        + FORMIQO_GROUNDING_DEVELOPER_PROMPT
    )
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": user_text},
        _anthropic_image_block(original_png),
    ]
    return system_text, user_content
