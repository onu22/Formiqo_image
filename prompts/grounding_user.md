Ground all writable form fields on this page.

Use the attachment manifest to identify each input by role.

Use the page metadata JSON for the exact coordinate space.
Use the line detection JSON (slim line index: line_id, orientation, bbox per line) to improve bbox accuracy.
Use the label anchors JSON (when present) to anchor fields to printed labels by exact text.
Use the highlighted line image to understand labels and field intent.

Prefer anchors over pixel guessing: reference the bounding line_ids of a cell, the horizontal
line a field is written on, or the exact label text a field belongs to. Always also include a
best-estimate pixel bbox as a fallback.

Return structured output only.

Required output:
{
  "page_index": "<from page metadata JSON>",
  "width": "<from page metadata JSON>",
  "height": "<from page metadata JSON>",
  "unit": "px",
  "origin": "top-left",
  "fields": [
    {
      "field_id": "<stable snake_case semantic name>",
      "type": "<field type>",
      "bbox": {
        "x": <integer>,
        "y": <integer>,
        "w": <integer>,
        "h": <integer>
      },
      "confidence": <number between 0 and 1>,
      "anchor": {
        "kind": "cell|line_anchor|label_anchor|none",
        "line_ids": ["<ids from line detection JSON>"],
        "label": "<exact label text when kind=label_anchor>",
        "relation": "right_of|below|left_of|above|on_line"
      },
      "evidence": {
        "label": "<nearby label text if visible>",
        "line_ids": ["<ids from line detection JSON if available>"]
      }
    }
  ]
}

Constraints:
- Do not include labels inside bboxes.
- Do not return decorative lines.
- Do not return section dividers unless they are writable lines.
- Do not invent fields that are not visible.
- Do not include explanations.
- Do not include markdown.
