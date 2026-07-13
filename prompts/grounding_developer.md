Ground fields using a hybrid visual + deterministic structure approach.

Use page_NNN.json and detected lines.json as the authoritative coordinate space.

Priority order:
1. Use the attached page metadata JSON as the source of truth for page_index, width, height, unit, and origin.
2. Use the attached line detection JSON as authoritative structural evidence. It is a slim index only: each line has line_id, orientation, and bbox (no detector metadata or duplicate endpoints).
3. Use the label anchors JSON (when attached) to anchor fields to printed labels by exact text.
4. Use the highlighted line image for semantic understanding, labels, nearby context, and visual confirmation.

Anchor-first contract (this is the accuracy step-change):
- For every field, prefer an `anchor` over a hand-drawn pixel bbox.
- `anchor.kind` is one of: "cell", "line_anchor", "label_anchor", "none".
  - "cell": set `anchor.line_ids` to the horizontal + vertical line_ids that bound the cell.
  - "line_anchor": set `anchor.line_ids` to the horizontal writing line_id the field sits on.
  - "label_anchor": set `anchor.label` to the exact printed label text and `anchor.relation`
    to where the field is written relative to that label.
  - "none": no structural anchor applies; provide only a pixel bbox estimate.
- Reference only line_id values present in the line detection JSON.
- Always also provide `bbox` as a best pixel estimate (used as a fallback).

Rules:
- Return only writable/input regions.
- Exclude labels, captions, instructions, headers, footers, logos, and decorative elements.
- For text fields, bbox should cover the writable area, not the label.
- For checkbox/radio fields, bbox should tightly cover the actual box/circle only.
- For table cells, bbox should cover the writable interior of the cell.
- For signature/date fields, bbox should cover the writable signing/date area.
- Prefer conservative boxes over oversized boxes.
- Keep aligned fields geometrically consistent.
- Use integer pixel coordinates only.

Important:
- Do not blindly box every detected line.
- Do not treat every highlighted line as a field.
- The overlay is only a guide; infer actual writable regions from document semantics and structure.
- If a field cannot be confidently grounded, omit it rather than guessing.

Output schema:

{
  "page_index": 0,
  "width": 0,
  "height": 0,
  "unit": "px",
  "origin": "top-left",
  "fields": [
    {
      "field_id": "stable_semantic_name",
      "type": "text|textarea|checkbox|radio|signature|date|table_cell",
      "bbox": { "x": 0, "y": 0, "w": 0, "h": 0 },
      "confidence": 0.0,
      "anchor": {
        "kind": "cell|line_anchor|label_anchor|none",
        "line_ids": [],
        "label": "exact label text when kind=label_anchor",
        "relation": "right_of|below|left_of|above|on_line"
      },
      "evidence": {
        "label": "nearby visible label or empty string",
        "line_ids": []
      }
    }
  ]
}

Do not return markdown.
Do not explain anything outside the structured output.
