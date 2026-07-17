# Grounding prompt changelog

Prompt assets under `prompts/` are versioned here. Bump this file whenever the model-facing
contract changes; `tests/test_grounding_prompt.py` guards the invariants.

## v3 — 2026-07-17 (E5 vision QA judge)

- Added `grounding_qa_judge.md`: the system/developer prompt for the E5 vision QA placement
  judge. The judge reviews per-field zoomed crops of the stamped preview and returns a
  per-field verdict (`ok | shift | unsure`) with a bounded pixel `dx/dy` correction in
  full-page top-left coordinate space.
- Judge output shape is enforced by structured provider outputs (OpenAI strict `json_schema`
  `field_verdicts`, Anthropic tool-use `emit_field_verdicts`); see `grounding_schema.py`.
- The judge defaults to the provider that differs from the grounder (correlated-blind-spot
  avoidance); model/provider come from `grounding_qa_*` config, never hardcoded keys.

## v2 — 2026-07-13 (E4 anchor-first grounding)

- Introduced the **anchor-first contract**: each field may carry an `anchor` object
  (`kind`: `cell | line_anchor | label_anchor | none`) so bounding boxes are computed
  deterministically in `app/services/form_geometry.py` instead of trusting raw pixel output.
- Added optional `label_anchors_json` attachment (printed label text + pixel bbox) extracted
  from digital PDFs via PyMuPDF, so the model can reference labels by exact text.
- Noted that the highlighted image may include a labeled coordinate grid overlay
  (`prompts/grounding_grid_overlay.md`) to improve pixel-fallback accuracy.
- `bbox` remains required as a best-estimate fallback; the anchor is authoritative.
- Structured provider outputs now enforce the shape (OpenAI strict `json_schema`,
  Anthropic tool-use `emit_grounded_fields`); see `app/services/grounding_schema.py`.

## v1 — 2026-07-12 (initial)

- Hybrid visual + line-map grounding. Model emitted pixel bboxes with `evidence.line_ids`.
- Compact-JSON retry with regex fence-stripping on parse failure.
