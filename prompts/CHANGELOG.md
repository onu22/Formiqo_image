# Grounding prompt changelog

Prompt assets under `prompts/` are versioned here. Bump this file whenever the model-facing
contract changes; `tests/test_grounding_prompt.py` guards the invariants.

## v3 — 2026-07-17 (E5 vision QA refinement loop)

- Added the **placement judge** prompts (`grounding_qa_system.md`, `grounding_qa_user.md`):
  a narrow, per-field verdict (`ok | shift` + `dx`/`dy` pixel translation) used by the
  `POST /jobs/{id}/refine-grounding` closed loop in `app/services/grounding_qa.py`.
- Judge crops are ~3x zoomed regions around each stamped field (a debug box marks the current
  bbox); the judge references the full-page pixel coordinate system so deltas map back
  deterministically. Applied shifts are bounded by `grounding_qa_max_bbox_delta_px` per axis
  per iteration, with consensus translation merging per the `grounding_qa_consensus_*` config.
- Judge structured outputs enforce the verdict shape (OpenAI strict `json_schema`,
  Anthropic tool-use `emit_field_verdict`); see `app/services/grounding_qa_schema.py`.
- The judge provider/model come from config (`grounding_qa_judge_*`); it defaults to a
  different provider than the grounder when a key is available, to avoid correlated blind
  spots.

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
