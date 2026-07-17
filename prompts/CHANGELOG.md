# Grounding prompt changelog

Prompt assets under `prompts/` are versioned here. Bump this file whenever the model-facing
contract changes; `tests/test_grounding_prompt.py` and `tests/test_e5_refine.py` guard the
invariants.

## v3 — 2026-07-17 (E5 vision QA refinement judge)

- Added the **QA judge** prompt pair (`qa_judge_system.md`, `qa_judge_user.md`) for the
  closed refinement loop (`POST /jobs/{id}/refine-grounding`).
- The judge inspects a per-field **zoom crop** and returns a narrow structured verdict
  (`verdict: ok | adjust`, page-pixel `dx`/`dy`, `confidence`, `reason`); positional only.
- Corrections are applied as **bounded per-iteration deltas** (`grounding_qa_max_bbox_delta_px`)
  with optional consensus-translation merging; enforced in
  `app/services/qa_refinement.py`, schema in `app/services/qa_schema.py`.
- The judge should run on a **different provider/model than the grounder** when configured
  (`FORMIQO_GROUNDING_QA_PROVIDER` / `FORMIQO_GROUNDING_QA_MODEL`) to avoid correlated
  blind spots.

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
