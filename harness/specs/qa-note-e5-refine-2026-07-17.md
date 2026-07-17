# E5 QA Note — Vision QA Refinement Loop

**Epic:** E5 (LLM Engineer) 
**Sprint:** sprint-002 (tasks T030–T036) 
**Date:** 2026-07-17 
**Depends on:** G2 QA APPROVED (parity stamping), E4 (grounding_source/confidence) 
**Prompts:** `prompts/grounding_qa_system.md`, `prompts/grounding_qa_user.md` (CHANGELOG v3)

## What shipped

Closed loop behind `POST /api/v1/jobs/{id}/refine-grounding`
(`app/services/grounding_qa.py`):

1. Stamp a debug preview of the current grounding via `stamp_qa_preview_pages()`
   (`draw_debug_boxes=True`).
2. For each **stampable** field, crop a ~3x zoomed region around the stamped bbox
   (`grounding_qa_crop_zoom`, `grounding_qa_crop_padding_px`) and ask a **vision judge** a
   narrow question: is the value on its line / in its cell? If not, the pixel shift to fix it.
3. Apply **bounded** per-axis deltas (`grounding_qa_max_bbox_delta_px`, default 30 px), with a
   single **consensus translation** merged when ≥ `grounding_qa_consensus_min_fields` fields
   agree within `grounding_qa_consensus_max_spread_px`.
4. Re-stamp and iterate until every field is clean or `grounding_qa_max_iterations` (default 6).

The judge provider/model come from config (`grounding_qa_judge_provider`/`_model`); when
unset, it prefers a **different** provider than the grounder (avoiding correlated blind
spots) if that key is present. Per-field crops keep judge images small. Errors are isolated
per field (a judge failure never sinks the page or the loop).

## Persistence

- Per field: `qa_status` ∈ `{confirmed, adjusted, flagged}` and the final judge `confidence`
  are written back to `page_XXXX.fields.json`.
  - `confirmed` — judge said `ok` and the field was never moved.
  - `adjusted` — judge said `ok` after ≥ 1 bounded move.
  - `flagged` — judge still wanted a shift at loop end (non-converged) **or** confidence
    below `grounding_qa_min_confidence` (default 0.4). The editor surfaces these
    ("check this field": amber overlay badge + inspector banner + field-list flag).
- Per job: `stages.qa_refine` = `{ status, iterations, converged, page_count, counts, cost }`,
  surfaced by `GET /jobs/{id}` for convergence-rate / iteration tracking.

## Before/after bbox error (deterministic oracle judge on perturbed fixtures)

Verified in `tests/test_e5_refine_grounding.py` using the E2 parity forms with deliberately
perturbed bboxes and an oracle judge (returns the true remaining shift; the loop clamps it):

| Scenario | Bound | Perturbation | Iterations | Median center error before → after | Outcome |
|----------|-------|--------------|-----------|------------------------------------|---------|
| Mixed per-field offsets (form_a, 3 fields) | 30 px | 12–20 px/axis | 2 | ~18 px → **0 px** | all `adjusted`, converged |
| Single large offset (form_b text) | 30 px | 100 px x | 1 | 100 px → 70 px (moved exactly 30) | `flagged`, non-converged |
| No perturbation (form_a) | 30 px | 0 | 1 | 0 → 0 (no movement) | all `confirmed` |

**Guarantees asserted by tests:** the loop strictly reduces (or leaves unchanged) bbox error;
no field moves more than the configured per-iteration bound; correct fields are confirmed and
never moved; a form with perturbed bboxes is measurably corrected.

## Judge cost / latency (for tuning the auto-run default)

Cost scales as **`judge_calls = (stampable fields) × iterations`** — one small crop image per
call. `stages.qa_refine.cost` records `judge_calls`, `input_tokens`, `output_tokens`,
`latency_s`, and the normalized `input_tokens_per_page_per_iteration` /
`output_tokens_per_page_per_iteration`.

Practical implications:
- Crops are per-field and small, so per-call input tokens are dominated by the fixed system
  prompt + one low-resolution image, not the whole page.
- A typical 2-page form with ~10 stampable fields converging in 2 iterations ≈ **~20 judge
  calls**. Auto-run (`grounding_qa_auto_run`, default **off**) multiplies upload latency by
  the judge round-trips, so it stays opt-in; the manual endpoint is the primary path for
  re-runs after user edits.
- Only **stampable** fields are judged (non-empty text / truthy toggles); empty fields are
  skipped, so an auto-run immediately after grounding (empty sample values) is effectively a
  no-op until values are set.

## Config (`app/config.py`, `FORMIQO_` env prefix)

| Setting | Default | Purpose |
|---------|---------|---------|
| `grounding_qa_max_iterations` | 6 | Loop cap |
| `grounding_qa_max_bbox_delta_px` | 30 | Per-axis, per-iteration move bound |
| `grounding_qa_consensus_translation_enabled` | true | Merge agreeing per-field shifts |
| `grounding_qa_consensus_min_fields` | 3 | Min fields for consensus |
| `grounding_qa_consensus_max_spread_px` | 4 | Max delta spread for consensus |
| `grounding_qa_auto_run` | false | Run refine as final pipeline stage |
| `grounding_qa_judge_provider` / `_model` | "" | Judge provider/model (empty → cross-provider default) |
| `grounding_qa_crop_zoom` | 3.0 | Context multiplier for crops |
| `grounding_qa_crop_padding_px` | 24 | Min crop padding |
| `grounding_qa_min_confidence` | 0.4 | Flag threshold |

Coordinate system unchanged: top-left pixel bboxes on 200-DPI page PNGs (PRD §2.2); deltas
and crops operate in that space and clamp to page bounds.

## API keys

The judge requires `FORMIQO_OPENAI_API_KEY` or `FORMIQO_ANTHROPIC_API_KEY`. Tests inject a
deterministic judge, so the whole loop is exercised offline (no network).

## Acceptance criteria (PRD E5)

- [x] Loop strictly reduces / leaves unchanged median bbox error; never exceeds per-iteration bound.
- [x] Convergence + iteration counts logged per job; surfaced in `GET /jobs/{id}` (`stages.qa_refine`).
- [x] Perturbed-bbox fixture is measurably corrected (`tests/test_e5_refine_grounding.py`).
- [x] Judge cost/latency documented (tokens per page per iteration) — this note + `stages.qa_refine.cost`.
- [x] `qa_status: flagged` surfaced in the editor.
- [x] Prompts versioned under `prompts/`; `tests/test_grounding_prompt.py` extended.
- [x] Full `pytest` suite green (69 passed).
