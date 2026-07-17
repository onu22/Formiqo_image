# E5 — Vision QA Refinement Loop — Engineering / QA Note

**Epic:** E5 (LLM Engineer)
**Date:** 2026-07-17
**Depends on:** G2 QA APPROVED (parity stamping), E4 (grounding source/confidence)
**Status:** Implemented; deterministic paths verified by unit tests (full suite green). Live
judge validation pending API keys.

## Scope delivered

1. **Closed refine loop** — `app/services/grounding_qa.py::run_grounding_qa_refinement`
   - Stamp a numbered **debug preview** per iteration via `stamp_qa_preview_pages()` with
     `draw_debug_boxes=True` so every field's target box is visible in its crop.
   - **Judge:** per-field **zoomed crops** (`grounding_qa_crop_zoom`, ~3x, plus
     `grounding_qa_crop_context_px` context) sent to a vision model. The judge defaults to the
     provider that **differs** from the grounder (`resolve_judge_config`) to avoid correlated
     blind spots. Crops are bundled up to `grounding_qa_max_fields_per_call` per call.
   - **Bounded deltas:** every `dx/dy` is clamped to `grounding_qa_max_bbox_delta_px`
     (default 30) per axis per iteration — the loop can never move a field further than the
     configured bound. When ≥ `grounding_qa_consensus_min_fields` fields agree on one shift
     within `grounding_qa_consensus_max_spread_px`, they merge into a single page-wide
     **consensus translation** (`apply_verdicts_to_fields`).
   - **Iterate** until the judge reports clean (no shift on any page → `converged`) or
     `grounding_qa_max_iterations` (default 6) is reached.
   - Work is done on a **copy** under `output/qa_refine/{run_id}/` so an interrupted loop never
     corrupts the canonical field files; final bboxes are written back at the end.

2. **Persistence** — per-field `qa_status` (`confirmed | adjusted | flagged`) and QA-informed
   `confidence` written back to `page_XXXX.fields.json`. `stages.qa_refine` in `job.json`
   records status, iterations, `converged`, field tallies, judge provider/model, and cost;
   surfaced by `GET /jobs/{id}` via `job_detail_projection`.
   - `confirmed`: judge said OK with confidence above the flag threshold.
   - `adjusted`: bbox was moved at least once during the loop.
   - `flagged`: judge unsure, still wanting to shift at the iteration cap, or confidence
     ≤ `grounding_qa_flag_low_confidence` — surfaced in the editor as "check this field".

3. **Manual endpoint** — `POST /api/v1/jobs/{id}/refine-grounding` (optional
   `max_iterations` override). Reads provider/model from `job.json`, builds the judge from
   config (no hardcoded keys), returns the run summary. Missing judge API key → `400`.

4. **Optional auto-run** — `grounding_qa_enabled` (default **off**) runs the loop as the final
   stage of the E3 upload pipeline (`job_pipeline._run_auto_qa_refine`). QA-loop errors are
   isolated: they mark `stages.qa_refine` failed but never flip an already-`ready` job to
   `failed`.

5. **UI flags** — `field-overlay.tsx` badge and `inspector.tsx` banner + list icon surface
   `qa_status: flagged` fields (T035; frontend typecheck clean).

## Coordinate semantics

All bboxes stay in **top-left pixel space of the 200 DPI page PNGs** (PRD §2.2). The judge
`dx/dy` correction is defined in that same full-page pixel space; deltas are applied with
`clamp_bbox_to_page`, so a field can never leave the page and can never move more than the
per-iteration bound.

## Judge cost / latency (tokens per page per iteration)

The judge cost is **one vision call per page per iteration** (or one per
`grounding_qa_max_fields_per_call` batch on dense pages). Each call carries:

| Component | Size driver | Notes |
|-----------|-------------|-------|
| System prompt | fixed (`prompts/grounding_qa_judge.md`) | ~250 tokens, constant |
| Crop manifest text | ~1 line/field | field_id + type + bbox + page dims |
| Field crops (images) | `grounding_qa_crop_zoom` × bbox size + context | small per-field PNGs, **not** the full page |
| Output | `fields[]` verdicts | tiny; bounded by `grounding_qa_judge_max_tokens` |

Cost is captured live in `stages.qa_refine.cost` (`judge_calls`, `input_tokens`,
`output_tokens`) and in the endpoint response, so the auto-run default can be tuned against
real token counts once keys are available.

**Cost-control levers:** crops (not full pages) keep image tokens small; a cheaper/different
judge model via `grounding_qa_judge_model`; `grounding_qa_max_fields_per_call` bounds per-call
size; the iteration cap bounds total rounds; auto-run is **off by default**.

Worst-case upper bound: `pages × iterations × ceil(fields_per_page / max_fields_per_call)`
judge calls. Typical convergence is well under the cap (a page-wide misregistration collapses
via consensus translation in 1–3 iterations).

## Before/after bbox error — methodology + result

Center error = Euclidean distance between a field's bbox top-left/center and the truth, in
page pixels. Verified deterministically by `test_perturbed_bbox_is_measurably_corrected`:

- **Fixture:** a field whose true top-left is `(150, 110)` is perturbed to `(230, 190)` —
  before-error `≈ 113 px`.
- **After the loop (bound 30 px/axis/iter):** the field converges exactly to `(150, 110)` —
  after-error `0 px` — in 3 shifting iterations plus a clean confirmation iteration.
- The loop **strictly reduces** the error and never applies a per-axis move above the bound
  (asserted per iteration); `qa_status` becomes `adjusted`.

> **Live regression numbers are pending API keys.** The loop and judge are wired to config;
> point a keyed run (`FORMIQO_OPENAI_API_KEY` / `FORMIQO_ANTHROPIC_API_KEY`) at the regression
> fixtures and diff `qa_status` distribution + median center error before/after. No live
> provider call was possible in this environment.

## Completion checklist

- [x] Stamp → judge → bounded delta → re-stamp loop (`grounding_qa.py`).
- [x] `qa_status` + final confidence persisted; `stages.qa_refine` wired into `job.json` and
      `GET /jobs/{id}`.
- [x] `POST /jobs/{id}/refine-grounding` + `grounding_qa_*` config (no hardcoded keys).
- [x] Optional auto-run final stage behind `grounding_qa_enabled` (default off).
- [x] `qa_status: flagged` surfaced in the editor.
- [x] Perturbed-bbox fixture measurably corrected; per-iteration bound never exceeded.
- [x] Judge cost/latency documented (tokens per page per iteration) + captured in `job.json`.
- [x] Prompts versioned (`prompts/CHANGELOG.md` v3) + `tests/test_grounding_prompt.py` extended.
- [x] Unit/integration tests green (`tests/test_e5_refine_grounding.py`; full suite 71 passed).
- [ ] **Live before/after center-error numbers + zero-flag regression** — pending keyed run.
