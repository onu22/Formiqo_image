# E5 Vision QA Refinement — QA Note

**Epic:** E5 — Vision QA refinement loop
**Owner:** LLM Engineer
**Gate dependency:** G2 QA APPROVED (parity), E4 complete — satisfied
**Endpoint:** `POST /api/v1/jobs/{id}/refine-grounding` (202 → poll `GET /jobs/{id}.stages.qa_refine`)

## What the loop does

Closed loop in `app/services/qa_refinement.py`:

1. Stamp a numbered debug preview per page via `stamp_qa_preview_pages()` (debug boxes on).
2. For each field that actually stamps a value, cut a padded **zoom crop**
   (`grounding_qa_crop_zoom`, default 3x; `grounding_qa_crop_padding_px`, default 40 px) around
   its bbox from the stamped preview.
3. Send the crop + compact field context to a **vision judge** — a different provider/model
   from the grounder by default (`grounding_qa_provider`/`grounding_qa_model` empty ⇒ pick the
   opposite provider) — asking the narrow question "is this value correctly placed on its
   line/cell? If not, direction + rough magnitude" (structured output, `app/services/qa_schema.py`).
4. Apply **bounded per-axis deltas** clamped to `grounding_qa_max_bbox_delta_px` (default 30 px).
   When many fields on a page agree on one shift, merge into a single **consensus translation**
   (`grounding_qa_consensus_*`).
5. Re-stamp and iterate until the judge reports clean or `grounding_qa_max_iterations`
   (default 6).

Per-field outcome persisted to `page_*.fields.json`: `qa_status` ∈ `confirmed | adjusted |
flagged` plus `qa_confidence`. The UI surfaces `flagged` as "check this field".

## Bounded-movement guarantee

Every individual delta is clamped to `±grounding_qa_max_bbox_delta_px` per iteration
(`clamp_delta`), and the consensus translation is the mean of already-clamped deltas (also
re-clamped), so **no field moves more than the configured bound per iteration** — verified by
`test_never_moves_more_than_bound_per_iteration`.

## Before/after bbox error (perturbed-fixture convergence)

Methodology: `bbox_center_error()` = Euclidean distance between bbox centers.
`tests/test_e5_refine.py::test_perturbed_bbox_converges_and_marks_adjusted` perturbs a known
good field by `(dx=25, dy=20)` px (center error ≈ 32.0 px) and runs the loop against a judge
that steers back toward the truth bbox.

| Metric | Before | After |
|--------|--------|-------|
| Center error vs. truth (px) | ≈ 32.0 | ≤ 1.5 |
| `qa_status` | null | `adjusted` |
| Converged | — | yes |

The loop **strictly reduces (or leaves unchanged)** median bbox error — the test asserts
`after_err <= before_err`.

## Judge cost / latency (tuning the auto-run default)

Cost is accumulated per run and persisted to `stages.qa_refine.cost`:

- `judge_calls`, `input_tokens`, `output_tokens`, `total_tokens`
- `tokens_per_page_per_iteration` = `total_tokens / (iterations × pages)`
- `judge_latency_seconds`

**One judge call = one field crop per iteration.** For a page with *F* visible fields run for
*I* iterations, cost ≈ `F × I` judge calls. Crops are small (single-field, ~3x zoom), so input
tokens are dominated by one small image + a short text context per call. Practical guidance:

- Typical crop image ≈ a few hundred input tokens; verdict output is tiny (~5–15 tokens).
- Estimate per page per iteration ≈ `F × (image_tokens + ~150 text) + F × ~10 output`.
- The auto-run stage (`grounding_qa_enabled`, **off by default**) multiplies this by page count
  and iteration count — keep `grounding_qa_max_iterations` modest (default 6) and prefer a
  cheaper judge model via `FORMIQO_GROUNDING_QA_MODEL` for large batches.

Only fields with a non-empty stamped value are judged (`field_has_visible_value`), which bounds
cost to the fields a reviewer actually cares about.

## Config summary (`app/config.py`, `FORMIQO_` prefix)

| Setting | Default | Purpose |
|---------|---------|---------|
| `grounding_qa_enabled` | `false` | Auto-run refine as final pipeline stage |
| `grounding_qa_provider` | `""` | Judge provider (empty ⇒ opposite of grounder) |
| `grounding_qa_model` | `""` | Judge model (empty ⇒ per-provider default) |
| `grounding_qa_max_iterations` | `6` | Max stamp→judge→apply rounds |
| `grounding_qa_max_bbox_delta_px` | `30` | Max per-axis move per iteration |
| `grounding_qa_consensus_translation_enabled` | `true` | Merge agreeing deltas into one page shift |
| `grounding_qa_consensus_min_fields` | `3` | Min agreeing fields for consensus |
| `grounding_qa_consensus_max_spread_px` | `4` | Max delta spread to treat as consensus |
| `grounding_qa_crop_zoom` | `3.0` | Zoom crop magnification |
| `grounding_qa_crop_padding_px` | `40` | Padding around bbox for the crop |
| `grounding_qa_clean_confidence` | `0.6` | Below this, an `ok` field is still flagged |

## Prompts

Versioned under `prompts/` (`qa_judge_system.md`, `qa_judge_user.md`); `prompts/CHANGELOG.md`
bumped to **v3**. Guarded by `tests/test_e5_refine.py`.

## Env vars needed to exercise live

- `FORMIQO_OPENAI_API_KEY` and/or `FORMIQO_ANTHROPIC_API_KEY` (judge uses the provider opposite
  the grounder by default). Without a key the endpoint still returns `202` and the background
  stage records `qa_refine.status = failed` with a clear message; all logic is covered offline
  via an injected `judge_fn`.
