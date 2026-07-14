# E4 — Grounding Accuracy — Engineering / QA Note

**Epic:** E4 (LLM Engineer)
**Date:** 2026-07-13
**Status:** Implemented (deterministic paths verified by unit tests); live provider validation
pending API keys.

## Scope delivered

1. **Structured provider outputs** — `app/services/grounding_schema.py`
   - OpenAI: strict `response_format={"type":"json_schema", ..., "strict":true}` with a closed
     schema (`additionalProperties:false`, every property required, optional fields expressed
     as nullable unions).
   - Anthropic: forced tool-use (`emit_grounded_fields`), structured input read directly from
     the `tool_use` block — no text JSON parsing.
   - The regex fence-strip + compact-JSON retry now only runs as a **thin fallback** for the
     unstructured path (`grounding_structured_outputs=false` or a rare parse error).

2. **Parallel per-page grounding** — `run_semantic_grounding_for_job`
   - `ThreadPoolExecutor` bounded by `FORMIQO_GROUNDING_MAX_CONCURRENCY` (default 4).
   - Per-page error isolation preserved: one page raising never sinks the others; failures are
     collected into `failed_pages` and the job still completes if any page succeeds.
   - Incremental `grounded_pages` progress written to `job.json` under a lock for `GET /jobs/{id}`.

3. **Anchor-first grounding** — `app/services/form_geometry.py`
   - Model contract changed from "output pixel bbox" to "output an `anchor`" (`cell`,
     `line_anchor`, `label_anchor`, or `none`). Bboxes are computed deterministically:
     - `cell` → intersection cell from bounding `line_id`s.
     - `line_anchor` → writable band on the referenced horizontal line.
     - `label_anchor` → placed relative to a PyMuPDF-extracted label (digital PDFs), via
       `app/services/label_anchors.py` (pixel-space, rotation-skipped, scanned → empty).
   - Unanchored fields fall back to the pixel estimate, optionally against a **labeled
     coordinate-grid overlay** (`app/services/grid_overlay.py`,
     `FORMIQO_GROUNDING_GRID_OVERLAY_ENABLED`, default off).

4. **Provenance persisted** — every field now carries `grounding_source`
   (`cell | line_anchor | label_anchor | pixel`) and `confidence`, consumed by the UI and E5.

## Before/after bbox error — methodology

Center error = Euclidean distance between a field's bbox center and the hand-labeled truth
center, in page pixels; report median + p90 over the regression form set.

- **Baseline (v1, pixel-only):** model pixel bbox → geometry snap.
- **After (v2, anchor-first):** anchor-resolved bbox (cell/line/label), pixel fallback only
  for unanchored fields.

Deterministic expectation, verified by unit tests: any field the model anchors to a cell or a
detected line collapses center error to the geometry's own precision (≤ inset px), independent
of the model's pixel guess. In `test_resolve_cell_anchor_sets_source_cell` a field whose raw
bbox is `(0,0,10,10)` resolves into the correct cell interior `x∈[100,400]`.

> **Live numbers (median/p90 before vs after) are pending** — they require API keys to run the
> regression set end-to-end. The harness is in place: point a keyed run at the regression
> fixtures and diff `grounding_source` distribution + center error. No live provider call was
> possible in this environment (`FORMIQO_OPENAI_API_KEY` / `FORMIQO_ANTHROPIC_API_KEY` unset).

## Cost / latency notes (for E5 tuning)

- Structured outputs add no extra round-trips; token cost ≈ unchanged vs v1 (schema/tool
  overhead is small and constant per page).
- `label_anchors_json` adds bounded input tokens (capped at 120 anchors/page, ≤80 chars each).
- Concurrency trades wall-clock for peak provider RPS: a 5-page doc grounds in ≈ the time of
  its slowest pages at `max_concurrency≥5`, not the sum. Tune `FORMIQO_GROUNDING_MAX_CONCURRENCY`
  to the provider rate limit.

## Completion checklist

- [x] Structured outputs (OpenAI json_schema + Anthropic tool-use); thin fallback retained.
- [x] Parallel per-page grounding with bounded concurrency + per-page error isolation.
- [x] Anchor-first deterministic bbox computation (cell / line_anchor / label_anchor).
- [x] `grounding_source` + `confidence` persisted.
- [x] Prompts versioned (`prompts/CHANGELOG.md` v2) + `tests/test_grounding_prompt.py` extended.
- [x] Unit tests green (`tests/test_e4_grounding.py`, full suite 51 passed).
- [ ] **Zero JSON parse failures on regression set** — pending keyed run (structurally enforced).
- [ ] **Live before/after center-error numbers** — pending keyed run.
