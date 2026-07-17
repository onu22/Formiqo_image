# QA Note — E7 Template Memory (2026-07-17)

**Epic:** E7 (stretch, M5) · **Owner:** LLM Engineer · **Branch:** `cursor/harness-conductor-mvp-post-mvp-7cdc`

## What shipped

Reuse of human-corrected grounding for pages whose **detected-line layout** matches a
previously corrected page, skipping the LLM for those pages. Tasks T040–T043.

- **T040 — Fingerprint** (`app/services/template_memory.py::page_fingerprint`)
  SHA-256 over the *normalized* detected-line set. Each line becomes a token
  `{h|v}:{qx},{qy},{qw},{qh}` where coordinates are divided by page dims and quantized into
  `template_fingerprint_bins` (default 200) bins per axis, making the fingerprint
  **scale/DPI invariant**. Tokens are deduped + sorted before hashing (order independent).
  Pages with fewer than `template_min_lines` (default 4) valid lines return `None` — they are
  not eligible, which prevents false-positive matches on near-empty pages.

- **T041 — Index** (`TemplateStore`) A small on-disk store under `templates_dir`
  (default `./data/templates`, no DB): `index.json` maps fingerprint → metadata
  (`source_job_id`, `page_index`, dims, `field_count`, timestamps); `<fingerprint>.fields.json`
  holds the corrected page payload. Captured on editor save via `capture_corrected_pages`,
  hooked into `PATCH /api/v1/jobs/{id}/fields` (best-effort; never breaks a save).

- **T042 — Reuse + skip LLM**
  `match_templates_for_job` fingerprints each page after line detection and returns template
  grounding for matches; `build_template_grounding_for_page` **scales stored bboxes** to the
  new page's pixel dims (preserving 200-DPI top-left semantics) and marks every field
  `grounding_source: template`. `run_semantic_grounding_for_job` accepts
  `template_page_results`, seeds matched pages as already-grounded, and **only constructs a
  provider client / requires an API key when at least one page still needs the LLM**. Wired
  into `job_pipeline.run_full_job_pipeline`.

- **T043 — Tests** `tests/test_e7_template_memory.py` (12 tests): scale invariance, min-lines
  gate, near-miss divergence, missing-dims guard, store round-trip, bbox scaling + source
  marking, capture→match across DPIs, near-miss non-match, disabled-flag no-op, and two
  grounding integration tests using a **call-counting fake OpenAI**.

## Acceptance criteria (PRD §E7)

- [x] Pages fingerprinted by normalized detected-line layout (stable, scale-invariant hash).
- [x] Re-uploading a previously corrected form reuses corrected grounding and skips the LLM;
      matched pages carry `grounding_source: template`.
      Verified by `test_fully_templated_reupload_makes_zero_llm_calls`: **0 client
      constructions, 0 model calls**, job reaches `ready`, fields match the corrected layout.
- [x] Near-miss (different layout) does **not** false-positive match
      (`test_near_miss_does_not_match`, `test_partial_template_grounds_only_unmatched_pages`).
- [x] Fingerprints stored in a small `data/` index (no database).

## Cost / latency

Template reuse is **pure I/O** — no tokens, no network. A fully templated re-upload makes zero
LLM calls; partial matches call the model only for unmatched pages (confirmed: exactly 1 call
for the single unmatched page in the 2-page test).

## Config

`FORMIQO_TEMPLATE_MEMORY_ENABLED` (default on), `FORMIQO_TEMPLATES_DIR` (`./data/templates`),
`FORMIQO_TEMPLATE_FINGERPRINT_BINS` (200), `FORMIQO_TEMPLATE_MIN_LINES` (4). Test suites point
`templates_dir` at a tmp dir to keep the real index clean.

## Tests

`python3 -m pytest -q` → **81 passed** (E7 file: 12 passed).
