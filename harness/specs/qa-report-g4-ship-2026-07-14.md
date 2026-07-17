# QA Report — G4 Ship Review — 2026-07-14

**Gate:** G4  
**Status:** QA APPROVED  
**Pass rate:** 14/14 criteria (3 documented Low-severity cosmetic notes)

**Reviewer:** QA Specialist (independent; did not implement E6)

## Acceptance criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| Jobs list matches `formiqo-jobs-list.png` | PASS | Table layout, status pills, search/filter, actions column faithful; seed data shows 3 jobs vs 5 in mockup (cosmetic) |
| Upload page matches `formiqo-upload-page.png` | PASS | Drag-drop zone, progress row, XFA warning copy, constraints text present |
| Processing state matches `formiqo-processing-state.png` | PASS | Checklist, progress ring, stage progress bar; footer lacks "Flat scanned PDF" metadata line (Low) |
| Review editor matches `formiqo-review-ui-mockup.png` | PASS | Three-pane layout, toolbar, inspector, thumbnails, overlays, flagged indicator |
| Export success matches `formiqo-export-success.png` | PASS | Modal over dimmed editor, download CTA, back/go links |
| Failed state matches `formiqo-failed-state.png` | PASS | Per-page errors, retry/review/delete actions; footer lacks upload date (Low) |
| Upload → processing poll → editor loads | PASS* | *Browser smoke uses seeded jobs (no LLM key in CI). `test_e3_api.py::test_upload_poll_fields_patch_stamp_export` covers live API path with mocked grounding |
| Move field (drag + arrow nudge) → Save → server preview match | PASS | Playwright smoke: 1px/Shift+10px nudge + drag; server bbox/value round-trip exact |
| Edit value + font size → Refresh Preview → server render | PASS | stamp-images 200; value/font_size_pt persisted via PATCH |
| Export PDF → download → opens correctly | PASS | stamp-pdf 200; export returns valid `%PDF` (969 bytes, 1 page) |
| Failed job per-page errors; retry path | PASS | Errors surfaced in UI + API; retry POST reaches grounding (blocked only by missing `FORMIQO_OPENAI_API_KEY`, expected in CI) |
| Full pytest suite green | PASS | 51/51 passed |
| G2 parity tests still pass | PASS | 4/4 in `test_e2_stamping_parity.py` |
| G3 security conditions satisfied | PASS | `test_g3_security.py` green; trusted-network + sanitized errors per G3 conditions |

## Tests executed

| Command | Result |
|---------|--------|
| `pytest tests/ -v` | **51 passed**, 1 warning |
| `pytest tests/test_e2_stamping_parity.py -v` | **4 passed** |
| `cd frontend && npm run build` | **Success** |
| `cd frontend && npm run typecheck` | **Success** |
| `cd frontend && npm run lint` | **Success** |
| `python scripts/seed-e2e-jobs.py` + Playwright `e2e/smoke.mjs` | **All 12 checks passed**, zero console errors |
| `curl POST .../stamp-pdf` + `GET .../export` | **200**, valid PDF |
| `curl POST .../ground-fields-from-lines` (failed job retry) | **400** missing API key (artifacts valid; path wired) |

## Mockup walkthrough

Screenshots captured at `/tmp/pw-screens/` during smoke run compared side-by-side against `docs/mockups/*.png`. All six states structurally match approved mockups. Editor seed content uses a minimal synthetic page rather than the rich patient-intake artwork in the mockup (acceptable for deterministic CI; overlays and inspector match spec).

## Issues

### Issue #1: Browser e2e skips live upload→grounding poll
- **Severity:** Low (waived — harness stop condition for missing LLM secret)
- **Steps:** Run smoke without `FORMIQO_OPENAI_API_KEY`
- **Expected:** Full browser upload through grounding to editor
- **Actual:** Smoke seeds ready/processing/failed jobs; API integration test covers upload path
- **Waiver:** PRD single-operator MVP; PM playbook documents API-key stop condition. Backend httpx test is authoritative for upload flow.

### Issue #2: Minor mockup cosmetic deltas
- **Severity:** Low
- **Notes:** Processing footer metadata, failed-state upload timestamp, jobs-list row count differ from static mockup artwork only.

## Recommendation

**QA APPROVED** — G4 criteria met. PM may mark MVP ship ready (G4 unblocks ship per `harness/RUN-MVP.md`).
