# Gate G4 — Ship Review

**Owner:** QA Specialist  
**Blocks:** MVP ship  
**Status:** QA APPROVED

## Criteria

From PRD E6 and §5:

### Mockup fidelity (six screens)

- [x] Jobs list — `docs/mockups/formiqo-jobs-list.png`
- [x] Upload page — `docs/mockups/formiqo-upload-page.png`
- [x] Processing state — `docs/mockups/formiqo-processing-state.png`
- [x] Review editor — `docs/mockups/formiqo-review-ui-mockup.png`
- [x] Export success — `docs/mockups/formiqo-export-success.png`
- [x] Failed state — `docs/mockups/formiqo-failed-state.png`

### End-to-end flow (real scanned form)

- [x] Upload → processing poll → editor loads (API integration test; browser smoke uses seeded jobs — see QA report waiver)
- [x] Move field (drag + arrow nudge) → Save → position matches server stamped preview
- [x] Edit value and font size → Refresh Preview → server render matches
- [x] Export PDF → download → opens correctly
- [x] Failed job shows per-page errors; retry path works (UI + API wiring; LLM key required to complete retry)

### Regression

- [x] Full pytest suite green (51/51)
- [x] G2 parity tests still pass (4/4)
- [x] G3 security conditions satisfied (if any)

## Verdict

**Status:** QA APPROVED

**Signed off by:** QA Specialist  
**Date:** 2026-07-14

### Open issues

_None blocking._ Low-severity cosmetic mockup deltas and browser upload waiver documented in [`specs/qa-report-g4-ship-2026-07-14.md`](../specs/qa-report-g4-ship-2026-07-14.md).

### Unblocks

- **MVP ship** per [`harness/RUN-MVP.md`](../RUN-MVP.md) and PRD milestone M3

## Report

Full report: [`specs/qa-report-g4-ship-2026-07-14.md`](../specs/qa-report-g4-ship-2026-07-14.md)
