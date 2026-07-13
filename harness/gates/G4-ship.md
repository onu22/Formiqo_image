# Gate G4 — Ship Review

**Owner:** QA Specialist  
**Blocks:** MVP ship  
**Status:** PENDING

## Criteria

From PRD E6 and §5:

### Mockup fidelity (six screens)

- [ ] Jobs list — `docs/mockups/formiqo-jobs-list.png`
- [ ] Upload page — `docs/mockups/formiqo-upload-page.png`
- [ ] Processing state — `docs/mockups/formiqo-processing-state.png`
- [ ] Review editor — `docs/mockups/formiqo-review-ui-mockup.png`
- [ ] Export success — `docs/mockups/formiqo-export-success.png`
- [ ] Failed state — `docs/mockups/formiqo-failed-state.png`

### End-to-end flow (real scanned form)

- [ ] Upload → processing poll → editor loads
- [ ] Move field (drag + arrow nudge) → Save → position matches server stamped preview
- [ ] Edit value and font size → Refresh Preview → server render matches
- [ ] Export PDF → download → opens correctly
- [ ] Failed job shows per-page errors; retry path works

### Regression

- [ ] Full pytest suite green
- [ ] G2 parity tests still pass
- [ ] G3 security conditions satisfied (if any)

## Verdict

**Status:** PENDING | QA APPROVED | QA BLOCKED

**Signed off by:** _QA Specialist_  
**Date:** _YYYY-MM-DD_

### Open issues

_None_
