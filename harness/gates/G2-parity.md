# Gate G2 — Preview/PDF Parity

**Owner:** QA Specialist  
**Blocks:** E5  
**Status:** QA APPROVED

## Criteria

From PRD E2 acceptance criteria:

- [x] No duplicated validation/discovery/manifest logic between `image_stamping.py` and `pdf_stamping.py`
- [x] Golden-file test set (≥2 forms, mixed text/checkbox/multiline)
- [x] Stamped PNG preview and rasterized stamped PDF page match within pixel tolerance
- [x] Same wrapping, truncation, and font rendering on both paths
- [x] Per-field `font_size_pt` override round-trips through `stamping.json`
- [x] `stamp-pdf` reads style from `stamping.json` (not hardcoded)
- [x] `multiline_text` wraps identically on both paths

## Test artifacts

- Golden fixtures: `tests/fixtures/parity/` (`builder.py` materializes `form_a` and `form_b` at test time)
- Parity tests: `tests/test_e2_stamping_parity.py`
- QA report: [`specs/qa-report-g2-parity-2026-07-13.md`](../specs/qa-report-g2-parity-2026-07-13.md)

## Verdict

**Status:** QA APPROVED

**Signed off by:** QA Specialist  
**Date:** 2026-07-13

### Failures (if QA BLOCKED)

_None_

### Unblocks

- **E5** Vision QA refinement loop may start (per [`specs/delegation-plan-mvp.md`](../specs/delegation-plan-mvp.md))

## Report

Full report: [`specs/qa-report-g2-parity-2026-07-13.md`](../specs/qa-report-g2-parity-2026-07-13.md)
