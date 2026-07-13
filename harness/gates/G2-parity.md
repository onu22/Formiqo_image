# Gate G2 — Preview/PDF Parity

**Owner:** QA Specialist  
**Blocks:** E5  
**Status:** PENDING

## Criteria

From PRD E2 acceptance criteria:

- [ ] No duplicated validation/discovery/manifest logic between `image_stamping.py` and `pdf_stamping.py`
- [ ] Golden-file test set (≥2 forms, mixed text/checkbox/multiline)
- [ ] Stamped PNG preview and rasterized stamped PDF page match within pixel tolerance
- [ ] Same wrapping, truncation, and font rendering on both paths
- [ ] Per-field `font_size_pt` override round-trips through `stamping.json`
- [ ] `stamp-pdf` reads style from `stamping.json` (not hardcoded)
- [ ] `multiline_text` wraps identically on both paths

## Test artifacts

- Golden fixtures: `tests/fixtures/parity/` _(create during E2)_
- QA report: `harness/specs/qa-report-g2-parity-<date>.md`

## Verdict

**Status:** PENDING | QA APPROVED | QA BLOCKED

**Signed off by:** _QA Specialist_  
**Date:** _YYYY-MM-DD_

### Failures (if QA BLOCKED)

_None_
