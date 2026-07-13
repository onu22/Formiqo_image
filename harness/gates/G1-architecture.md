# Gate G1 — Architecture Sign-off

**Owner:** Solution Architect  
**Blocks:** E1, E2, E3, E4  
**Status:** APPROVED WITH CONDITIONS

## Criteria

Review and approve before any E1 implementation:

- [x] [`specs/job-manifest-schema.md`](../specs/job-manifest-schema.md) — job.json layout and status machine
- [x] [`specs/field-schema.md`](../specs/field-schema.md) — deduplicated field + stamping.json contract
- [x] [`specs/api-contract.md`](../specs/api-contract.md) — E3 REST paths and payloads for UI
- [x] Coordinate system unchanged: top-left pixel space, 200 DPI PNGs (PRD §2.2)
- [x] No database introduced for MVP (filesystem under `data/jobs/`)

## Review checklist (SA)

- [x] Schemas are minimal and sufficient for E2 stamping unification
- [x] API contract supports E6 editor (poll, fields, PATCH, stamp, export)
- [x] Status transitions cover E5 optional qa_refine stage
- [x] Breaking changes from current on-disk layout documented for E1 migration

## Verdict

**Status:** APPROVED WITH CONDITIONS

**Signed off by:** Solution Architect  
**Date:** 2026-07-12

### Conditions (if APPROVED WITH CONDITIONS)

1. **E1 implements ADR-001** — [`specs/adr-001-on-disk-layout.md`](../specs/adr-001-on-disk-layout.md) is binding for paths, manifest ownership, and no migration shim.
2. **E1 keeps `image_style` in stamping.json** — unified `style`/`overrides` in PDF points is E2 scope; E1 only fixes empty default values.
3. **E3 breaking changes on stamp endpoints** — remove `provider` request body; adopt contract response shapes (`run_id`, `image_url`, `download_url`).
4. **E3 values PATCH before global style PATCH** — style PATCH may wait for E2 or write through to `image_style` temporarily.
5. **`refine-grounding` remains stub until E5** — `job.json.stages.qa_refine.status` defaults to `skipped`.
6. **Golden fixture required** — `tests/fixtures/golden-job/` must match signed schemas before E1 closes.

### Blockers (if BLOCKED)

_None — resolved during G1 review (path conflicts, manifest ownership, page numbering, field_surface drop)._

## Notes

- Specs updated 2026-07-12 to align with running code: flat `field_grounding/page_*.fields.json`, per-page `line_detection/`, PNGs at `converted_images/` root.
- `output/field_grounding/manifest.json` deprecated; provider/model moves to `job.json.grounding`.
- Pre-E1 jobs under `data/jobs/` are unsupported — re-upload required.
- **PM:** Unblock sprint tasks T005–T022 (E1, E3, E4). E6 scaffold (T018–T019) may proceed against API contract; live integration still requires G3.
