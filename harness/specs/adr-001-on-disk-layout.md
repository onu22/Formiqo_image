# ADR-001: On-disk job layout and manifest ownership

**Status:** Accepted  
**Date:** 2026-07-12  
**Deciders:** Solution Architect (G1)

## Context

G1 review found path and ownership conflicts between draft specs and the running codebase. E1 must not start until canonical paths, manifest ownership, and migration rules are frozen.

## Decision

1. **Single job manifest** at `data/jobs/{job_id}/job.json` replaces `output/document_manifest.json` and `output/converted_images/manifest.json`. `output/field_grounding/manifest.json` is **deprecated** after E1; provider/model/run metadata moves into `job.json.grounding`.

2. **Keep existing directory layout** for per-page artifacts (no mass renames):
   - PNGs: `output/converted_images/page_{NNNN}.png`
   - Page manifests: `output/converted_images/pages/page_{NNNN}.json`
   - Fields: `output/field_grounding/page_{NNNN}.fields.json` (flat dir, no `pages/` subfolder)
   - Lines: `output/line_detection/page_{NNNN}/detected_lines.json` (per-page; slim line objects only)

3. **Page numbering:** API paths and payloads use **1-based** `page_number` / `{n}`. On-disk filenames and internal `page_index` remain **0-based** (`page_0001` = page 1).

4. **Field dedupe:** Drop `nearby_label_text`, `supporting_lines`, `evidence.label`, and `field_surface` from persisted field JSON. Geometry may compute `field_surface` transiently during grounding normalization; it is not stored.

5. **stamping.json split by epic:** E1 changes default values to empty strings and keeps existing `image_style` shape. E2 introduces unified `style` (PDF points) + `overrides`; both stampers migrate in E2.

6. **No backward compatibility** for pre-E1 jobs. Existing jobs under `data/jobs/` must be re-processed or deleted.

7. **Legacy API:** `POST /jobs/{id}/ground-fields-from-lines` remains for batch/debug until E3 upload pipeline is stable; marked deprecated in OpenAPI. Primary intake is `POST /jobs`.

## Rationale

- Per-page line detection and flat field paths match working code and grounding/stamping readers — renaming would add risk with no PRD benefit.
- One `job.json` eliminates duplicate manifest data without collapsing unrelated per-page artifacts.
- Dropping `field_surface` from storage reduces schema surface; `form_geometry` can derive it from `type` when needed.
- Deferring `stamping.json` style unification to E2 matches PRD epic boundaries (E1 = data cleanup, E2 = stamping parity).

## Alternatives considered

| Alternative | Rejected because |
|-------------|------------------|
| Single `output/detected_lines.json` | Breaks per-page isolation and existing grounding pipeline |
| `field_grounding/pages/` subdir | Unnecessary rename; all readers use flat path today |
| Migrate old jobs with a shim | Out of MVP scope; operator can re-upload |
| Persist `field_surface` | Redundant with `type`; used only during normalization |

## Consequences

- E1 must update all manifest readers/writers and delete duplicate manifest files.
- E3 implements new endpoints; stamp response shapes change when provider body is removed.
- E2 owns `stamping.json` style migration and stamper updates for PDF-point model.
- Golden fixture at `tests/fixtures/golden-job/` documents the canonical post-E1 layout.

## Compliance

Meets PRD §2.2 coordinate system, §E1 scope, filesystem-only MVP, and G1 gate criteria. Security path containment unchanged (G3 reviews E3 file serving).
