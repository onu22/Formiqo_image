# Job Manifest Schema

**Status:** APPROVED (G1 2026-07-12)  
**Epic:** E1  
**Replaces:** duplicated `output/document_manifest.json` and `output/converted_images/manifest.json`  
**ADR:** [`adr-001-on-disk-layout.md`](adr-001-on-disk-layout.md)

## Location

One file per job at job root:

```
data/jobs/{job_id}/job.json
```

## JSON schema (informal)

```json
{
  "job_id": "uuid",
  "source_filename": "form.pdf",
  "created_at": "ISO-8601",
  "updated_at": "ISO-8601",
  "status": "converting | grounding | ready | failed | exported",
  "page_count": 2,
  "dpi": 200,
  "detected_pdf_type": "flat | scanned | acroform | xfa_rejected",
  "stages": {
    "convert": { "status": "pending | running | done | failed", "error": null, "failed_pages": [] },
    "line_detect": { "status": "...", "error": null, "failed_pages": [] },
    "grounding": { "status": "...", "error": null, "grounded_pages": 0, "total_pages": 0, "failed_pages": [] },
    "qa_refine": { "status": "skipped | running | done | failed", "iterations": 0, "error": null }
  },
  "artifacts": {
    "input_pdf": "input.pdf",
    "converted_images_dir": "output/converted_images",
    "page_manifests_dir": "output/converted_images/pages",
    "fields_dir": "output/field_grounding",
    "stamping_json": "output/field_grounding/stamping.json",
    "latest_stamp_run_id": "string | null",
    "latest_stamped_pdf": "output/stamped_pdfs/{run_id}/stamped.{provider}.pdf",
    "latest_stamped_images_dir": "output/stamped_images/{run_id} | null"
  },
  "grounding": {
    "provider": "openai | anthropic",
    "model": "string",
    "run_id": "string | null"
  },
  "retention": {
    "max_stamp_runs": 3
  }
}
```

## Status transitions

```
(converting) → grounding → ready
                ↓           ↓
              failed    exported
```

- `converting`: PDF intake + page rasterization + line detection
- `grounding`: LLM field grounding (and optional E5 qa_refine when enabled)
- `ready`: grounding complete; editor may load fields
- `failed`: unrecoverable stage error; per-page errors in `stages.*.failed_pages`
- `exported`: latest `stamp-pdf` succeeded (E3)

## Page numbering

- **API** (E3): 1-based `page_number` in paths and JSON (`/pages/1/image`, `page_number: 1`).
- **On disk:** 0-based `page_index` in filenames (`page_0001` = first page).

## Manifest ownership

| File | After E1 |
|------|----------|
| `job.json` | **Canonical** — status, stages, grounding provider/model, artifact pointers |
| `output/document_manifest.json` | **Removed** |
| `output/converted_images/manifest.json` | **Removed** |
| `output/field_grounding/manifest.json` | **Deprecated** — delete after E1; data merged into `job.json.grounding` |

Per-run manifests under `stamped_images/{run_id}/manifest.json` and `stamped_pdfs/{run_id}/manifest.json` remain run-scoped metadata (not duplicated in `job.json` except via `latest_*` pointers).

## Rules

1. **Single source of truth** — no duplicate manifest files with the same fields.
2. **Atomic updates** — update `updated_at` on every status write.
3. **Readers** — E3 `GET /jobs/{id}` returns a projection of this file; stampers read paths from `artifacts`.
4. **Delete** — `DELETE /jobs/{id}` removes the entire `data/jobs/{job_id}/` tree.
5. **No migration shim** — pre-E1 jobs are unsupported (see ADR-001).

## Acceptance (E1)

- [ ] `job.json` written at job creation
- [ ] Status transitions match pipeline stages
- [ ] No `output/document_manifest.json` or `output/converted_images/manifest.json`
- [ ] `output/field_grounding/manifest.json` no longer written
- [ ] Unit tests for read/write and transitions
- [ ] Golden fixture under `tests/fixtures/golden-job/`
