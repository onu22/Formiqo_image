# E3 API Contract

**Status:** APPROVED (G1 2026-07-12)  
**Epic:** E3  
**Base path:** `/api/v1`  
**ADR:** [`adr-001-on-disk-layout.md`](adr-001-on-disk-layout.md)

## Authentication

None for MVP (single-operator deployment).

## Conventions

- **Page numbers** in URL paths and JSON are **1-based** (`/pages/1/image`, `"page_number": 1`).
- **Error bodies** use `{ "error": "<code>", "message": "human-readable" }` (not bare FastAPI `detail` strings).
- **Job IDs** must be UUIDs; reject path traversal in `{id}`, `{n}`, and run ids.

## Jobs

### `POST /jobs`

Multipart PDF upload. Creates job and runs background pipeline: convert → line-detect → grounding → (optional E5) → `ready`.

**Request:** `multipart/form-data`, field `file` (PDF)

**Responses:**
- `201` — `{ "job_id": "uuid", "status": "converting" }`
- `400` — XFA rejected, not a PDF, oversize (`Settings.max_upload_bytes`)
- `413` — payload too large

**Errors body:**
```json
{ "error": "xfa_not_supported | invalid_pdf | file_too_large", "message": "human-readable" }
```

### `GET /jobs`

List jobs.

**Response `200`:**
```json
{
  "jobs": [
    {
      "job_id": "uuid",
      "source_filename": "form.pdf",
      "page_count": 2,
      "status": "ready",
      "created_at": "ISO-8601"
    }
  ]
}
```

### `GET /jobs/{id}`

Job detail for polling.

**Response `200`:**
```json
{
  "job_id": "uuid",
  "source_filename": "form.pdf",
  "status": "grounding",
  "page_count": 5,
  "stages": {
    "grounding": { "grounded_pages": 2, "total_pages": 5 }
  },
  "errors": [],
  "artifacts": {
    "has_stamped_preview": false,
    "has_export_pdf": false
  }
}
```

### `DELETE /jobs/{id}`

Remove job directory.

**Response:** `204`

## Pages & images

### `GET /jobs/{id}/pages/{n}/image`

Page PNG for editor canvas. `{n}` is 1-based page number.

**Query:** `?variant=source|stamped` (default `source`; `stamped` = latest preview from `job.json.artifacts.latest_stamped_images_dir`)

**Response:** `image/png` (`FileResponse`)

## Fields & values

### `GET /jobs/{id}/fields`

All grounded fields + values + style — see [`field-schema.md`](field-schema.md) API projection.

### `PATCH /jobs/{id}/fields`

Persist editor changes to field geometry and review metadata.

**Request:**
```json
{
  "fields": [
    {
      "field_id": "field_001",
      "page_number": 1,
      "bbox": { "x": 10, "y": 20, "w": 100, "h": 14 },
      "font_size_pt": 10,
      "reviewed": true
    }
  ]
}
```

**Response `200`:** updated field subset

### `PATCH /jobs/{id}/values`

Persist value edits.

**Request:**
```json
{
  "values": { "field_001": "John Doe" },
  "style": { "font_size_pt": 11 }
}
```

Until E2, `style` PATCH may write through to `image_style` or be rejected with `501` — E3 implements values PATCH first; global style PATCH lands with E2.

## Stamping & export

### `POST /jobs/{id}/stamp-images`

Generate server-rendered preview PNGs. No request body (provider read from `job.json.grounding`).

**Response `200`:**
```json
{
  "run_id": "20260712T120000Z",
  "pages": [
    { "page_number": 1, "image_url": "/api/v1/jobs/{id}/pages/1/image?variant=stamped" }
  ]
}
```

E3 replaces the current verbose `StampImagesResponse` (relative paths, provider body).

### `POST /jobs/{id}/stamp-pdf`

Export flattened PDF. Sets job status `exported`. No request body.

**Response `200`:**
```json
{
  "run_id": "20260712T120000Z",
  "download_url": "/api/v1/jobs/{id}/export"
}
```

### `GET /jobs/{id}/export`

Download latest stamped PDF from `job.json.artifacts.latest_stamped_pdf`.

**Response:** `application/pdf`

## Grounding refinement (E5)

### `POST /jobs/{id}/refine-grounding`

Manual QA refinement loop re-run. **Not implemented until E5.**

**Response `202`:** `{ "status": "running" }` — poll `GET /jobs/{id}` for `stages.qa_refine`

## Static frontend (E3.11)

FastAPI serves production build from `frontend/dist/` via `StaticFiles` at `/`.  
Dev mode: existing `cors_allow_origins` for Vite dev server.

## Security requirements (G3)

- Resolve all file paths strictly under `data/jobs/{job_id}/`
- Reject path traversal in `{id}`, `{n}`, run ids
- Validate PDF magic bytes on upload
- No secrets in API responses

## Legacy endpoints

| Endpoint | Disposition |
|----------|-------------|
| `POST /user-uploads/process-convert-line-detect` | **Keep** — batch/CLI intake; secondary in OpenAPI |
| `POST /jobs/{id}/ground-fields-from-lines` | **Deprecated** — debug/re-run only after `POST /jobs` ships |
| `POST /jobs/{id}/stamp-images` (with `provider` body) | **Breaking change in E3** — remove body; read from job manifest |
| `POST /jobs/{id}/stamp-pdf` (with `provider` body) | **Breaking change in E3** — remove body; read from job manifest |

## Acceptance (E3)

- [ ] Integration test: upload → poll ready → PATCH → stamp → export
- [ ] Distinct error bodies for XFA, non-PDF, oversize
- [ ] Path traversal tests for file-serving routes
