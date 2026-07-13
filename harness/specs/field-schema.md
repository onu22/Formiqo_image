# Grounded Field Schema

**Status:** APPROVED (G1 2026-07-12)  
**Epic:** E1  
**Coordinate system:** top-left pixels on 200 DPI page PNGs (PRD §2.2)  
**ADR:** [`adr-001-on-disk-layout.md`](adr-001-on-disk-layout.md)

## Per-page fields file

```
data/jobs/{job_id}/output/field_grounding/page_{NNNN}.fields.json
```

`NNNN` is 1-based page number zero-padded (`page_0001` = page 1). Internal `page_index` in the file body is 0-based.

## Page wrapper

```json
{
  "page_index": 0,
  "width_px": 1700,
  "height_px": 2200,
  "unit": "px",
  "origin": "top-left",
  "fields": []
}
```

Drop triplicated dimension keys from the wrapper when present in legacy files.

## Field object

```json
{
  "field_id": "field_001",
  "type": "text | multiline_text | checkbox | radio",
  "bbox": { "x": 0, "y": 0, "w": 0, "h": 0 },
  "confidence": 0.0,
  "label": "Printed label text",
  "evidence": {
    "line_ids": ["line_h_008", "line_v_001"]
  },
  "grounding_type": "text",
  "grounding_source": "cell | line_anchor | label_anchor | pixel | template | null",
  "qa_status": "confirmed | adjusted | flagged | null",
  "reviewed": false,
  "font_size_pt": null
}
```

### Keep (E1)

| Field | Notes |
|-------|-------|
| `field_id` | Stable identifier |
| `type` | Rendering + editor behavior |
| `bbox` | `{x,y,w,h}` pixels, top-left origin |
| `confidence` | Model confidence 0–1 |
| `label` | Single canonical label string |
| `evidence.line_ids` | Detected line references |
| `grounding_type` | When it differs from `type` |
| `grounding_source` | E4/E5/E7 provenance (null until E4) |
| `qa_status` | E5 outcome (null until E5) |
| `reviewed` | UI review flag (default false) |
| `font_size_pt` | Per-field override (E2/E6; null until set) |

### Drop (E1 dedupe)

- `nearby_label_text`
- `supporting_lines` (duplicate of `evidence.line_ids`)
- `evidence.label` (duplicate of top-level `label`)
- `field_surface` (derive transiently from `type` in geometry layer; do not persist)

## Values file (separate)

```
data/jobs/{job_id}/output/field_grounding/stamping.json
```

### E1 shape (until E2 lands)

```json
{
  "values": {
    "field_001": ""
  },
  "require_all_values": false,
  "image_style": {
    "font_size_px": 22,
    "font_color": "#111111",
    "padding_px": 3,
    "draw_debug_boxes": false,
    "debug_box_color": "#ff0000"
  }
}
```

- Default values: **empty strings** (never `field_id[:10]` placeholders)
- E1 keeps `image_style` (pixels) — stampers unchanged until E2

### E2 target shape (stamping unification)

```json
{
  "style": {
    "font_size_pt": 11,
    "text_color": "#000000"
  },
  "values": {
    "field_001": ""
  },
  "overrides": {
    "field_001": { "font_size_pt": 10 }
  },
  "require_all_values": false
}
```

- Global `style` in PDF points; pixel sizes derived via page manifest scale (E2)
- `stamp-pdf` and `stamp-images` both read this file after E2

## Page manifest (deduped)

```
data/jobs/{job_id}/output/converted_images/pages/page_{NNNN}.json
```

Keep:
- `width_px`, `height_px` (single pair — drop triplicated dimension keys)
- `image_to_pdf_scale_x`, `image_to_pdf_scale_y`
- Drop `mapping.formula` documentation strings

## Detected lines (slim, per-page)

```
data/jobs/{job_id}/output/line_detection/page_{NNNN}/detected_lines.json
```

Per line in the `lines[]` array: `line_id`, `orientation`, `bbox`, `thickness`, `line_style`  
Drop redundant `x1,y1,x2,y2` when derivable from `bbox`.  
Keep the per-page wrapper (image metadata, detector info, counts) — only slim individual line objects.

## API projection (E3)

`GET /api/v1/jobs/{id}/fields` returns all pages:

```json
{
  "job_id": "uuid",
  "style": { "font_size_pt": 11, "text_color": "#000000" },
  "pages": [
    {
      "page_number": 1,
      "width_px": 1700,
      "height_px": 2200,
      "fields": [ "..." ]
    }
  ],
  "values": { "field_001": "" }
}
```

Until E2, `style` is **best-effort derived** from `image_style` (e.g. approximate `font_size_pt` from `font_size_px` and page scale). After E2, `style` reads directly from `stamping.json`.

## Acceptance (E1)

- [ ] Writers emit deduplicated schema
- [ ] `image_stamping.py`, `pdf_stamping.py`, `stamping_config.py` updated same change
- [ ] Existing tests pass + new schema tests
- [ ] Golden fixture at `tests/fixtures/golden-job/` reflects layout
