## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```bash
pip3 install -r requirements-dev.txt
pytest
```


## Field Grounding POC

## Convert + Ground in One Request (Provider-Specific)

Use one of these endpoints to upload a PDF, convert pages once, and run grounding
for that specific provider:

- `POST /api/v1/convert-and-ground/anthropic`
- `POST /api/v1/convert-and-ground/openai`

Request (`multipart/form-data`) for both:

- `file` (required PDF upload)
- `dpi` (optional, default `200.0`)
- `allow_rotated_pages` (optional, default `false`)

Anthropic example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/convert-and-ground/anthropic" \
  -F "file=@/absolute/path/to/form.pdf" \
  -F "dpi=200" \
  -F "allow_rotated_pages=false"
```

OpenAI example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/convert-and-ground/openai" \
  -F "file=@/absolute/path/to/form.pdf"
```

Model selection is fixed by endpoint defaults:

- `/convert-and-ground/anthropic` uses `claude-opus-4-7`
- `/convert-and-ground/openai` uses `gpt-5.5`

Response shape:

```json
{
  "job_id": "...",
  "convert": {
    "job_id": "...",
    "page_count": 1,
    "dpi": 200.0,
    "allow_rotated_pages": false,
    "source_filename": "form.pdf",
    "document_manifest": { "...": "..." }
  },
  "grounding": {
    "job_id": "...",
    "provider": "anthropic",
    "model": "claude-opus-4-7",
    "run_id": "...",
    "run_dir": "field_grounding/anthropic_claude-opus-4-7",
    "page_count": 1,
    "succeeded_count": 1,
    "failed_count": 0,
    "output_dir": "field_grounding/anthropic_claude-opus-4-7",
    "manifest_path": "field_grounding/anthropic_claude-opus-4-7/manifest.json",
    "stamping_sample_path": "field_grounding/anthropic_claude-opus-4-7/stamping.json",
    "pages": []
  }
}
```

After each successful `/convert-and-ground/*` call, a paste-ready stamping sample is auto-generated for that same provider/model run:

- `output/field_grounding/{provider}_{model}/stamping.json`

Sample shape:

```json
{
  "values": {
    "your_full_legal_name.given_name": "your_full_"
  },
  "require_all_values": false
}
```

The value rule is deterministic: for each `field_id`, value is the first 10 characters of that `field_id`.
You can paste this JSON directly into both:

- `/api/v1/jobs/{job_id}/stamp-images/{provider}`
- `/api/v1/jobs/{job_id}/stamp-pdf/{provider}`

Failure behavior:

- `400`: invalid upload/request parameters.
- `413`: upload too large.
- `422`: conversion failed, or grounding failed for all pages.

## Image Stamping POC (Provider-Specific Endpoints)

This stamps values directly onto the converted PNG page images. Coordinates stay
in image space: pixels (`px`) with a top-left origin. No PDF point conversion is
used in this endpoint.

The endpoints read:

- `output/converted_images/page_*.png`
- `output/converted_images/pages/page_*.json`
- `output/field_grounding/{provider}_{model}/page_*.fields.json`

It writes stamped copies under:

```text
output/stamped_images/{provider}_{model}/{stamp_run_id}/
  manifest.json
  page_0001.{provider}.stamped.png
```

Run Anthropic stamping (uses Anthropic grounding JSON):

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/jobs/<job_id>/stamp-images/anthropic" \
  -H "Content-Type: application/json" \
  -d '{
    "values": {
      "first_name": "Jane",
      "last_name": "Doe"
    }
  }'
```

Run OpenAI stamping (uses OpenAI grounding JSON):

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/jobs/<job_id>/stamp-images/openai" \
  -H "Content-Type: application/json" \
  -d '{
    "values": {
      "first_name": "Jane",
      "last_name": "Doe"
    },
    "style": {
      "font_size_px": 22,
      "font_color": "#111111",
      "padding_px": 3,
      "draw_debug_boxes": false,
      "debug_box_color": "#ff0000"
    },
    "require_all_values": false
  }'
```

Request body:

```json
{
  "values": {
    "first_name": "Jane",
    "last_name": "Doe"
  },
  "style": {
    "font_size_px": 22,
    "font_color": "#111111",
    "padding_px": 3,
    "draw_debug_boxes": false,
    "debug_box_color": "#ff0000"
  },
  "require_all_values": false
}
```

Response body:

```json
{
  "job_id": "...",
  "provider": "anthropic",
  "model": "claude-opus-4-7",
  "stamp_run_id": "20260425T113800123456Z",
  "run_dir": "stamped_images/anthropic_claude-opus-4-7/20260425T113800123456Z",
  "manifest_path": "stamped_images/anthropic_claude-opus-4-7/20260425T113800123456Z/manifest.json",
  "page_count": 1,
  "succeeded_count": 1,
  "failed_count": 0,
  "pages": [
    {
      "page_index": 0,
      "status": "succeeded",
      "source_image": "converted_images/page_0001.png",
      "grounding_file": "field_grounding/anthropic_claude-opus-4-7/page_0001.fields.json",
      "output_image": "stamped_images/anthropic_claude-opus-4-7/20260425T113800123456Z/page_0001.anthropic.stamped.png",
      "field_count": 4,
      "stamped_count": 2,
      "missing_value_count": 2,
      "unsupported_field_count": 0,
      "warnings": []
    }
  ]
}
```

Notes:

- Provider is locked by endpoint path:
  - `/stamp-images/anthropic` always uses Anthropic grounding folders.
  - `/stamp-images/openai` always uses OpenAI grounding folders.
- Model is locked by provider default:
  - `/stamp-images/anthropic` uses `claude-opus-4-7`.
  - `/stamp-images/openai` uses `gpt-5.5`.
- Missing values are skipped by default.
- Set `"require_all_values": true` to fail a page when any grounded `field_id`
  has no value.
- Set `"draw_debug_boxes": true` to draw field rectangles for visual debugging.
- The original files in `converted_images/` are never modified.

## PDF Stamping POC (Provider-Specific Endpoints)

This stamps values onto the original job PDF using grounded image-space bboxes.

Endpoints:

- `POST /api/v1/jobs/<job_id>/stamp-pdf/anthropic`
- `POST /api/v1/jobs/<job_id>/stamp-pdf/openai`

Provider/model behavior:

- `/stamp-pdf/anthropic` uses `claude-opus-4-7`
- `/stamp-pdf/openai` uses `gpt-5.5`
- No `model` override field is accepted.
- No `style` input field is accepted; PDF style uses internal defaults.

Request body:

```json
{
  "values": {
    "first_name": "Jane",
    "last_name": "Doe"
  },
  "require_all_values": false
}
```

Coordinate conversion (image top-left px -> PDF bottom-left pt):

- `scale_x = pdf_width_pt / image_width_px`
- `scale_y = pdf_height_pt / image_height_px`
- `pdf_x = x * scale_x`
- `pdf_y = pdf_height_pt - ((y + h) * scale_y)`
- `pdf_w = w * scale_x`
- `pdf_h = h * scale_y`

Outputs are written under:

```text
output/stamped_pdfs/{provider}_{model}/{stamp_run_id}/
  stamped.{provider}.pdf
  manifest.json
```

Anthropic example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/jobs/<job_id>/stamp-pdf/anthropic" \
  -H "Content-Type: application/json" \
  -d '{
    "values": {
      "first_name": "Jane",
      "last_name": "Doe"
    }
  }'
```
