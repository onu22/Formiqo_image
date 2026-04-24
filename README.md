## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


{
  "provider": "anthropic",
  "model": "claude-opus-4-7"
}

{
  "provider": "openai",
  "model": "gpt-4o"
}

## Stamp PDF

Consumes the stamp plan produced by `prepare-stamp` and writes values onto the
original PDF using PyMuPDF. One canonical stamped PDF is produced per
`{provider}_{model}` run (overwrite-in-place).

### Endpoint

`POST /api/v1/jobs/{job_id}/stamp-pdf`

Request body (all keys optional):

```json
{
  "provider": "anthropic",
  "model": "claude-opus-4-7",
  "values": {
    "first_name": "Jane",
    "last_name": "Doe",
    "how_old_are_you": "32"
  },
  "strict": false,
  "options": {
    "fontsize": 10,
    "fontname": "helv",
    "color_rgb": [0, 0, 0],
    "align": "left",
    "autoshrink": true,
    "min_fontsize": 6
  }
}
```

Behavior:

- `provider` and `model` fall back to `FORMIQO_GROUNDING_PROVIDER` and
  `FORMIQO_GROUNDING_MODEL` if omitted. The endpoint looks up the plan at
  `output/stamp_plans/{provider}_{model}/`.
- `values` is a `{field_id: string}` map. Missing field ids are skipped (left
  blank) and recorded as `no_value`. Unknown field ids are recorded as
  `unknown_field_ids`.
- `strict: true` rejects the request (400) if any field id is unknown or if any
  known field has no value.
- `options.autoshrink` fits text into the field rectangle by shrinking the font
  down to `min_fontsize` when needed. If it still doesn't fit, the field is
  stamped at the smallest size and marked `overflow`.

### Curl

Stamp:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/jobs/<job_id>/stamp-pdf" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "anthropic",
    "model": "claude-opus-4-7",
    "values": { "first_name": "Jane", "last_name": "Doe" }
  }'
```

Download the stamped PDF:

```bash
curl -OJ "http://127.0.0.1:8000/api/v1/jobs/<job_id>/stamped.pdf?provider=anthropic&model=claude-opus-4-7"
```

### Output layout

```
output/
  stamped_pdfs/
    {provider}_{model}/
      stamped.pdf          # the filled PDF
      stamp_result.json    # audit of every field: status, final fontsize, errors
```

### Response shape

```json
{
  "job_id": "...",
  "provider": "anthropic",
  "model": "claude-opus-4-7",
  "run_dir": "stamped_pdfs/anthropic_claude-opus-4-7",
  "stamped_pdf_path": "stamped_pdfs/anthropic_claude-opus-4-7/stamped.pdf",
  "result_path": "stamped_pdfs/anthropic_claude-opus-4-7/stamp_result.json",
  "page_count": 1,
  "stamped_field_count": 2,
  "skipped_missing_values": ["personality_trait"],
  "unknown_field_ids": [],
  "download_url": "http://127.0.0.1:8000/api/v1/jobs/<job_id>/stamped.pdf?provider=anthropic&model=claude-opus-4-7",
  "pages": [
    {
      "page_index": 0,
      "stamped": 2,
      "fields": [
        { "field_id": "first_name", "status": "stamped", "final_fontsize": 10 },
        { "field_id": "last_name",  "status": "stamped", "final_fontsize": 10 },
        { "field_id": "personality_trait", "status": "no_value" }
      ]
    }
  ]
}
```

Errors: `400` (bad job, missing stamp plan, strict violation, bad options),
`404` (job not found), `422` (failed to save PDF).

### Alternative consumers

The stamp plan JSON is library-agnostic. If you prefer to stamp client-side,
use the ReportLab + pypdf snippet from "Prepare Stamp" on `pdf_bbox_bl`, or
write directly on the plan's `pdf_rect_tl` with PyMuPDF. The `stamp-pdf`
endpoint is simply the first-party PyMuPDF implementation of that snippet.
