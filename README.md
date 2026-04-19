# Formiqo PDF → images (grounding)

Source repository: [https://github.com/onu22/Formiqo_image](https://github.com/onu22/Formiqo_image)

## Clone (Git)

```bash
git clone https://github.com/onu22/Formiqo_image.git
cd Formiqo_image
```

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

For tests:

```bash
pip3 install -r requirements-dev.txt
pytest
```

## HTTP API (FastAPI + Swagger)

Run the service from the repository root:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI JSON**: `GET /openapi.json`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/convert` | Multipart upload: field `file` (`.pdf`), form fields `dpi` (default 200), `allow_rotated_pages` (default false). Returns JSON including `document_manifest` and `links` to download artifacts. |
| `GET` | `/api/v1/jobs/{job_id}/archive.zip` | Zip of the `output/` tree for a completed job |
| `GET` | `/api/v1/jobs/{job_id}/document_manifest.json` | Same manifest as embedded in the convert response |

### Environment (`FORMIQO_` prefix)

| Variable | Default | Meaning |
|----------|---------|---------|
| `FORMIQO_JOBS_DIR` | `./data/jobs` | Per-job workspace root (UUID subfolders) |
| `FORMIQO_MAX_UPLOAD_BYTES` | `52428800` | Max PDF upload size |
| `FORMIQO_CORS_ALLOW_ORIGINS` | *(empty)* | Comma-separated origins; if set, enables CORS for browser clients |

Failed conversions remove the job directory; successful jobs leave files on disk until you delete them (add a retention job in production if needed).

### Docker

```bash
docker build -t formiqo-grounding .
docker run --rm -p 8000:8000 -v formiqo-jobs:/data/jobs formiqo-grounding
```

## Usage (CLI)

```bash
python scripts/convert_pdf_pages_for_grounding.py /path/to/form.pdf --output-dir ./out --dpi 200
```

- **Output layout**: `out/converted_images/page_0001.png`, `out/converted_images/pages/page_0001.json`, `out/converted_images/manifest.json`, `out/document_manifest.json`.
- **Paths in JSON**: image and manifest paths are relative to `--output-dir` unless noted otherwise in the document manifest (`source_pdf` is absolute when possible).
- **Coordinates**: image origin is top-left; PDF origin is bottom-left. See `mapping.formula` in each per-page JSON and comments in the script.

### Options

| Flag | Description |
|------|-------------|
| `--output-dir` | Required. Root directory for images and manifests. |
| `--dpi` | Rasterization resolution (default 200). Uniform scale only. |
| `--overwrite` | Replace existing outputs under `converted_images`. |
| `--allow-rotated-pages` | Allow non-zero `page.rotation` (simple linear mapping may not match PDF user space; see manifest `mapping.status`). |
| `--self-check` | Run a built-in temp-PDF round-trip and exit 0 on success. |

## Coordinate mapping

Per-page manifests include `image_to_pdf_scale_x` / `image_to_pdf_scale_y` and string formulas for converting a top-left image bbox `(x, y, w, h)` to PDF space. Use `map_image_bbox_to_pdf()` from the script module in Python code.

### Example per-page manifest (`converted_images/pages/page_0001.json`)

```json
{
  "manifest_version": "1.0",
  "page_index": 0,
  "pdf": {
    "width_pt": 612.0,
    "height_pt": 792.0,
    "origin": "bottom-left"
  },
  "image": {
    "path": "converted_images/page_0001.png",
    "format": "png",
    "width_px": 1700,
    "height_px": 2200,
    "origin": "top-left",
    "rendered_image_width_px": 1700,
    "rendered_image_height_px": 2200,
    "saved_image_width_px": 1700,
    "saved_image_height_px": 2200
  },
  "rendering": {
    "dpi": 200,
    "zoom": 2.7777777777777777,
    "library": "pymupdf",
    "pymupdf_version": "1.27.2"
  },
  "mapping": {
    "image_to_pdf_scale_x": 0.36,
    "image_to_pdf_scale_y": 0.36,
    "formula": {
      "pdf_x": "x * image_to_pdf_scale_x",
      "pdf_y": "pdf_height_pt - ((y + h) * image_to_pdf_scale_y)",
      "pdf_w": "w * image_to_pdf_scale_x",
      "pdf_h": "h * image_to_pdf_scale_y"
    }
  }
}
```

Numeric fields are written at full float precision in real outputs; the sample above uses rounded scales for readability.

## Assumptions

Non-zero page rotation is rejected by default. Rasterization uses PyMuPDF’s `Page.rect` and a uniform DPI matrix—see script docstring for fidelity caveats.
