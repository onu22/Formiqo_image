# QA Report — G2 Preview/PDF Parity — 2026-07-13

**Gate:** G2  
**Status:** QA APPROVED  
**Pass rate:** 7/7 criteria, 4/4 parity tests, 34/34 full suite

**Reviewed commit:** `e4afd47` (E2 stamping unification)  
**Reviewer:** QA Specialist (independent; did not implement E2)

## Acceptance criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| No duplicated validation/discovery/manifest logic between stampers | PASS | Shared logic lives in `app/services/stamping_common.py` (bbox validation, page discovery, grounding assertion, hex-color validation, JSON loading, stamp-run loop/manifest scaffold, text fitting/wrapping). `image_stamping.py` and `pdf_stamping.py` retain only renderer-specific code (Pillow draw vs PyMuPDF insert). |
| Golden-file test set (≥2 forms, mixed text/checkbox/multiline) | PASS | `tests/fixtures/parity/builder.py` materializes two distinct layouts (`form_a`, `form_b`), each with `text`, `multiline_text`, and `checkbox` fields. Programmatic fixtures keep the suite deterministic across environments. |
| Stamped PNG preview and rasterized PDF match within pixel tolerance | PASS | `test_preview_pdf_parity[form_a|form_b]` compares ink bounding boxes per field with `PIXEL_TOLERANCE = 8` px; preview and rasterized PDF share identical dimensions. |
| Same wrapping, truncation, and font rendering on both paths | PASS | Shared `fit_text_to_width_pt`, `wrap_text_to_width_pt`, and `fit_multiline_to_box_pt` in `stamping_common.py`; DejaVu Sans TTF shipped at `app/assets/fonts/DejaVuSans.ttf` and resolved as `FONT_PATH` for both Pillow (`load_pil_font`) and PDF (`fontfile=` insert). |
| Per-field `font_size_pt` override round-trips through `stamping.json` | PASS | `test_font_size_override_round_trips_from_stamping_json` sets override in `stamping.json` (`font_size_pt + 3` on text field); both paths honor it and produce matching glyph-band heights. |
| `stamp-pdf` reads style from `stamping.json` (not hardcoded) | PASS | `POST /jobs/{job_id}/stamp-pdf` in `app/routers/jobs.py` loads stamping via `_load_stamping`, `stamping_style`, and `stamping_overrides` before calling `run_pdf_stamping_for_job`. No `StampPdfStyle()` hardcoding remains. |
| `multiline_text` wraps identically on both paths | PASS | Parity tests count horizontal ink bands in multiline bboxes; `test_multiline_overflow_truncates_identically` confirms shrink-then-truncate produces a single line on both paths when the box is too small. |

## Tests executed

| Command | Result |
|---------|--------|
| `pytest tests/test_e2_stamping_parity.py -v` | **4 passed** in 1.14s |
| `pytest tests/ -v` | **34 passed**, 1 warning (Starlette `httpx` deprecation) in 2.29s |

### Parity test coverage

| Test | What it validates |
|------|-------------------|
| `test_preview_pdf_parity[form_a]` | Two-form golden set — layout A: text, multiline wrap (≥2 lines), checkbox |
| `test_preview_pdf_parity[form_b]` | Two-form golden set — layout B: different bbox positions/sizes |
| `test_multiline_overflow_truncates_identically` | Overflow parity: shrink-then-ellipsis on a 24px-tall multiline bbox |
| `test_font_size_override_round_trips_from_stamping_json` | `stamping.json` per-field `font_size_pt` override honored on both paths |

## Code review notes

- **Placement parity:** `bottom_padding_pt` uses real font descent metrics instead of the old `rect.y0 + font_size` approximation (PRD E2.4).
- **Font caching:** `load_pil_font` and `_metrics_font` use `functools.lru_cache` (PRD E2.8).
- **API wiring:** `stamp-images` route follows the same `stamping.json` style/overrides loading pattern as `stamp-pdf`.

## Issues

_No blocking or non-blocking issues identified._

## Recommendation

**QA APPROVED** — G2 criteria are met. PM may unblock **E5** (Vision QA refinement loop).
