## Install
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

deactivate
```

- OpenAI: `gpt-5.5`
- Anthropic: `claude-opus-4-7`

Stamping **checkbox** and **radio** fields draws a **vector checkmark** (two strokes inside the bbox), not a text glyph—consistent appearance across fonts and PDF viewers.

### Refine grounding (vision QA loop)

`POST /api/v1/jobs/{job_id}/refine-grounding` runs up to `FORMIQO_GROUNDING_QA_MAX_ITERATIONS` rounds (default 3): stamp previews from `field_grounding/refined/` → vision QA → optional **`page_translation`** then per-field **`bbox_delta`** (clamped by `FORMIQO_GROUNDING_QA_MAX_BBOX_DELTA_PX`) → repeat. Consensus merge options: `FORMIQO_GROUNDING_QA_CONSENSUS_TRANSLATION_ENABLED`, `FORMIQO_GROUNDING_QA_CONSENSUS_MIN_FIELDS`, `FORMIQO_GROUNDING_QA_CONSENSUS_MAX_SPREAD_PX`. Previews and patches live under `field_grounding/qa_refinement/<session_id>/` and `stamped_images/qa_refinement/<session_id>/` (per-iteration `iter_*` plus a **`final/`** folder stamped once from refined after the loop). When every page is **acceptable**, refined JSON is copied onto `field_grounding/page_*.fields.json`.

Stamp routes (`/stamp-images/{provider}`, `/stamp-pdf/{provider}`) and refine-grounding read **`field_grounding/stamping.json`** (values, `require_all_values`, `image_style`) and **`field_grounding/manifest.json`** (provider, model). POST bodies are not used for those endpoints; the stamp URL provider must match the manifest provider.
