# PM Summary - Run MVP - 2026-07-13

## Cycle status

The run-mvp conductor is active on branch `cursor/formiqo-mvp-progress-7c96`.

**Outcome:** HARD_BLOCKED by `harness/RUN-MVP.md` human stop condition: missing LLM provider secrets for live LLM epic validation.

Current gates after this cycle:

- G1 architecture: APPROVED WITH CONDITIONS
- G2 parity: QA APPROVED
- G3 security: APPROVED WITH CONDITIONS
- G4 ship: PENDING

Latest `./scripts/harness-next.sh` result:

```text
ACTION=gate
TARGET=G4
AGENT=qa-specialist
PARALLEL=no
REASON=E6 complete — ship review (G4)
```

PM did not delegate G4 because the current environment cannot complete the live upload -> grounding -> review validation path without LLM provider keys.

## Startup commands

Executed successfully:

```bash
./scripts/harness-status.sh
./scripts/harness-next.sh
```

Initial next action was E2, with E4 and E6 considered for parallel work. PM launched E2 and E6 in parallel and held E4 to avoid simultaneous edits to `semantic_grounding.py`.

## Work completed this cycle

- Delegated E2 to Backend Developer.
  - Implemented shared stamping engine in `app/services/stamping_common.py`.
  - Unified preview/PDF style handling around PDF points and per-field `font_size_pt` overrides.
  - Added shipped DejaVu Sans font asset and parity fixtures/tests.
  - Removed E2-scoped dead code/config.
  - Sprint T020 marked DONE.
- Delegated E6 to Frontend Developer.
  - Added `frontend/` Vite + React + TypeScript app with Tailwind-style UI primitives.
  - Implemented jobs list, upload, processing, failed, editor, and export-success states.
  - Wired live backend API client and editor interactions for drag/nudge/save/refresh/export.
  - Adjusted FastAPI SPA fallback routing for deep links without weakening API/file-route security.
  - Sprint T018/T019 marked DONE; T023/T024 added and marked DONE for live integration and editor flow.
- Delegated G2 to QA Specialist.
  - G2 parity gate signed **QA APPROVED** in `harness/gates/G2-parity.md`.
  - QA report written: `harness/specs/qa-report-g2-parity-2026-07-13.md`.
- Delegated E4 to LLM Engineer.
  - Implemented structured OpenAI json_schema and Anthropic tool-use response handling.
  - Added bounded parallel per-page grounding with per-page error isolation and job manifest progress.
  - Added anchor-first grounding support (`cell`, `line_anchor`, `label_anchor`, `pixel`) plus PyMuPDF label anchors and coordinate-grid fallback assets.
  - Persisted `grounding_source` and `confidence` for E5/UI prioritization.
  - Versioned prompts under `prompts/` and added E4 tests.
  - Sprint T021/T022 marked DONE and T025 added/DONE.

## Verification

PM verification after E2/E6 implementation:

```bash
pytest tests/
npm run typecheck
npm run lint
npm run build
```

Results:

- Backend: 34 passed, 1 Starlette/httpx deprecation warning.
- Frontend: typecheck, oxlint, and Vite production build passed.

QA verification for G2:

- `pytest tests/test_e2_stamping_parity.py -v`: 4 passed.
- `pytest tests/ -v`: 34 passed, 1 warning.

PM verification after E4:

```bash
pytest tests/
```

Result:

- Backend/LLM: 51 passed, 1 Starlette/httpx deprecation warning.

## Commits and pushes

- `e4afd47` - Implement E2 parity and E6 review UI (pushed to `origin/cursor/formiqo-mvp-progress-7c96`)
- `579974e` - Approve G2 parity gate (pushed to `origin/cursor/formiqo-mvp-progress-7c96`)
- `5f181db` - Implement E4 grounding accuracy (pushed to `origin/cursor/formiqo-mvp-progress-7c96`)

This PM summary update is pending the final commit in the conductor loop.

## Hard block

`harness/RUN-MVP.md` lists "Missing secret (API key) for LLM epics — document env vars needed" as a human stop condition.

E4 deterministic/unit-testable work is complete, but the E4 QA note still has two unchecked live-validation items:

- Zero JSON parse failures across the regression set.
- Live before/after median/p90 bbox center-error numbers.

Required environment variables for a keyed validation run:

- `FORMIQO_OPENAI_API_KEY`
- `FORMIQO_ANTHROPIC_API_KEY`

Once keys are available, re-trigger `/run-mvp continue`. The conductor should:

1. Run the keyed E4 regression validation and update `harness/specs/e4-grounding-accuracy-note-2026-07-13.md`.
2. Re-run `./scripts/harness-next.sh`.
3. If G4 remains next, delegate G4 to QA Specialist for ship review.
