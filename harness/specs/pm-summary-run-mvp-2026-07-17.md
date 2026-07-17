# PM Summary — Run MVP — 2026-07-17

## Current cycle

Branch: `cursor/harness-conductor-mvp-post-mvp-dd38`

The conductor continued beyond G4 into the active post-MVP sprint, as required by
`harness/RUN-MVP.md`. All gates remain open:

- G1 architecture: APPROVED WITH CONDITIONS
- G2 parity: QA APPROVED
- G3 security: APPROVED WITH CONDITIONS
- G4 ship: QA APPROVED

## E5 — Vision QA refinement loop

Status: **DONE** (`sprint-002` tasks T030–T036)

The LLM Engineer delivered the full stamp → judge → bounded-delta → re-stamp loop,
manual refine endpoint, optional final pipeline stage, persisted QA outcomes and
metrics, perturbed-bbox convergence coverage, cost/latency documentation, and the
editor warning for flagged fields.

Acceptance results:

- [x] Median bbox error cannot increase; per-iteration movement is bounded.
- [x] Convergence, iterations, confidence, token cost, and latency are persisted.
- [x] Deliberately perturbed bboxes are measurably corrected.
- [x] Judge cost/latency is documented in `harness/specs/qa-e5-refine-note.md`.
- [x] Versioned judge prompts and offline deterministic tests are present.

Verification:

- `python3 -m pytest -q`: **73 passed**, one third-party deprecation warning.
- `npm run typecheck && npm run lint && npm run build`: **passed**.
- Live provider judging remains optional and requires an OpenAI or Anthropic API key;
  the refinement algorithm and API behavior are covered with an injected judge.

## Conductor state

Startup:

```text
ACTION=epic
TARGET=E5
AGENT=llm-engineer
REASON=Post-MVP M4 — E5 vision QA refine loop
```

After E5:

```text
ACTION=epic
TARGET=E7
AGENT=llm-engineer
REASON=Post-MVP M5 stretch — E7 template memory
```

Sprint counts after E5: `DONE=7`, `TODO=4`, `BLOCKED=0`.

The conductor is not at a stop condition. E7 is queued and must run before
`ACTION=complete`.
