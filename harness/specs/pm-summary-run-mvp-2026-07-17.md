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

## E7 — Template memory

Status: **DONE** (`sprint-002` tasks T040–T043)

The LLM Engineer delivered scale-invariant fingerprints from normalized detected-line
layouts, a durable atomic template index under `data/templates`, corrected-field
capture after editor updates, and template reuse before LLM client construction.
Matched pages retain the corrected layout with `grounding_source: template`.

Acceptance results:

- [x] A re-upload of a corrected form reaches the template path with zero LLM calls.
- [x] Reused fields match the corrected layout.
- [x] Near-miss layouts and sparse pages do not false-positive match.
- [x] The template index is written atomically and can be disabled by configuration.

Verification:

- `python3 -m pytest -q`: **87 passed**, one third-party deprecation warning.
- The E7 suite adds 14 fingerprint, persistence, zero-LLM, and near-miss tests.

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

After E7:

```text
ACTION=complete
TARGET=post-MVP
AGENT=product-manager
REASON=G4 shipped; E5 and E7 complete — post-MVP done
```

Final sprint counts: `DONE=11`, `TODO=0`, `BLOCKED=0`.

## Outcome

**Post-MVP complete.** G4 remains QA APPROVED, required E5 is complete, queued
stretch E7 is complete, and the active post-MVP sprint has no remaining work.
