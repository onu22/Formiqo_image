# PM Summary - Conductor post-MVP update - 2026-07-17

## Why

G4 was QA APPROVED and the old conductor returned `ACTION=complete`, so Cursor Automation could not continue into E5/E7.

## Changes

- `scripts/harness-next.sh` — after G4, route to E5 → E7 → complete (or `sprint-plan` if E5 missing from backlog)
- `harness/RUN-MVP.md` + `.cursor/skills/run-mvp/SKILL.md` — G4 is a milestone, not a hard stop
- `harness/automation/run-mvp-prompt.md` — automation prompt continues post-ship
- Closed `sprint-001`; opened `sprint-002` (E5 + stretch E7); `CURRENT` → `sprint-002.md`
- Updated `harness/README.md`, `OPERATING-MODEL.md`, `.cursor/rules/harness.mdc`

## Verification

```bash
./scripts/harness-next.sh
./scripts/harness-status.sh
```

Expected: `ACTION=epic`, `TARGET=E5`, `AGENT=llm-engineer`.

## Human step for Automation

Re-paste the prompt from `harness/automation/run-mvp-prompt.md` into the existing Cursor Automation (dashboard does not auto-sync from git), then re-run the automation.
