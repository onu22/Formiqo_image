# Formiqo MVP Harness

Static multi-agent operating system for delivering [`docs/PRD.md`](../docs/PRD.md).

Pattern inspired by [10Legs/freelance-developer-harness](https://github.com/10Legs/freelance-developer-harness):
markdown agents, stop-the-line gates, PM-routed epics, sprint tracking — no runtime framework required.

## Quick start

1. Read [`OPERATING-MODEL.md`](OPERATING-MODEL.md) and [`AGENTS.md`](../AGENTS.md).
2. Check gate status in [`gates/`](gates/) — **G1 must be APPROVED before E1**.
3. Open the active sprint: [`sprints/CURRENT`](sprints/CURRENT).
4. In Cursor chat:
   - **`/run-mvp`** — **Autonomous conductor** — runs epics and gates to MVP ship (sit and wait)
   - **`/pm`** — PM orchestrates next work from the sprint backlog
   - **`/epic E1`** — Start a specific epic with the correct role context
   - **`/gate G1`** — Run architecture sign-off (Solution Architect)
   - **`/sprint-plan`** — Create or refresh sprint plan
   - **`/run`** / **`/stop`** — Dev server

### Sit-and-wait mode

```bash
./scripts/harness-status.sh   # gates + sprint counts
./scripts/harness-next.sh     # next action for /run-mvp
```

Playbook: [`RUN-MVP.md`](RUN-MVP.md)  
Optional Cursor Automation: [`automation/run-mvp-prompt.md`](automation/run-mvp-prompt.md)

## Directory map

| Path | Purpose |
|------|---------|
| `RUN-MVP.md` | Autonomous conductor playbook |
| `automation/run-mvp-prompt.md` | Cursor Automation prompt template |
| `specs/delegation-plan-mvp.md` | Epic → owner → dependency → gate |
| `specs/job-manifest-schema.md` | G1 job.json contract (E1) |
| `specs/field-schema.md` | G1 grounded field contract (E1) |
| `specs/api-contract.md` | G1 REST API for UI (E3) |
| `gates/G*.md` | Stop-the-line sign-off records |
| `sprints/sprint-*.md` | Backlog and status |
| `templates/` | Blank forms for gates, QA, ADRs, sprints |
| `../.cursor/agents/` | Role personas for subagents |
| `../.cursor/skills/` | Slash commands (`/run-mvp`, `/pm`, `/epic`, `/gate`, …) |
| `../scripts/harness-next.sh` | Machine-readable next harness action |

## Current state

- **PRD:** Approved (`docs/PRD.md`)
- **G1:** APPROVED WITH CONDITIONS (2026-07-12)
- **G2:** PENDING — needs E2 parity + golden tests
- **G3:** APPROVED WITH CONDITIONS (2026-07-12)
- **G4:** PENDING — needs E6 + ship review
- **Active sprint:** `sprint-001` (E1/E3 done; E2, E4, E6 TODO)
- **Next `/run-mvp` action:** run `./scripts/harness-next.sh`

## Adding a gate sign-off

1. Copy [`templates/gate-signoff.md`](templates/gate-signoff.md) content into the target gate file, or edit the gate file in place.
2. Set `Status: APPROVED | BLOCKED | APPROVED WITH CONDITIONS`.
3. List criteria checked and any conditions.
4. PM updates sprint tasks and unblocks downstream epics.
