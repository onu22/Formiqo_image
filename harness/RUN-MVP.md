# Autonomous MVP Conductor Playbook

**Command:** `/run-mvp` in Cursor (skill: [`.cursor/skills/run-mvp/`](../.cursor/skills/run-mvp/SKILL.md))

Runs the Formiqo harness from current state to **G4 QA APPROVED** (MVP ship) without manual `/epic` or `/gate` between steps.

## Machine-readable next action

```bash
./scripts/harness-next.sh          # human-readable
./scripts/harness-next.sh --json   # for agents
./scripts/harness-status.sh        # gates + sprint counts
```

## Pipeline (target end state)

```mermaid
flowchart LR
    G1[G1 APPROVED] --> E1[E1 done]
    E1 --> E2[E2 parity]
    E1 --> E3[E3 API]
    E1 --> E4[E4 grounding]
    E2 --> G2[G2 QA APPROVED]
    G2 --> E5[E5 QA loop]
    E3 --> G3[G3 APPROVED]
    G3 --> E6[E6 UI]
    E4 --> E5
    E6 --> G4[G4 QA APPROVED]
    E5 --> Ship[MVP ship]
    G4 --> Ship
```

## Gate open rules

| Gate | Open when |
|------|-----------|
| G1 | SA signs schemas + API contract |
| G2 | E2 complete; parity golden tests pass |
| G3 | E3 upload + file serving implemented |
| G4 | E6 matches mockups; e2e flow passes |

A gate **blocks** downstream work only as defined in [`specs/delegation-plan-mvp.md`](specs/delegation-plan-mvp.md).

## Autonomous priority (what `/run-mvp` runs next)

When multiple epics are unblocked, pick in this order:

| Priority | Condition | Action |
|----------|-----------|--------|
| 1 | Required gate PENDING and epic work complete | Run gate review |
| 2 | E2 not done (stamping parity) | `/epic E2` — blocks G2 → E5 |
| 3 | E4 not done | `/epic E4` — parallel with E2 |
| 4 | E6 not done and G1 + G3 approved | `/epic E6` — UI toward G4 |
| 5 | E2 done, G2 not QA APPROVED | `/gate G2` |
| 6 | G2 approved, E4 done, E5 not done | `/epic E5` |
| 7 | E6 done, G4 not QA APPROVED | `/gate G4` |
| 8 | All sprint TODO done, G4 approved | **COMPLETE** — MVP ship |
| 9 | Sprint empty but work remains | `/sprint-plan` — extend backlog |

**Stretch:** E7 only after G4 approved.

## Parallelization policy

| Concurrent | Allowed when |
|------------|--------------|
| E2 + E4 | E1 done; E4 does not start until E1 field schema landed |
| E2 + E6 | E6 uses API contract; live API after G3 |
| E4 + E6 | Default after G3 |
| E5 + E6 polish | After G2 + E4 |

**Never parallel:** Two agents editing `semantic_grounding.py` (sequence E1 → E4).

## Human stop conditions

The conductor **stops and asks** only when:

1. Gate verdict is **BLOCKED**
2. Epic acceptance criteria cannot be met without PRD change
3. Merge conflict or test failure after 2 fix attempts
4. Missing secret (API key) for LLM epics — document env vars needed

## Artifacts per cycle

- Sprint task status updates in `harness/sprints/sprint-NNN.md`
- PM summary: `harness/specs/pm-summary-run-mvp-YYYY-MM-DD.md`
- Gate reports when run: `harness/specs/security-review-*.md`, QA reports, etc.

## Cursor Automation

Optional scheduled kickoff: see [`automation/run-mvp-prompt.md`](automation/run-mvp-prompt.md).  
Open Automations UI: run `/run-mvp` once manually, or use the automation template created in-repo.
