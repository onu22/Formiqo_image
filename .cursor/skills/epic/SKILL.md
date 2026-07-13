---
name: epic
description: Start work on a specific Formiqo PRD epic (E1-E7) with the correct role agent, gate checks, and acceptance criteria from docs/PRD.md.
---

# Formiqo Epic Runner

Run a single PRD epic with the owning role.

## Usage

`/epic E1` — user provides epic id: E1, E2, E3, E4, E5, E6, or E7.

## Steps

1. Parse epic id from user message (e.g. `E1`, `e3`).
2. Read the epic section in [`docs/PRD.md`](../../docs/PRD.md).
3. Read [`harness/specs/delegation-plan-mvp.md`](../../harness/specs/delegation-plan-mvp.md) for owner and dependencies.
4. **Gate check** before coding:

   | Epic | Required gate / dependency |
   |------|---------------------------|
   | E1 | G1 APPROVED |
   | E2 | E1 complete |
   | E3 | E1 complete (G1) |
   | E4 | E1 complete |
   | E5 | G2 APPROVED + E2 + E4 |
   | E6 | G1 API contract; live API needs G3 |
   | E7 | E6 (stretch) |

   If blocked, report blocker and stop.

5. Adopt the role from [`.cursor/agents/`](../agents/):

   | Epic | Agent file |
   |------|------------|
   | E1, E2, E3 | `backend-developer.md` |
   | E4, E5, E7 | `llm-engineer.md` |
   | E6 | `frontend-developer.md` |

6. Read G1 specs when touching schemas or API: [`harness/specs/`](../../harness/specs/).
7. Implement per PRD acceptance criteria; run tests.
8. Update matching tasks in [`harness/sprints/CURRENT`](../../harness/sprints/CURRENT) to `IN_PROGRESS` then `DONE` or `BLOCKED`.
9. Report completion against each acceptance criterion checkbox.

## Parallel epics

E2, E3, E4 may run in parallel after E1 — warn if shared files (`semantic_grounding.py`) conflict.
