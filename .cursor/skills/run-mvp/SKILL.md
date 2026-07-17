---
name: run-mvp
description: Autonomous Formiqo harness conductor — gates and epics through G4 MVP ship, then post-MVP E5 and stretch E7 via automation-friendly PM loop.
---

# Formiqo Autonomous Runner (`/run-mvp`)

Run the harness through **MVP ship (G4)** and **continue into post-MVP** (**E5**, then stretch **E7**) without the user typing `/epic` or `/gate` between steps.

**Playbook:** [`harness/RUN-MVP.md`](../../harness/RUN-MVP.md)  
**Next-action hint:** `./scripts/harness-next.sh`  
**Status:** `./scripts/harness-status.sh`

## On invoke

User message: `/run-mvp` or `/run-mvp continue` (resume after context limit).

### You are the PM conductor

Act as **Product Manager** per [`.cursor/agents/product-manager.md`](../../.cursor/agents/product-manager.md). You **orchestrate only** — delegate implementation to role agents via the **Task** tool.

### Autonomous rules (non-negotiable)

1. **Do not ask** the user to run `/epic` or `/gate` — you run them by delegating to the correct subagent.
2. **Do not pause** between steps unless:
   - A gate is **BLOCKED** and needs a human product decision
   - A task is **BLOCKED** with no automatic resolution
   - **`ACTION=complete`** (post-MVP done — E5 finished; E7 finished or not queued)
   - Context window is exhausted — write resume artifact (below) and tell user to send `/run-mvp continue`
3. **G4 QA APPROVED is not a stop.** After ship, keep looping on E5 → E7 (or `sprint-plan` if post-MVP backlog is missing).
4. **Never bypass gates** — run gate reviews when criteria are met.
5. **Update sprint** task statuses (`IN_PROGRESS` → `DONE` / `BLOCKED`) after each delegation.
6. **Write PM summary** after each major cycle: `harness/specs/pm-summary-run-mvp-<date>.md`.

### Loop (repeat until stop condition)

```
while not POST_MVP_COMPLETE and not HARD_BLOCKED:
  1. Run: ./scripts/harness-next.sh
  2. Read output ACTION, TARGET, AGENT, REASON
  3. If ACTION=complete → stop (post-MVP done; report G4 ship + E5/E7 status)
  4. If ACTION=blocked → stop; report blocker to user
  5. If ACTION=gate → Task(subagent=AGENT, prompt="Run /gate {TARGET} per gate-review skill")
  6. If ACTION=epic → Task(subagent=AGENT, prompt="Run /epic {TARGET} per epic skill; full PRD acceptance criteria")
  7. If ACTION=sprint-plan → Task(subagent=product-manager, prompt="Plan post-MVP sprint via /sprint-plan; include E5 TODO and optional E7 stretch")
  8. Verify: pytest green for backend/LLM changes; update sprint; append pm-summary
  9. Continue immediately to next iteration (no user prompt)
```

### Delegation map

| TARGET | Task `subagent_type` | Skill |
|--------|----------------------|-------|
| G1 | `solution-architect` | gate-review |
| G2, G4 | `qa-specialist` | gate-review |
| G3 | `security-reviewer` | gate-review |
| E1, E2, E3 | `backend-developer` | epic |
| E4, E5, E7 | `llm-engineer` | epic |
| E6 | `frontend-developer` | epic |
| sprint-plan | `product-manager` | sprint-plan |

When spawning Task agents, include in the prompt:

- Read the epic/gate section in `docs/PRD.md`
- Read `harness/RUN-MVP.md` for pipeline position
- Follow the matching skill (`epic`, `gate-review`, or `sprint-plan`)
- Return: acceptance criteria checklist, files changed, blockers

### Parallel work

When `harness-next.sh` reports `PARALLEL= yes` and lists multiple targets, launch **up to 3 Task agents concurrently** if they touch disjoint files:

- **Safe parallel:** E2 + E4 + E6 (coordinate: E4 rebases on E1; no simultaneous edits to `semantic_grounding.py` from E2)
- **Not parallel:** E5 until G2 approved; E6 live API until G3 approved (G3 already signed — live API OK)

### Resume after `/run-mvp continue`

1. Read latest `harness/specs/pm-summary-run-mvp-*.md`
2. Read `harness/sprints/CURRENT` (or latest sprint file)
3. Run `./scripts/harness-next.sh`
4. Continue loop from step 5

### Stop messages

| Outcome | User message |
|---------|----------------|
| **Post-MVP complete** | Link pm-summary; note G4 ship + E5/E7 status |
| **Hard blocked** | Name gate/epic, reason, what human must decide |
| **Context limit** | "Send `/run-mvp continue` to resume" + path to pm-summary |

## Do not

- Implement large code changes directly (delegate to role agents)
- Skip pytest after backend/LLM epics
- Mark gates APPROVED without gate owner review
- Stop solely because G4 is QA APPROVED
- Run E7 before E5 TODO rows are cleared (stretch after M4)
