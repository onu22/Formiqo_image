---
name: sprint-plan
description: Plan or create a Formiqo harness sprint — write harness/sprints/sprint-NNN.md and update CURRENT symlink from templates and PRD epics.
---

# Formiqo Sprint Planning

Plan sprint work per [`harness/templates/sprint-plan.md`](../../harness/templates/sprint-plan.md).

## Usage

`/sprint-plan` with optional goal text, e.g. `/sprint-plan M2 stamping and grounding`.

## Steps

1. Read [`docs/PRD.md`](../../docs/PRD.md) epics and [`harness/specs/delegation-plan-mvp.md`](../../harness/specs/delegation-plan-mvp.md).
2. Read current sprint via [`harness/sprints/CURRENT`](../../harness/sprints/CURRENT) if it exists.
3. Determine next sprint number:

```bash
ls harness/sprints/sprint-*.md 2>/dev/null | sed -E 's/.*sprint-0*([0-9]+)\.md/\1/' | sort -n | tail -1
```

4. If user wants a **new** sprint (not edit current):
   - Copy template to `harness/sprints/sprint-NNN.md`
   - Fill frontmatter, milestones, backlog from PRD + remaining work
   - Update symlink:

```bash
cd harness/sprints && ln -sfn sprint-NNN.md CURRENT
```

5. If user wants to **update active sprint**, edit the file `CURRENT` points to.
6. Backlog rows must include: ID, Epic, Milestone, Task, Owner (SA/BE/LLM/FE/QA/SEC), Deps, Status.
7. Map milestones to PRD: M1 (G1+E1+E3), M2 (E2+E4+G2), M3 (E6+G3+G4), M4 (E5), M5 (E7).

## PM rules

- G1 tasks before E1 tasks
- Mark tasks BLOCKED when gate not approved
- Carry incomplete TODO items into next sprint on close

## Close sprint

Set `status: closed`, `closed_date`, fill Retrospective section in sprint file.
