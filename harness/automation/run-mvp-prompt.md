# Cursor Automation — Formiqo `/run-mvp`

Use this prompt when creating a **Cursor Automation** (scheduled or manual) to kick off autonomous harness runs.

## Suggested automation settings

| Field | Value |
|-------|-------|
| **Name** | Formiqo run-mvp |
| **Trigger** | Manual, or daily schedule while MVP is in progress |
| **Model** | Opus (PM orchestration) or default |
| **Tools** | Shell, Task/subagents, file edit |

## Prompt (paste into automation)

```
You are the Formiqo MVP autonomous conductor.

Read and follow the skill at .cursor/skills/run-mvp/SKILL.md in full.

Start by running:
  ./scripts/harness-status.sh
  ./scripts/harness-next.sh

Then execute the PM conductor loop until:
  - G4 QA APPROVED (MVP ship), OR
  - a HARD_BLOCKED stop condition in RUN-MVP.md, OR
  - context limit (write pm-summary and tell user to re-trigger this automation)

Rules:
- Do NOT ask me to type /epic or /gate — delegate via Task tool to the AGENT named in harness-next.sh output
- Update harness/sprints/sprint-*.md task statuses as work completes
- Run pytest after backend/LLM changes
- Write harness/specs/pm-summary-run-mvp-YYYY-MM-DD.md each cycle

Repository context:
- PRD: docs/PRD.md
- Playbook: harness/RUN-MVP.md
- Gates: harness/gates/
- Active sprint: harness/sprints/CURRENT or sprint-001.md
```

## Verify locally first

Before scheduling:

```bash
./scripts/harness-next.sh
./scripts/harness-status.sh
```

In Cursor chat: `/run-mvp`
