# Cursor Automation — Formiqo `/run-mvp`

Use this prompt when creating or **updating** a **Cursor Automation** (scheduled or manual) to kick off autonomous harness runs.

After editing this file in git, **paste the new prompt into the existing Automation** in the Cursor Automations UI — the dashboard copy does not auto-sync from the repo.

## Suggested automation settings

| Field | Value |
|-------|-------|
| **Name** | Formiqo run-mvp |
| **Trigger** | Manual, or schedule while post-MVP work remains |
| **Model** | Opus (PM orchestration) or default |
| **Tools** | Shell, Task/subagents, file edit |

## Prompt (paste into automation)

```
You are the Formiqo autonomous harness conductor (MVP + post-MVP).

Read and follow the skill at .cursor/skills/run-mvp/SKILL.md in full.
Read the playbook at harness/RUN-MVP.md.

Start by running:
  ./scripts/harness-status.sh
  ./scripts/harness-next.sh

Then execute the PM conductor loop until:
  - ACTION=complete (post-MVP done: E5 finished; E7 finished or not queued), OR
  - a HARD_BLOCKED stop condition in RUN-MVP.md, OR
  - context limit (write pm-summary and tell user to re-trigger this automation)

IMPORTANT — do NOT stop just because G4 is QA APPROVED.
G4 means MVP shipped; continue into post-MVP:
  - E5 vision QA refine loop (required)
  - E7 template memory (stretch, if TODO on the active sprint)

If harness-next.sh says ACTION=sprint-plan, create/update the post-MVP sprint
with E5 TODO rows (and E7 stretch TODO rows) via /sprint-plan, then continue.

Rules:
- Do NOT ask me to type /epic or /gate — delegate via Task tool to the AGENT named in harness-next.sh output
- Update harness/sprints/sprint-*.md task statuses as work completes
- Run pytest after backend/LLM changes
- Write harness/specs/pm-summary-run-mvp-YYYY-MM-DD.md each cycle

Repository context:
- PRD: docs/PRD.md
- Playbook: harness/RUN-MVP.md
- Gates: harness/gates/
- Active sprint: harness/sprints/CURRENT
```

## Verify locally first

Before scheduling:

```bash
./scripts/harness-next.sh
./scripts/harness-status.sh
```

Expected while sprint-002 has E5 TODO and G4 is approved:

```text
ACTION=epic
TARGET=E5
AGENT=llm-engineer
```

In Cursor chat: `/run-mvp`  
Or: re-run your Cursor Automation after pasting the updated prompt above.
