# Formiqo MVP — Harness Operating Model

You are operating inside a **structured multi-agent harness** to deliver the Formiqo MVP
defined in [`docs/PRD.md`](../docs/PRD.md).

Inspired by [10Legs/freelance-developer-harness](https://github.com/10Legs/freelance-developer-harness).

## Core Philosophy

1. **PRD is law** — Scope, acceptance criteria, and non-goals come from the PRD unless PM explicitly amends them in writing.
2. **Gates are stop-the-line** — Downstream epics do not start until the blocking gate is signed off in `harness/gates/`.
3. **PM directs; specialists implement** — The Product Manager orchestrates; role agents write code and artifacts.
4. **Artifacts are the audit trail** — Schemas, ADRs, gate sign-offs, QA reports, and sprint updates are first-class deliverables.
5. **Coordinate system is sacred** — All field bboxes live in top-left pixel space of 200 DPI page PNGs; see PRD §2.2.

## Team Structure

The **Product Manager** leads. Work flows through PM routing to the Technical and Delivery councils.

```
User request
  → PM (assess, delegate, track gates)
  → Technical Council (SA → BE / LLM / FE)
  → Delivery Council (QA, SEC)
  → Ship (G4)
```

See [`AGENTS.md`](../AGENTS.md) for the full roster.

## Pipeline Stages

```
G1 Architecture sign-off
  → E1 Data cleanup
  → E2 | E3 | E4 (parallel)
  → G2 Parity (blocks E5)
  → G3 Security (blocks E6 integration)
  → E5 QA refine loop
  → E6 Review UI
  → G4 Ship review
  → MVP ship
  → E7 Template memory (stretch)
```

## Non-Negotiable Gates

| Gate | Owner | Blocks | Sign-off file |
|------|-------|--------|---------------|
| **G1** | Solution Architect | E1, E2, E3, E4 | [`harness/gates/G1-architecture.md`](gates/G1-architecture.md) |
| **G2** | QA Specialist | E5 | [`harness/gates/G2-parity.md`](gates/G2-parity.md) |
| **G3** | Security Reviewer | E6 backend integration | [`harness/gates/G3-security.md`](gates/G3-security.md) |
| **G4** | QA Specialist | MVP ship | [`harness/gates/G4-ship.md`](gates/G4-ship.md) |

A gate is **open** only when its sign-off file has `Status: APPROVED` and a dated signature line.

## Workspace Layout

```
docs/PRD.md                    # Product requirements (source of truth for scope)
docs/mockups/                  # UI design source of truth (E6)
harness/
  OPERATING-MODEL.md           # This file
  specs/
    delegation-plan-mvp.md     # Epic routing table
    job-manifest-schema.md     # G1 — job.json contract
    field-schema.md            # G1 — grounded field contract
    api-contract.md            # G1 — REST API for UI
    adr-*.md                   # Architecture decision records
    pm-summary-*.md            # PM run summaries
  gates/
    G1-architecture.md         # Gate sign-offs
    G2-parity.md
    G3-security.md
    G4-ship.md
  sprints/
    sprint-NNN.md              # Sprint backlog + status
    CURRENT → sprint-NNN.md  # Symlink to active sprint
  templates/                   # Copy before filling
app/                           # FastAPI backend (implementation)
frontend/                      # React UI (E6 — not yet created)
prompts/                       # LLM prompts (versioned)
tests/                         # pytest suite
```

## Automated Workflow (PM)

When the PM receives work:

1. **Assess** — New epic, gate review, sprint planning, or bugfix?
2. **Check gates** — Is the requested work blocked? If yes, stop and report the blocker.
3. **Load context** — Read PRD epic section, active sprint (`harness/sprints/CURRENT`), relevant gate files.
4. **Delegate** — Spawn or instruct the owning role agent; update sprint backlog status.
5. **Verify** — When work completes, check acceptance criteria from the PRD and epic.
6. **Advance** — Update gate artifacts if a gate review was requested; unblock downstream epics.
7. **Summarize** — Write `harness/specs/pm-summary-<topic>-<date>.md` for substantive sessions.

## User Story Format

```
As a [user type],
I want to [action],
So that [outcome].

Acceptance Criteria:
- [ ] criterion from PRD epic
```

## Sprint Conventions

- Sprint files: `harness/sprints/sprint-NNN.md` with YAML frontmatter
- Active sprint: `harness/sprints/CURRENT` symlink
- Task status: `TODO` · `IN_PROGRESS` · `BLOCKED` · `IN_REVIEW` · `DONE`
- Every task row references an **Epic** column (E1–E7) and optional **Gate**

## Role Agent Invocation

Cursor subagent definitions live in `.cursor/agents/`. When working an epic:

1. Read the epic section in `docs/PRD.md`
2. Read the role's agent file in `.cursor/agents/`
3. Read G1 specs if touching schemas or API
4. Implement; run tests; update sprint backlog
5. Signal "Ready for QA" or request gate review via `/gate`

## Milestones (from PRD)

| Milestone | Contents | Outcome |
|-----------|----------|---------|
| M1 UI-ready backend | G1 + E1 + E3 | Upload, status, fields API over HTTP |
| M2 Trustworthy output | E2 + E4 + G2 | Preview matches export; faster grounding |
| M3 Usable product | E6 + G3 + G4 | Full browser flow |
| M4 Self-correcting | E5 | Vision QA loop |
| M5 Compounding | E7 | Template memory (stretch) |

## Escalation

If a gate is blocked:
1. Name the blocker (missing spec, failing test, ambiguous requirement)
2. Do not bypass — PM re-routes or requests human decision
3. Record `BLOCKED` on affected sprint tasks with reason
