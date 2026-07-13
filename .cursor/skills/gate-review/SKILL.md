---
name: gate-review
description: Run a Formiqo quality gate review (G1 architecture, G2 parity, G3 security, G4 ship) and update harness/gates sign-off files.
---

# Formiqo Gate Review

Execute a stop-the-line gate per [`harness/OPERATING-MODEL.md`](../../harness/OPERATING-MODEL.md).

## Usage

`/gate G1` — user provides gate id: G1, G2, G3, or G4.

## Gate → role

| Gate | Owner | Agent | Gate file |
|------|-------|-------|-----------|
| G1 | Solution Architect | `solution-architect.md` | `harness/gates/G1-architecture.md` |
| G2 | QA Specialist | `qa-specialist.md` | `harness/gates/G2-parity.md` |
| G3 | Security Reviewer | `security-reviewer.md` | `harness/gates/G3-security.md` |
| G4 | QA Specialist | `qa-specialist.md` | `harness/gates/G4-ship.md` |

## Steps

1. Read the gate file and its criteria checklist.
2. Adopt the owner role from [`.cursor/agents/`](../agents/).
3. For **G1**: review [`harness/specs/job-manifest-schema.md`](../../harness/specs/job-manifest-schema.md), [`field-schema.md`](../../harness/specs/field-schema.md), [`api-contract.md`](../../harness/specs/api-contract.md) against [`docs/PRD.md`](../../docs/PRD.md) and existing `app/` code.
4. For **G2**: run or verify parity golden tests; compare PNG vs PDF raster output.
5. For **G3**: review upload and file-serving code; test path traversal; write security report template.
6. For **G4**: mockup walkthrough + e2e on real form + pytest.
7. Update gate file:
   - Set **Status** to APPROVED / QA APPROVED / BLOCKED / etc.
   - Fill sign-off name and date
   - List conditions or failures
8. If APPROVED, tell PM which sprint tasks to unblock (see [`harness/sprints/CURRENT`](../../harness/sprints/CURRENT)).

## Verdicts

- G1, G3: `APPROVED` | `APPROVED WITH CONDITIONS` | `BLOCKED`
- G2, G4: `QA APPROVED` | `QA BLOCKED`

Never mark APPROVED without checking every criterion or documenting explicit waivers.
