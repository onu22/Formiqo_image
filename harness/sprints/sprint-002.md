---
sprint_number: sprint-002
status: active
start_date: 2026-07-17
end_date: 2026-07-31
closed_date:
project: formiqo-mvp
goal: Post-MVP M4 — E5 vision QA refine loop; stretch M5 E7 template memory
---

# Sprint 002 Plan

**Project:** Formiqo MVP (post-ship)  
**Sprint Dates:** 2026-07-17 → 2026-07-31  
**Sprint Goal:** Implement E5 vision QA refinement loop; optionally land stretch E7 template memory so automation can continue past G4.

## Milestones

| Milestone | Target | Owner | Exit criteria |
|-----------|--------|-------|---------------|
| M4a E5 refine endpoint | 2026-07-24 | LLM Engineer | `POST /jobs/{id}/refine-grounding` closed loop works |
| M4b E5 auto + UI flags | 2026-07-28 | LLM + FE | Pipeline toggle + `qa_status` surfaced in editor |
| M5 E7 stretch | 2026-07-31 | LLM Engineer | Re-upload reuses corrected template when fingerprint matches |

## Sprint backlog

| ID | Epic | Milestone | Task | Owner | Deps | Status |
|----|------|-----------|------|-------|------|--------|
| T030 | E5 | M4a | Implement stamp → judge → bounded delta → re-stamp loop | LLM | G2, E4 | DONE |
| T031 | E5 | M4a | Persist `qa_status` / confidence; wire `stages.qa_refine` in job.json | LLM | T030 | DONE |
| T032 | E5 | M4a | `POST /api/v1/jobs/{id}/refine-grounding` + config (`grounding_qa_*`) | LLM | T030 | DONE |
| T033 | E5 | M4a | Unit/integration tests + perturbed-bbox fixture convergence | LLM | T032 | DONE |
| T034 | E5 | M4b | Optional auto-run refine as final pipeline stage (config toggle) | LLM | T032 | DONE |
| T035 | E5 | M4b | Surface `qa_status: flagged` in editor ("check this field") | FE | T031 | DONE |
| T036 | E5 | M4b | Document judge cost/latency (tokens per page per iteration) | LLM | T033 | DONE |
| T040 | E7 | M5 | Page fingerprint from normalized detected-line layout | LLM | E6 | TODO |
| T041 | E7 | M5 | Template index under `data/` + reuse on fingerprint match | LLM | T040 | TODO |
| T042 | E7 | M5 | Skip LLM for matched pages; `grounding_source: template` | LLM | T041 | TODO |
| T043 | E7 | M5 | Near-miss / false-positive tests | LLM | T042 | TODO |

Status: `TODO` · `IN_PROGRESS` · `BLOCKED` · `IN_REVIEW` · `DONE`

## Critical path

G4 shipped → T030–T033 (E5) → T034–T036 → T040–T043 (E7 stretch)

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Judge cost / latency | Slow ready path | Config toggle off by default; cheap judge model; crop-only images |
| Over-correction of good bboxes | Worse layouts | Bound deltas; never exceed `grounding_qa_max_bbox_delta_px` |
| E7 false-positive fingerprints | Wrong fields reused | Strict hash; near-miss tests required |

## Definition of done

- [ ] All E5 tasks DONE (M4)
- [ ] E7 stretch DONE or explicitly deferred (remove TODO rows)
- [ ] pytest green; refine path exercised with API keys when available
- [ ] PM summary written for the cycle

## Retrospective

_To be filled when sprint closes._
