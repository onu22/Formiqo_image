---
sprint_number: sprint-001
status: active
start_date: 2026-07-12
end_date: 2026-07-26
closed_date:
project: formiqo-mvp
goal: G1 sign-off plus E1 data cleanup and E3 API skeleton — UI-ready backend (M1)
---

# Sprint 001 Plan

**Project:** Formiqo MVP  
**Sprint Dates:** 2026-07-12 → 2026-07-26  
**Sprint Goal:** Architecture signed off; job manifest and deduplicated schemas landed; upload/list/detail/fields API working over HTTP.

## Milestones

| Milestone | Target | Owner | Exit criteria |
|-----------|--------|-------|---------------|
| M1a G1 sign-off | 2026-07-14 | Solution Architect | G1 APPROVED; specs frozen |
| M1b E1 data cleanup | 2026-07-18 | Backend Developer | job.json, deduped fields, tests green |
| M1c E3 API core | 2026-07-22 | Backend Developer | upload → poll → GET fields integration test |
| M1d E6 scaffold | 2026-07-26 | Frontend Developer | Vite app + routes; mocked API optional |

## Sprint backlog

| ID | Epic | Milestone | Task | Owner | Deps | Status |
|----|------|-----------|------|-------|------|--------|
| T001 | G1 | M1a | Review job-manifest-schema.md | SA | — | DONE |
| T002 | G1 | M1a | Review field-schema.md | SA | — | DONE |
| T003 | G1 | M1a | Review api-contract.md | SA | — | DONE |
| T004 | G1 | M1a | Sign G1-architecture.md | SA | T001–T003 | DONE |
| T005 | E1 | M1b | Implement job.json read/write + status transitions | BE | T004 | DONE |
| T006 | E1 | M1b | Deduplicate field schema in semantic_grounding | BE | T004 | DONE |
| T007 | E1 | M1b | Slim page manifests + detected_lines.json | BE | T004 | DONE |
| T008 | E1 | M1b | Empty-string stamping defaults + stamp run retention | BE | T004 | DONE |
| T009 | E1 | M1b | Update stampers + stamping_config for new schema | BE | T006 | DONE |
| T010 | E1 | M1b | Golden fixture + unit tests for job manifest | BE | T005 | DONE |
| T011 | E3 | M1c | POST /jobs multipart upload + background pipeline | BE | T005 | DONE |
| T012 | E3 | M1c | GET /jobs, GET /jobs/{id} status polling | BE | T005 | DONE |
| T013 | E3 | M1c | GET /jobs/{id}/fields | BE | T009 | DONE |
| T014 | E3 | M1c | PATCH /jobs/{id}/fields and /values | BE | T013 | DONE |
| T015 | E3 | M1c | GET page image + stamp-images + stamp-pdf + export | BE | T013 | DONE |
| T016 | E3 | M1c | httpx integration test happy path | BE | T015 | DONE |
| T017 | E3 | M1c | Path traversal tests for file routes | BE | T015 | DONE |
| T018 | E6 | M1d | Scaffold frontend/ (Vite + React + Tailwind + shadcn) | FE | T003 | DONE |
| T019 | E6 | M1d | Jobs list + upload UI from mockups (mock API) | FE | T018 | DONE |
| T023 | E6 | M1d | Processing/failed/editor views + live API integration (post-G3) | FE | T019 | DONE |
| T024 | E6 | M1d | Editor drag/nudge/save/refresh-preview/export flow + Playwright e2e smoke | FE | T023 | DONE |
| T020 | E2 | — | stamping_common.py extraction | BE | T009 | DONE |
| T021 | E4 | — | Structured outputs for grounding (OpenAI json_schema + Anthropic tool-use) | LLM | T006 | DONE |
| T022 | E4 | — | Parallel per-page grounding (bounded concurrency + per-page isolation) | LLM | T021 | DONE |
| T025 | E4 | — | Anchor-first grounding (cell/line/label anchors) + grounding_source | LLM | T021 | DONE |

## Critical path

T001–T003 → T004 (G1) → T005 → T011 → T016 → G3 → E6 integration (sprint 002)

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| G1 delayed | Blocks all backend epics | SA prioritizes T001–T004 first |
| E1/E4 merge conflict on semantic_grounding.py | Rework | E1 lands before E4 starts |
| E3 without E2 | Preview drift in UI | Document; G2 in sprint 002 |

## Definition of done

- [x] G1 APPROVED WITH CONDITIONS (2026-07-12)
- [x] E1 acceptance criteria from PRD met
- [x] E3 integration test passes
- [x] G4 QA APPROVED (2026-07-14) — MVP ship unblocked
- [ ] Sprint retro filled via `/sprint-plan` close or manual edit

## Retrospective

_To be filled when sprint closes._
