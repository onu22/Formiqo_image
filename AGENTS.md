# Formiqo MVP — Agent Team Roster

Multi-agent harness for building the Formiqo MVP per [`docs/PRD.md`](docs/PRD.md).
Inspired by [10Legs/freelance-developer-harness](https://github.com/10Legs/freelance-developer-harness).

**Operating model:** [`harness/OPERATING-MODEL.md`](harness/OPERATING-MODEL.md)  
**Delegation plan:** [`harness/specs/delegation-plan-mvp.md`](harness/specs/delegation-plan-mvp.md)  
**Active sprint:** [`harness/sprints/CURRENT`](harness/sprints/CURRENT)

---

## Leadership

### Product Manager
**Authority:** Owns product vision, epic prioritization, agent routing, gate tracking, pipeline continuity  
**Gate:** BLOCKING — No work is scoped or delegated without PM approval  
**Spawning:** First on any new request; orchestrates all councils  
**Agent file:** [`.cursor/agents/product-manager.md`](.cursor/agents/product-manager.md)

Responsibilities:
- Translate PRD epics (E1–E7) into sprint backlog tasks
- Route work to the correct role; never implement technical work directly
- Track gate status (G1–G4); halt downstream epics when a gate is blocked
- Write PM summaries to `harness/specs/pm-summary-*.md`
- Re-route or escalate when dependencies stall

---

## Technical Council

### Solution Architect
**Authority:** System design, schema contracts, ADRs, technical feasibility  
**Gate:** BLOCKING — **G1** architecture sign-off before E1 starts  
**Spawning:** Independent — validates others' proposals  
**Agent file:** [`.cursor/agents/solution-architect.md`](.cursor/agents/solution-architect.md)

Owns sign-off on:
- Job manifest schema (`harness/specs/job-manifest-schema.md`)
- Deduplicated field schema (`harness/specs/field-schema.md`)
- E3 API contract (`harness/specs/api-contract.md`)

---

### Backend Developer
**Authority:** FastAPI services, on-disk job layout, stamping engine, REST API  
**Gate:** NON-BLOCKING — implements approved architecture  
**Epics:** E1 (owner), E2 (owner), E3 (owner)  
**Agent file:** [`.cursor/agents/backend-developer.md`](.cursor/agents/backend-developer.md)

Restrictions:
- Do not deviate from signed G1 schemas or API contract without SA sign-off
- Do not skip gate artifacts when a gate blocks downstream work

---

### LLM Engineer
**Authority:** Grounding pipeline, structured outputs, vision QA loop, template memory  
**Gate:** NON-BLOCKING — implements approved prompts and inference design  
**Epics:** E4 (owner), E5 (owner), E7 stretch (owner)  
**Agent file:** [`.cursor/agents/llm-engineer.md`](.cursor/agents/llm-engineer.md)

Restrictions:
- Prompt changes must be versioned under `prompts/`
- Do not start E5 until G2 (preview/PDF parity) is signed off

---

### Frontend Developer
**Authority:** React review UI, editor interactions, mockup fidelity  
**Gate:** NON-BLOCKING — implements approved API contract and mockups  
**Epics:** E6 (owner)  
**Agent file:** [`.cursor/agents/frontend-developer.md`](.cursor/agents/frontend-developer.md)

Design source of truth: `docs/mockups/*.png` (six screens).  
Restrictions:
- Do not integrate against live backend until G3 security sign-off
- May scaffold against mocked API once G1 API contract is signed

---

## Delivery Council

### QA Specialist
**Authority:** Acceptance criteria validation, golden tests, ship review  
**Gate:** BLOCKING — **G2** (parity) and **G4** (ship)  
**Spawning:** Always independent — never self-review  
**Agent file:** [`.cursor/agents/qa-specialist.md`](.cursor/agents/qa-specialist.md)

Verdicts: `QA APPROVED` | `QA BLOCKED` — no ambiguous states.

---

### Security Reviewer
**Authority:** Upload validation, path traversal, file serving, threat modeling  
**Gate:** BLOCKING — **G3** before E6 backend integration  
**Spawning:** Always independent  
**Agent file:** [`.cursor/agents/security-reviewer.md`](.cursor/agents/security-reviewer.md)

Verdicts: `APPROVED` | `BLOCKED` | `APPROVED WITH CONDITIONS`

---

## Governance Matrix

| Artifact | Author | Sign-off |
|----------|--------|----------|
| PRD / requirements | PM | — (approved) |
| Job manifest + field schemas + API contract | SA | **G1** SA |
| Stamping parity golden tests | BE | **G2** QA |
| Upload + file-serving security | SEC | **G3** SEC |
| Mockup fidelity + e2e flow | FE + QA | **G4** QA |
| ADRs | SA | SA |
| Sprint backlog | PM | PM |
| QA reports | QA | QA |
| Security reports | SEC | SEC |

---

## Epic → Owner Map

| Epic | Owner | Gate to start | Gate before downstream |
|------|-------|---------------|------------------------|
| E1 Data model cleanup | Backend | G1 | — |
| E2 Stamping unification | Backend | E1 | G2 blocks E5 |
| E3 API surface | Backend | E1 | G3 blocks E6 integration |
| E4 Grounding accuracy | LLM | E1 | — |
| E5 Vision QA loop | LLM | E2 + E4 | — |
| E6 Review UI | Frontend | E3 contract (G1) | G4 before ship |
| E7 Template memory | LLM | E6 | stretch |

---

## Model tiers (pinned in `.cursor/agents/`)

| Agent | Tier | Model |
|-------|------|-------|
| Product Manager | Opus | `claude-opus-4-8-thinking-high` |
| Solution Architect | Opus | `claude-opus-4-8-thinking-high` |
| Backend Developer | Sonnet | `claude-sonnet-5-thinking-high` |
| LLM Engineer | Opus | `claude-opus-4-8-thinking-high` |
| Frontend Developer | Sonnet | `claude-sonnet-5-thinking-high` |
| QA Specialist | Fast | `composer-2.5-fast` |
| Security Reviewer | Opus | `claude-opus-4-8-thinking-high` |

Models are set in each agent's YAML frontmatter (`model:` field). Cursor may fall back if a model is unavailable on your plan or requires Max Mode.

## Harness Commands (Cursor skills)

| Skill | Purpose |
|-------|---------|
| `/run-mvp` | Autonomous MVP conductor — gates + epics to G4 ship without manual steps |
| `/pm` | PM orchestration — delegation, routing, gate checks |
| `/epic E1` | Start work on a specific epic with role context |
| `/gate G1` | Run a quality gate review (SA, QA, or SEC by gate) |
| `/sprint-plan` | Plan or update the active sprint |
| `/run` | Start FastAPI + Vite (backend + frontend) |
| `/stop` | Stop both dev servers |
