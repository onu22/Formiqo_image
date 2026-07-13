# Delegation Plan — Formiqo MVP — 2026-07-12

**Vision:** A single operator can upload a flat PDF form, review AI-filled fields in the browser, and export a trustworthy flattened PDF.

**Goal:** All PRD epics E1–E6 complete, gates G1–G4 signed, MVP shippable. E7 is stretch.

**PRD:** [`docs/PRD.md`](../../docs/PRD.md)

---

## Epic routing

| Epic | Owner | Role agent | Depends on | Gate before start | Gate before downstream |
|------|-------|------------|------------|-------------------|------------------------|
| **E1** Data model cleanup | Backend Developer | `backend-developer` | — | **G1** | — |
| **E2** Stamping unification | Backend Developer | `backend-developer` | E1 | G1 | **G2** blocks E5 |
| **E3** API surface for UI | Backend Developer | `backend-developer` | E1 | G1 | **G3** blocks E6 integration |
| **E4** Grounding accuracy | LLM Engineer | `llm-engineer` | E1 | G1 | — |
| **E5** Vision QA refinement | LLM Engineer | `llm-engineer` | E2, E4 | **G2** | — |
| **E6** Review UI frontend | Frontend Developer | `frontend-developer` | E3 contract | G1 (API contract) | **G4** before ship |
| **E7** Template memory | LLM Engineer | `llm-engineer` | E6 | — | stretch |

## Quality gates

| Gate | Owner | Blocks | Artifact |
|------|-------|--------|----------|
| **G1** Architecture sign-off | Solution Architect | E1 start | [`gates/G1-architecture.md`](../gates/G1-architecture.md) |
| **G2** Preview/PDF parity | QA Specialist | E5 start | [`gates/G2-parity.md`](../gates/G2-parity.md) |
| **G3** Security review | Security Reviewer | E6 integration | [`gates/G3-security.md`](../gates/G3-security.md) |
| **G4** Ship review | QA Specialist | MVP ship | [`gates/G4-ship.md`](../gates/G4-ship.md) |

## Critical path

```
G1 sign-off
  → E1 (job manifest, field dedupe)
  → E3 (upload + poll + fields API)     ← M1
  → G3
  → E6 UI (mock API until G3, then integrate)
  → parallel: E2 + E4 after E1
  → G2
  → E5
  → G4
  → Ship
```

## Parallelization

Once **E1** lands, agents may work in parallel:

- **Backend:** E2 (stamping) and E3 (API) — coordinate on shared modules
- **LLM:** E4 (grounding) — rebase on E1 field schema
- **Frontend:** E6 scaffolding against [`api-contract.md`](api-contract.md) after G1; live integration after G3

**Merge rule:** E1 lands first; E4 rebases on E1 before touching `semantic_grounding.py`.

## First actions (PM)

1. Route **Solution Architect** to review and sign **G1** (schemas + API contract).
2. On G1 approval, assign **Backend Developer** to **E1** sprint tasks.
3. In parallel after G1, **Frontend Developer** may scaffold `frontend/` from mockups; no live API until G3.

## Milestone mapping

| Milestone | Epics + gates |
|-----------|---------------|
| M1 UI-ready backend | G1 + E1 + E3 |
| M2 Trustworthy output | E2 + E4 + G2 |
| M3 Usable product | E6 + G3 + G4 |
| M4 Self-correcting | E5 |
| M5 Compounding | E7 |
