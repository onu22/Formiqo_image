# Workflow: Delivery (Formiqo MVP)

Adapted from [freelance-developer-harness](https://github.com/10Legs/freelance-developer-harness).

## Inputs

- PRD epic acceptance criteria (`docs/PRD.md`)
- G1 schemas signed (`harness/specs/`)
- Active sprint (`harness/sprints/CURRENT`)

## Development → QA flow

```
Implementer: completes epic tasks → updates sprint status → "Ready for QA"
    ↓
QA Specialist: validates acceptance criteria / golden tests
    ↓
QA APPROVED → next epic or gate sign-off
QA BLOCKED → back to implementer with issues in qa-report
```

## Gate flow

```
Implementer or PM: `/gate G2`
    ↓
Gate owner reviews criteria in harness/gates/G*.md
    ↓
APPROVED → PM unblocks downstream epics in sprint backlog
BLOCKED → PM marks tasks BLOCKED with reason
```

## Release (MVP ship)

1. G2 parity signed (before E5)
2. G3 security signed (before E6 live integration)
3. E6 complete against mockups
4. G4 ship review
5. PM marks MVP shippable in pm-summary

## Sprint cadence

- Plan: `/sprint-plan`
- Track: edit `harness/sprints/CURRENT` backlog statuses
- Close: update sprint `status: closed`, fill retrospective, plan next sprint
