# Gate G3 — Security Review

**Owner:** Security Reviewer  
**Blocks:** E6 backend integration (live API + file serving)  
**Status:** APPROVED WITH CONDITIONS

## Scope

E3 upload validation and file-serving endpoints per PRD E3 and [`specs/api-contract.md`](../specs/api-contract.md).

## Checklist

### Upload (`POST /jobs`)
- [x] Enforces `Settings.max_upload_bytes`
- [x] PDF content-type and magic-byte validation
- [x] XFA rejection with safe error message (no stack traces)
- [x] Filename sanitized; no path injection via original name

### File serving
- [x] `GET .../pages/{n}/image` resolves strictly inside job directory
- [x] `GET .../export` resolves strictly inside job directory
- [x] Job ID validated (UUID format); reject traversal sequences
- [x] No arbitrary filesystem read via query parameters

### General
- [x] No secrets or internal paths leaked in error responses (E3 routes)
- [x] CORS/static file serving configuration reviewed for dev vs prod
- [x] Dependencies scanned for known critical CVEs (best effort — manual pin review)

## Threat model summary

Single-operator MVP, no authentication. Attack surfaces: multipart upload, UUID job paths, `job.json` artifact pointers, optional StaticFiles mount. Path containment via `resolve_under_output_dir`. Full STRIDE in [`specs/security-review-g3-2026-07-12.md`](../specs/security-review-g3-2026-07-12.md).

## Verdict

**Status:** APPROVED WITH CONDITIONS

**Signed off by:** Security Reviewer  
**Date:** 2026-07-12

### Findings

| ID | Severity | Status | Notes |
|----|----------|--------|-------|
| G3-001 | Low | Accepted | No auth — PRD single-operator |
| G3-002 | Medium | Mitigated | Generic E3 error messages; paths redacted |
| G3-003 | Low | Accepted | Extension + magic bytes sufficient for MVP |
| G3-004 | Low | Accepted | No rate limit — post-MVP |
| G3-005 | Medium | Mitigated | Artifact path traversal blocked + tested |
| G3-006 | Low | Open | Pipeline stage errors may store raw exception text |
| G3-007 | Low | Accepted | Legacy routes use unstructured errors |
| G3-008 | Info | OK | CORS off by default |

### Conditions

1. Deploy on trusted network (localhost or ACL); document no-auth residual risk.
2. Optional: sanitize `job.json` stage error strings before production hardening (G3-006).
3. **PM:** E6 may integrate against live backend — T018–T019 unblocked for live API wiring.

## Report

Full report: [`specs/security-review-g3-2026-07-12.md`](../specs/security-review-g3-2026-07-12.md)
