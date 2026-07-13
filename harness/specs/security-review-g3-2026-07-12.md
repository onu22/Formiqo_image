# Security Review — E3 Upload & File Serving — 2026-07-12

**Gate:** G3  
**Status:** APPROVED WITH CONDITIONS  
**Reviewer:** Security Reviewer

## Threat model summary

Formiqo MVP is a single-operator deployment with **no authentication**. Attack surfaces reviewed:

| Surface | Risk | Mitigation |
|---------|------|------------|
| `POST /jobs` multipart upload | Malicious PDF, DoS via size | Size cap, magic bytes, XFA rejection, server-generated job UUID |
| Job ID path parameters | Path traversal to read arbitrary files | Strict UUID validation; paths resolved under `data/jobs/{id}/output/` |
| `job.json` artifact pointers | Traversal via tampered manifest | `resolve_under_output_dir` + `relative_to` containment check |
| `GET /jobs` list | Information disclosure | Accepted — all jobs visible to any client on the network |
| StaticFiles `/` | Unintended file exposure | Serves only `frontend/dist/` when present |
| Background pipeline | Resource exhaustion | No rate limit (MVP); disk bounded by operator |

STRIDE: **Spoofing** — N/A (no auth). **Tampering** — PATCH endpoints write only under job output. **Repudiation** — logs only. **Information disclosure** — primary risk; error bodies sanitized in E3 routes. **DoS** — upload size limit; unbounded concurrent jobs accepted for MVP. **Elevation** — no shell execution from uploads reviewed.

## Findings

| ID | Title | Severity | Status |
|----|-------|----------|--------|
| G3-001 | No authentication on any endpoint | Low | Accepted — PRD single-operator MVP |
| G3-002 | Error responses could leak absolute filesystem paths | Medium | **Mitigated** — E3 routes use generic messages; `stamping_config` paths redacted |
| G3-003 | Upload validates filename extension, not Content-Type header | Low | Accepted — `%PDF-` magic bytes enforced |
| G3-004 | No upload rate limiting / concurrent job cap | Low | Accepted — out of MVP scope; document for deployment |
| G3-005 | `job.json` artifact paths trusted from disk | Medium | **Mitigated** — `resolve_under_output_dir` blocks `..`; test in `test_g3_security.py` |
| G3-006 | Pipeline stage errors in `GET /jobs/{id}` may store raw exception text | Low | Open — operator-only; sanitize in hardening sprint |
| G3-007 | Legacy endpoints use unstructured FastAPI `detail` errors | Low | Accepted — deprecated batch/ground routes |
| G3-008 | CORS enabled only when `FORMIQO_CORS_ALLOW_ORIGINS` set | Info | OK — default closed |

## Checklist (G3 gate)

### Upload (`POST /jobs`)
- [x] Enforces `Settings.max_upload_bytes`
- [x] PDF magic-byte validation (`%PDF-`)
- [x] XFA rejection with safe user message (no stack traces in API body)
- [x] Filename sanitized via `Path(name).name` — no path injection

### File serving
- [x] Page image resolves under job `output/`
- [x] Export PDF resolves under job `output/`
- [x] Job ID validated as canonical UUID
- [x] Query `variant` restricted to `source|stamped` regex

### General
- [x] E3 API errors use `{error, message}` without internal paths (after remediation)
- [x] CORS/static serving reviewed — StaticFiles only when `frontend/dist` exists
- [x] Dependencies — no automated CVE scan; FastAPI/Pillow/PyMuPDF pinned in requirements (best effort)

## Conditions for approval

1. **Deployment doc** — Operator must bind to localhost or place behind network ACL; no auth is intentional for MVP.
2. **G3-006** — Optional follow-up: sanitize `job.json` stage error strings before E6 production hardening.
3. **Rate limiting** — Track as post-MVP if exposed beyond single-operator LAN.

## Recommendation

**APPROVED WITH CONDITIONS** — E3 upload validation and file-serving path containment are sufficient for E6 live backend integration on a trusted single-operator network. Unblocks sprint tasks T018–T019 (E6 scaffold with live API).
