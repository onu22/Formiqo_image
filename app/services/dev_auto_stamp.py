"""Dev/test: fill stamping values from fixtures and stamp after grounding.

Does not change grounding or stamping algorithms. Opt-in via
``FORMIQO_DEV_AUTO_STAMP_AFTER_GROUNDING=true``.

Fixture formats (extensible for other forms)
--------------------------------------------
1. Mapped nested profile::

    {
      "id": "imm5645e-okafor",
      "match": ["imm5645e", "IMM5645"],
      "mapper": "imm5645e",
      "data": { ... nested profile ... }
    }

2. Flat field_id → string (any form; no mapper needed)::

    {
      "id": "my-form-sample",
      "match": ["formcode"],
      "values": { "field_id": "value", ... }
    }

3. Bare flat map (whole file is ``values``)::

    { "field_id": "value", ... }

Add a new form by dropping a fixture under ``fixtures/applicant-data/``.
Register a nested mapper in ``MAPPERS`` only when the nested shape needs it;
otherwise use format (2) or (3).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.fields_service import patch_values
from app.services.stamping_config import (
    load_job_grounding_info,
    load_stamping_json_parsed,
    stamping_overrides,
    stamping_style,
)

LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MapperFn = Callable[[Mapping[str, Any]], dict[str, str]]

_PERSON_SCALAR_KEYS: tuple[tuple[str, str], ...] = (
    ("name", "name"),
    ("relationship", "relationship"),
    ("dateOfBirth", "date_of_birth"),
    ("countryOfBirth", "country_of_birth"),
    ("presentAddress", "present_address"),
    ("presentOccupation", "present_occupation"),
    ("maritalStatus", "marital_status"),
)


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _apply_person(prefix: str, person: Mapping[str, Any], out: dict[str, str]) -> None:
    if not isinstance(person, Mapping):
        return
    for src, suffix in _PERSON_SCALAR_KEYS:
        if src in person and person[src] is not None:
            out[f"{prefix}_{suffix}"] = _as_str(person[src])
    if "willAccompanyToCanada" in person and person["willAccompanyToCanada"] is not None:
        yes = bool(person["willAccompanyToCanada"])
        out[f"{prefix}_will_accompany_yes"] = "true" if yes else ""
        out[f"{prefix}_will_accompany_no"] = "" if yes else "true"


def map_imm5645e(data: Mapping[str, Any]) -> dict[str, str]:
    """Nested family profile → IMM 5645E grounded ``field_id`` values."""
    out: dict[str, str] = {}
    section_a = data.get("sectionA")
    if isinstance(section_a, Mapping):
        for key, prefix in (
            ("applicant", "section_a_applicant"),
            ("spouse", "section_a_spouse_or_common_law_partner"),
            ("mother", "section_a_mother"),
            ("father", "section_a_father"),
        ):
            person = section_a.get(key)
            if isinstance(person, Mapping):
                _apply_person(prefix, person, out)

    section_b = data.get("sectionB")
    if isinstance(section_b, Mapping):
        children = section_b.get("children")
        if isinstance(children, list):
            for i, child in enumerate(children, start=1):
                if isinstance(child, Mapping):
                    _apply_person(f"section_b_child_{i}", child, out)

    section_c = data.get("sectionC")
    if isinstance(section_c, Mapping):
        siblings = section_c.get("siblings")
        if isinstance(siblings, list):
            for i, sibling in enumerate(siblings, start=1):
                if isinstance(sibling, Mapping):
                    _apply_person(f"sibling_{i}", sibling, out)

    section_d = data.get("sectionD")
    if isinstance(section_d, Mapping):
        cert = section_d.get("certification")
        if isinstance(cert, Mapping):
            if cert.get("signature") is not None:
                out["certification_signature"] = _as_str(cert["signature"])
            if cert.get("date") is not None:
                out["certification_date"] = _as_str(cert["date"])

    return out


def _apply_yes_no(*, yes: bool, yes_key: str, no_key: str, out: dict[str, str]) -> None:
    out[yes_key] = "true" if yes else ""
    out[no_key] = "" if yes else "true"


def map_company_candidate_qual(data: Mapping[str, Any]) -> dict[str, str]:
    """Nested company/candidate/qualifications profile → grounded field ids.

    Used for forms like ``4abad1c6-61dc-47d4-b17e-afdb2b12577a.pdf``.
    """
    out: dict[str, str] = {}

    company = data.get("company")
    if isinstance(company, Mapping):
        for src, field_id in (
            ("companyName", "company_name"),
            ("email", "email"),
            ("agentName", "agent_name"),
            ("mobileNumber", "mobile_no_company"),
        ):
            if src in company and company[src] is not None:
                out[field_id] = _as_str(company[src])

    candidate = data.get("candidate")
    if isinstance(candidate, Mapping):
        for src, field_id in (
            ("surname", "surname"),
            ("fullNames", "full_names"),
            ("maidenName", "maiden_name"),
            ("dateOfBirth", "date_of_birth"),
            ("idNumber", "id_number_or_identifier"),
            ("identifierDescription", "description_of_identifier"),
            ("physicalAddress", "physical_address"),
            ("mobileNumber", "mobile_number"),
        ):
            if src in candidate and candidate[src] is not None:
                out[field_id] = _as_str(candidate[src])
        # Grounding also exposes candidate_mobile_number separately.
        if "mobileNumber" in candidate and candidate["mobileNumber"] is not None:
            out["candidate_mobile_number"] = _as_str(candidate["mobileNumber"])
        if "previousChargesOrConvictions" in candidate and candidate["previousChargesOrConvictions"] is not None:
            _apply_yes_no(
                yes=bool(candidate["previousChargesOrConvictions"]),
                yes_key="previous_charges_yes",
                no_key="previous_charges_no",
                out=out,
            )
        details = candidate.get("convictionDetails")
        if isinstance(details, Mapping):
            for src, field_id in (
                ("dateConvicted", "date_convicted"),
                ("offence", "offence"),
                ("sentence", "sentence"),
            ):
                if src in details and details[src] is not None:
                    out[field_id] = _as_str(details[src])

    qualifications = data.get("qualifications")
    if isinstance(qualifications, list):
        for i, qual in enumerate(qualifications, start=1):
            if not isinstance(qual, Mapping):
                continue
            for src, suffix in (
                ("qualificationName", "qualification_name"),
                ("institutionName", "institution_name"),
                ("dateObtained", "date_obtained"),
                ("studentNumber", "student_no"),
                ("certificateNumber", "certificate_no"),
                ("examNumber", "exam_no"),
            ):
                if src in qual and qual[src] is not None:
                    out[f"{suffix}_{i}"] = _as_str(qual[src])

    signatures = data.get("signatures")
    if isinstance(signatures, Mapping):
        for src, field_id in (
            ("candidateSignature", "candidate_signature"),
            ("candidateMobileNumber", "candidate_mobile_number"),
            ("companyAgentSignature", "company_agent_signature"),
            ("companyDate", "company_agent_date"),
        ):
            if src in signatures and signatures[src] is not None:
                out[field_id] = _as_str(signatures[src])
        # Grounding exposes both candidate_date and candidate_signature_date.
        if "candidateDate" in signatures and signatures["candidateDate"] is not None:
            date_val = _as_str(signatures["candidateDate"])
            out["candidate_date"] = date_val
            out["candidate_signature_date"] = date_val

    # consent.forcedOrCoerced has no grounded field_id on this form yet; ignored safely.

    return out


def map_labeled_questions(data: Mapping[str, Any]) -> dict[str, str]:
    """Map ``questions[{label, value}]`` via ``label_to_field`` to grounded field ids.

    Example ``data``::

        {
          "label_to_field": {"First name": "first_name", ...},
          "questions": [{"label": "First name", "value": "Ada"}, ...]
        }
    """
    out: dict[str, str] = {}
    raw_map = data.get("label_to_field")
    if not isinstance(raw_map, Mapping):
        raise ValueError("labeled_questions mapper requires data.label_to_field object")

    # Exact + casefold lookup
    label_to_field: dict[str, str] = {}
    label_to_field_cf: dict[str, str] = {}
    for label, field_id in raw_map.items():
        if not isinstance(label, str) or not isinstance(field_id, str):
            continue
        label_to_field[label] = field_id
        label_to_field_cf[label.casefold()] = field_id

    questions = data.get("questions")
    if not isinstance(questions, list):
        return out

    for item in questions:
        if not isinstance(item, Mapping):
            continue
        label = item.get("label")
        if not isinstance(label, str):
            continue
        field_id = label_to_field.get(label) or label_to_field_cf.get(label.casefold())
        if not field_id:
            continue
        if "value" in item and item["value"] is not None:
            out[field_id] = _as_str(item["value"])
    return out


def _set_aliases(out: dict[str, str], value: str, *field_ids: str) -> None:
    """Write the same value under several candidate field_ids; unknown ones are filtered later."""
    for field_id in field_ids:
        out[field_id] = value


def map_i765(data: Mapping[str, Any]) -> dict[str, str]:
    """USCIS Form I-765 nested profile → likely grounded ``field_id`` values.

    Emits a few snake_case aliases per logical field so the first grounding run
    can still match via ``filter_to_known_fields``.
    """
    out: dict[str, str] = {}

    part1 = data.get("part1")
    if isinstance(part1, Mapping):
        reason = part1.get("reasonForApplying")
        if isinstance(reason, Mapping):
            selected = reason.get("selectedOption")
            initial = bool(reason.get("initialPermission")) if "initialPermission" in reason else selected == "initialPermission"
            replacement = (
                bool(reason.get("replacementOrCorrection"))
                if "replacementOrCorrection" in reason
                else selected == "replacementOrCorrection"
            )
            renewal = bool(reason.get("renewal")) if "renewal" in reason else selected == "renewal"
            _set_aliases(
                out,
                "true" if initial else "",
                "initial_permission",
                "reason_initial_permission",
                "part1_initial_permission",
                "part1_1a_initial_permission",
            )
            _set_aliases(
                out,
                "true" if replacement else "",
                "replacement_or_correction",
                "reason_replacement_or_correction",
                "part1_replacement_or_correction",
                "part1_1b_replacement_or_correction",
            )
            _set_aliases(
                out,
                "true" if renewal else "",
                "renewal",
                "reason_renewal",
                "part1_renewal",
                "part1_1c_renewal",
            )

    part2 = data.get("part2")
    if isinstance(part2, Mapping):
        legal = part2.get("fullLegalName")
        if isinstance(legal, Mapping):
            if legal.get("familyName") is not None:
                _set_aliases(
                    out,
                    _as_str(legal["familyName"]),
                    "family_name",
                    "full_legal_name_family_name",
                    "part2_family_name",
                    "part2_1a_family_name",
                )
            if legal.get("givenName") is not None:
                _set_aliases(
                    out,
                    _as_str(legal["givenName"]),
                    "given_name",
                    "full_legal_name_given_name",
                    "part2_given_name",
                    "part2_1b_given_name",
                )
            if legal.get("middleName") is not None:
                _set_aliases(
                    out,
                    _as_str(legal["middleName"]),
                    "middle_name",
                    "full_legal_name_middle_name",
                    "part2_middle_name",
                    "part2_1c_middle_name",
                )
        others = part2.get("otherNamesUsed")
        if isinstance(others, list):
            for i, name in enumerate(others, start=1):
                if not isinstance(name, Mapping):
                    continue
                if name.get("familyName") is not None:
                    _set_aliases(
                        out,
                        _as_str(name["familyName"]),
                        f"other_name_{i}_family_name",
                        f"other_names_{i}_family_name",
                        f"part2_other_name_{i}_family_name",
                    )
                if name.get("givenName") is not None:
                    _set_aliases(
                        out,
                        _as_str(name["givenName"]),
                        f"other_name_{i}_given_name",
                        f"other_names_{i}_given_name",
                        f"part2_other_name_{i}_given_name",
                    )
                if name.get("middleName") is not None:
                    _set_aliases(
                        out,
                        _as_str(name["middleName"]),
                        f"other_name_{i}_middle_name",
                        f"other_names_{i}_middle_name",
                        f"part2_other_name_{i}_middle_name",
                    )

    uscis = data.get("uscisUseOnly")
    if isinstance(uscis, Mapping):
        for src, aliases in (
            (
                "authorizationValidFrom",
                (
                    "authorization_valid_from",
                    "uscis_authorization_valid_from",
                ),
            ),
            (
                "authorizationValidThrough",
                (
                    "authorization_valid_through",
                    "uscis_authorization_valid_through",
                ),
            ),
            (
                "alienRegistrationNumber",
                (
                    "alien_registration_number",
                    "uscis_alien_registration_number",
                    "a_number",
                ),
            ),
            ("remarks", ("remarks", "uscis_remarks")),
        ):
            if src in uscis and uscis[src] is not None:
                _set_aliases(out, _as_str(uscis[src]), *aliases)

    rep = data.get("representative")
    if isinstance(rep, Mapping):
        if "formG28Attached" in rep and rep["formG28Attached"] is not None:
            yes = bool(rep["formG28Attached"])
            _set_aliases(
                out,
                "true" if yes else "",
                "form_g28_attached",
                "g28_attached",
                "attorney_form_g28_attached",
            )
            # If the form uses yes/no pair checkboxes:
            out["form_g28_attached_yes"] = "true" if yes else ""
            out["form_g28_attached_no"] = "" if yes else "true"
        if rep.get("attorneyStateBarNumber") is not None:
            _set_aliases(
                out,
                _as_str(rep["attorneyStateBarNumber"]),
                "attorney_state_bar_number",
                "state_bar_number",
            )
        if rep.get("uscisOnlineAccountNumber") is not None:
            _set_aliases(
                out,
                _as_str(rep["uscisOnlineAccountNumber"]),
                "uscis_online_account_number",
                "online_account_number",
            )

    return out


MAPPERS: dict[str, MapperFn] = {
    "imm5645e": map_imm5645e,
    "company_candidate_qual": map_company_candidate_qual,
    "labeled_questions": map_labeled_questions,
    "i765": map_i765,
}


def register_mapper(name: str, fn: MapperFn) -> None:
    """Register or replace a nested-profile mapper for future form fixtures."""
    key = name.strip().lower()
    if not key:
        raise ValueError("mapper name must be non-empty")
    MAPPERS[key] = fn


def _load_json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Fixture must be a JSON object: {path}")
    return raw


def _is_flat_string_map(obj: Mapping[str, Any]) -> bool:
    if not obj:
        return False
    reserved = {"id", "match", "mapper", "data", "values"}
    if any(k in reserved for k in obj):
        return False
    return all(isinstance(v, (str, int, float, bool)) or v is None for v in obj.values())


def fixture_to_values(payload: Mapping[str, Any]) -> dict[str, str]:
    """Resolve a fixture envelope (or bare flat map) to ``field_id → str``."""
    if "values" in payload:
        values = payload["values"]
        if not isinstance(values, Mapping):
            raise ValueError("fixture.values must be an object")
        return {str(k): _as_str(v) for k, v in values.items()}

    mapper_name = payload.get("mapper")
    if isinstance(mapper_name, str) and mapper_name.strip():
        key = mapper_name.strip().lower()
        fn = MAPPERS.get(key)
        if fn is None:
            known = ", ".join(sorted(MAPPERS)) or "(none)"
            raise ValueError(f"Unknown fixture mapper {mapper_name!r}. Known: {known}")
        data = payload.get("data")
        if data is None:
            data = {k: v for k, v in payload.items() if k not in ("id", "match", "mapper", "values")}
        if not isinstance(data, Mapping):
            raise ValueError("fixture.data must be an object when mapper is set")
        return fn(data)

    if _is_flat_string_map(payload):
        return {str(k): _as_str(v) for k, v in payload.items()}

    raise ValueError(
        "Fixture must have 'values', a registered 'mapper'+'data', or be a flat field_id map"
    )


def resolve_fixture_path(
    settings: Settings,
    *,
    source_filename: str | None = None,
) -> Path | None:
    """Pick the fixture file from explicit path or directory match rules."""
    explicit = (settings.dev_auto_stamp_fixture or "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return path if path.is_file() else path

    fixture_dir = settings.dev_auto_stamp_fixture_dir
    if not fixture_dir.is_absolute():
        fixture_dir = _PROJECT_ROOT / fixture_dir
    if not fixture_dir.is_dir():
        return None

    haystack = (source_filename or "").lower()
    candidates = sorted(fixture_dir.glob("*.json"))
    if not haystack:
        return candidates[0] if len(candidates) == 1 else None

    for path in candidates:
        try:
            payload = _load_json_object(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        match_list = payload.get("match")
        if isinstance(match_list, list):
            for token in match_list:
                if isinstance(token, str) and token.strip() and token.strip().lower() in haystack:
                    return path
        # Filename stem fallback: imm5645e-okafor.json matches imm5645e-1.pdf
        stem = path.stem.lower()
        for part in stem.split("-"):
            if len(part) >= 4 and part in haystack:
                return path
    return None


def filter_to_known_fields(values: Mapping[str, str], known_field_ids: set[str]) -> dict[str, str]:
    """Keep only keys present in grounded ``stamping.json`` (safe across forms)."""
    return {k: v for k, v in values.items() if k in known_field_ids}


def run_dev_auto_stamp_after_grounding(
    *,
    job_id: str,
    job_root: Path,
    input_pdf: Path,
    output_dir: Path,
    settings: Settings,
    source_filename: str,
) -> dict[str, Any] | None:
    """Apply fixture values and stamp. Returns a summary dict, or None if skipped."""
    if not settings.dev_auto_stamp_after_grounding:
        return None

    mode = (settings.dev_auto_stamp_mode or "both").strip().lower()
    if mode not in ("images", "pdf", "both"):
        LOG.warning("dev_auto_stamp: invalid mode %r; skipping job_id=%s", mode, job_id)
        return None

    fixture_path = resolve_fixture_path(settings, source_filename=source_filename)
    if fixture_path is None:
        LOG.warning(
            "dev_auto_stamp: no fixture matched source=%r dir=%s; skipping job_id=%s",
            source_filename,
            settings.dev_auto_stamp_fixture_dir,
            job_id,
        )
        return None
    if not fixture_path.is_file():
        LOG.warning("dev_auto_stamp: fixture not found %s; skipping job_id=%s", fixture_path, job_id)
        return None

    payload = _load_json_object(fixture_path)
    mapped = fixture_to_values(payload)
    stamping = load_stamping_json_parsed(output_dir)
    known = set(stamping.values.keys())
    filtered = filter_to_known_fields(mapped, known)
    skipped = sorted(set(mapped) - known)

    if not filtered:
        LOG.warning(
            "dev_auto_stamp: fixture %s produced no matching field_ids for job_id=%s",
            fixture_path.name,
            job_id,
        )
        return {
            "skipped": True,
            "reason": "no_matching_fields",
            "fixture": str(fixture_path),
            "unmapped_keys": skipped,
        }

    patch_values(output_dir=output_dir, values=filtered)
    stamping = load_stamping_json_parsed(output_dir)
    style = stamping_style(stamping)
    overrides = stamping_overrides(stamping)
    provider, model = load_job_grounding_info(job_root)

    summary: dict[str, Any] = {
        "fixture": str(fixture_path),
        "fixture_id": payload.get("id"),
        "applied_count": len(filtered),
        "unmapped_keys": skipped,
        "mode": mode,
        "images": None,
        "pdf": None,
    }

    if mode in ("images", "both"):
        from app.services.image_stamping import run_image_stamping_for_job

        summary["images"] = run_image_stamping_for_job(
            job_id=job_id,
            output_dir=output_dir,
            provider=provider,
            model=model,
            values=stamping.values,
            style=style,
            overrides=overrides,
            require_all_values=stamping.require_all_values,
        )

    if mode in ("pdf", "both"):
        from app.services.pdf_stamping import run_pdf_stamping_for_job

        summary["pdf"] = run_pdf_stamping_for_job(
            job_id=job_id,
            input_pdf=input_pdf,
            output_dir=output_dir,
            provider=provider,
            model=model,
            values=stamping.values,
            style=style,
            overrides=overrides,
            require_all_values=stamping.require_all_values,
        )

    LOG.info(
        "dev_auto_stamp done job_id=%s fixture=%s applied=%d images_run=%s pdf_run=%s",
        job_id,
        fixture_path.name,
        len(filtered),
        (summary["images"] or {}).get("stamp_run_id") if isinstance(summary["images"], dict) else None,
        (summary["pdf"] or {}).get("stamp_run_id") if isinstance(summary["pdf"], dict) else None,
    )
    return summary
