"""Native AcroForm field extraction and filling (no OCR, no flatten)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import fitz

from app.services.pdf_pipeline.errors import PdfPipelineError

LOG = logging.getLogger(__name__)

_ACROFORM_SUBDIR = "acroform"
_FIELDS_JSON = "form_fields.json"
_VALUES_JSON = "values.json"
_FILLED_PDF = "filled.pdf"


def _field_type_label(ft: int) -> str:
    mapping = {
        fitz.PDF_WIDGET_TYPE_TEXT: "text",
        fitz.PDF_WIDGET_TYPE_CHECKBOX: "checkbox",
        fitz.PDF_WIDGET_TYPE_RADIOBUTTON: "radiobutton",
        fitz.PDF_WIDGET_TYPE_BUTTON: "button",
        fitz.PDF_WIDGET_TYPE_COMBOBOX: "combobox",
        fitz.PDF_WIDGET_TYPE_LISTBOX: "listbox",
        fitz.PDF_WIDGET_TYPE_SIGNATURE: "signature",
        fitz.PDF_WIDGET_TYPE_UNKNOWN: "unknown",
    }
    return mapping.get(ft, "unknown")


def extract_form_fields(input_pdf: Path) -> list[dict[str, Any]]:
    """Serialize widget metadata for each form field."""
    fields: list[dict[str, Any]] = []
    with fitz.open(input_pdf) as doc:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            widgets = page.widgets()
            if widgets is None:
                continue
            for w in widgets:
                name = w.field_name or ""
                ft = int(w.field_type)
                rect = [float(w.rect.x0), float(w.rect.y0), float(w.rect.x1), float(w.rect.y1)]
                entry: dict[str, Any] = {
                    "field_name": name,
                    "field_type": _field_type_label(ft),
                    "field_type_code": ft,
                    "page_index": page_index,
                    "rect": rect,
                    "field_flags": int(w.field_flags),
                    "current_value": w.field_value,
                }
                choices = getattr(w, "choice_values", None)
                if choices:
                    entry["choice_values"] = list(choices)
                states = None
                try:
                    states = w.button_states()
                except (AttributeError, RuntimeError):
                    pass
                if states:
                    entry["button_states"] = states
                fields.append(entry)
    return fields


def _sample_text_placeholder(field_name: str) -> str:
    """Same spirit as vision ``stamping.json`` samples: short deterministic placeholder."""
    n = (field_name or "").strip() or "field"
    return n[:10]


def _pick_non_off_export(meta: dict[str, Any]) -> str | None:
    """First appearance-on export from widget ``button_states`` (skip Off)."""
    bs = meta.get("button_states")
    if not isinstance(bs, dict):
        return None
    seen: list[str] = []
    for key in ("normal", "down"):
        lst = bs.get(key)
        if isinstance(lst, list):
            seen.extend(str(x) for x in lst if x is not None)
    for raw in seen:
        token = raw.strip().lstrip("/").strip().lower()
        if token and token != "off":
            return raw.strip().lstrip("/")
    return None


def _default_value_for_field(meta: dict[str, Any]) -> str | bool:
    """Populate visible defaults so ``filled.pdf`` is not blank on first run."""
    ft = meta.get("field_type", "")
    name = meta.get("field_name") or ""

    if ft == "checkbox":
        return True
    if ft == "radiobutton":
        export = _pick_non_off_export(meta)
        if export is not None:
            return export
        choices = meta.get("choice_values") or []
        if choices:
            return str(choices[0]).strip().lstrip("/")
        return ""
    if ft in ("listbox", "combobox"):
        choices = meta.get("choice_values") or []
        if choices:
            return str(choices[0])
        return _sample_text_placeholder(name)
    if ft == "button":
        return ""
    if ft == "signature":
        return ""
    if ft == "text":
        return _sample_text_placeholder(name)
    # unknown + anything else treat like text for sample visibility
    return _sample_text_placeholder(name)


def build_default_values(fields_meta: list[dict[str, Any]]) -> dict[str, Any]:
    """shape: {\"values\": {full_name: ...}} matching stamping-style payloads."""
    values: dict[str, str | bool] = {}
    for meta in fields_meta:
        name = meta.get("field_name") or ""
        if not name:
            continue
        if meta.get("field_type") == "button":
            continue
        dv = _default_value_for_field(meta)
        if isinstance(dv, bool):
            values[name] = dv
        else:
            values[name] = str(dv)
    return {"values": values}


def load_or_build_values(acro_dir: Path, fields_meta: list[dict[str, Any]]) -> dict[str, Any]:
    values_path = acro_dir / _VALUES_JSON
    if values_path.is_file():
        raw = json.loads(values_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "values" not in raw:
            raise PdfPipelineError(f"{values_path} must be a JSON object with a 'values' key.")
        if not isinstance(raw["values"], dict):
            raise PdfPipelineError(f"{values_path}: 'values' must be an object.")
        return raw
    payload = build_default_values(fields_meta)
    values_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOG.info("acroform wrote default %s", values_path)
    return payload


def fill_acroform_pdf(*, input_pdf: Path, output_pdf: Path, values_map: dict[str, Any]) -> int:
    """Apply values to widgets; returns count of widgets updated."""
    updated = 0
    doc = fitz.open(input_pdf)
    try:
        for page in doc:
            widgets = page.widgets()
            if widgets is None:
                continue
            for w in widgets:
                name = w.field_name or ""
                if not name or name not in values_map:
                    continue
                raw_val = values_map[name]
                ft = int(w.field_type)

                if ft == fitz.PDF_WIDGET_TYPE_SIGNATURE:
                    LOG.debug("acroform skip signature field %s", name)
                    continue
                flags = int(getattr(w, "field_flags", 0))
                if ft == fitz.PDF_WIDGET_TYPE_BUTTON and flags & fitz.PDF_BTN_FIELD_IS_PUSHBUTTON:
                    continue

                try:
                    if ft == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                        if isinstance(raw_val, bool):
                            w.field_value = raw_val
                        else:
                            s = str(raw_val).strip().lower()
                            w.field_value = s in ("true", "yes", "1", "on")
                    elif ft == fitz.PDF_WIDGET_TYPE_RADIOBUTTON:
                        w.field_value = str(raw_val)
                    elif ft in (fitz.PDF_WIDGET_TYPE_COMBOBOX, fitz.PDF_WIDGET_TYPE_LISTBOX):
                        w.field_value = str(raw_val)
                    elif ft == fitz.PDF_WIDGET_TYPE_TEXT:
                        w.field_value = str(raw_val)
                    elif ft == fitz.PDF_WIDGET_TYPE_BUTTON:
                        continue
                    else:
                        w.field_value = str(raw_val)
                    w.update()
                    updated += 1
                except Exception as exc:
                    LOG.warning("acroform could not set field %s: %s", name, exc)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        doc.save(output_pdf, incremental=False, garbage=4, deflate=True)
    finally:
        doc.close()
    return updated


class AcroFormPdfPipeline:
    """Extract AcroForm metadata, ensure values JSON, fill PDF natively."""

    def run(self, *, job_id: str, input_pdf: Path, output_dir: Path) -> dict[str, Any]:
        acro_dir = output_dir / _ACROFORM_SUBDIR
        acro_dir.mkdir(parents=True, exist_ok=True)

        fields_meta = extract_form_fields(input_pdf)
        fields_path = acro_dir / _FIELDS_JSON
        fields_path.write_text(json.dumps({"fields": fields_meta}, indent=2) + "\n", encoding="utf-8")
        LOG.info("acroform job=%s extracted %d fields → %s", job_id, len(fields_meta), fields_path)

        values_payload = load_or_build_values(acro_dir, fields_meta)
        values_map = values_payload["values"]
        if not isinstance(values_map, dict):
            raise PdfPipelineError("'values' must be an object.")

        out_pdf = acro_dir / _FILLED_PDF
        filled = fill_acroform_pdf(input_pdf=input_pdf, output_pdf=out_pdf, values_map=values_map)

        root = output_dir.parent
        LOG.info(
            "acroform job=%s pipeline=acroform filled_widgets=%d output=%s",
            job_id,
            filled,
            out_pdf.relative_to(root),
        )

        return {
            "pipeline": "acroform",
            "job_id": job_id,
            "form_fields_path": str(fields_path.relative_to(root)).replace("\\", "/"),
            "values_path": str((acro_dir / _VALUES_JSON).relative_to(root)).replace("\\", "/"),
            "filled_pdf_path": str(out_pdf.relative_to(root)).replace("\\", "/"),
            "widgets_updated": filled,
            "field_count": len(fields_meta),
        }
