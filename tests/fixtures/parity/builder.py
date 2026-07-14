"""Builders for E2 stamping-parity golden fixtures (PRD E2, Gate G2).

Each "form" is a minimal post-E1 job directory (job.json, page manifest, source PNG,
grounded fields, stamping.json) with a mix of ``text``, ``multiline_text``, and
``checkbox`` fields. Forms are materialized at test time (rather than committed as
static binaries) so the parity suite stays deterministic across environments while
still exercising two distinct page layouts end-to-end through both stampers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image

from app.services.jobs import new_job_manifest, set_job_grounding, write_job_manifest

PDF_WIDTH_PT = 612.0
PDF_HEIGHT_PT = 792.0
DPI = 200.0
SCALE = 72.0 / DPI  # PDF points per pixel, matches the conversion pipeline (72/200 = 0.36)
IMAGE_WIDTH_PX = round(PDF_WIDTH_PT / SCALE)
IMAGE_HEIGHT_PX = round(PDF_HEIGHT_PT / SCALE)

GROUNDING_PROVIDER = "openai"
GROUNDING_MODEL = "gpt-parity-test"

_FORM_SPECS: dict[str, dict[str, dict[str, object]]] = {
    "form_a": {
        "text": {"bbox": {"x": 200, "y": 300, "w": 500, "h": 40}, "value": "Jane Q. Public"},
        "multiline": {
            "bbox": {"x": 200, "y": 400, "w": 400, "h": 120},
            "value": (
                "This is a long multiline answer that must wrap across several lines "
                "inside the field box and may even truncate if it overflows the height."
            ),
        },
        "checkbox": {"bbox": {"x": 200, "y": 560, "w": 30, "h": 30}, "value": "true"},
    },
    "form_b": {
        "text": {"bbox": {"x": 150, "y": 250, "w": 300, "h": 32}, "value": "42 Example Ave, Springfield"},
        "multiline": {
            "bbox": {"x": 150, "y": 320, "w": 260, "h": 90},
            "value": "Short wrap test with a couple of words that should still break onto two lines.",
        },
        "checkbox": {"bbox": {"x": 150, "y": 450, "w": 24, "h": 24}, "value": "yes"},
    },
}


@dataclass(frozen=True)
class ParityForm:
    job_id: str
    root: Path
    output_dir: Path
    input_pdf: Path
    field_ids: dict[str, str]


def build_parity_job(tmp_path: Path, *, variant: str, job_id: str, font_size_pt: float = 11.0) -> ParityForm:
    """Materialize a minimal post-E1 job directory for one parity form *variant*."""
    spec = _FORM_SPECS[variant]
    root = tmp_path / job_id
    output_dir = root / "output"
    pages_dir = output_dir / "converted_images" / "pages"
    fg_dir = output_dir / "field_grounding"
    pages_dir.mkdir(parents=True)
    fg_dir.mkdir(parents=True)

    input_pdf = root / "input.pdf"
    doc = fitz.open()
    try:
        doc.new_page(width=PDF_WIDTH_PT, height=PDF_HEIGHT_PT)
        doc.save(input_pdf)
    finally:
        doc.close()

    Image.new("RGB", (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX), "white").save(
        output_dir / "converted_images" / "page_0001.png", format="PNG"
    )

    page_manifest = {
        "manifest_version": "1.0",
        "page_index": 0,
        "pdf": {"width_pt": PDF_WIDTH_PT, "height_pt": PDF_HEIGHT_PT, "origin": "bottom-left"},
        "image": {
            "path": "converted_images/page_0001.png",
            "format": "png",
            "width_px": IMAGE_WIDTH_PX,
            "height_px": IMAGE_HEIGHT_PX,
            "origin": "top-left",
        },
        "mapping": {"image_to_pdf_scale_x": SCALE, "image_to_pdf_scale_y": SCALE},
    }
    (pages_dir / "page_0001.json").write_text(json.dumps(page_manifest, indent=2) + "\n", encoding="utf-8")

    field_ids = {
        "text": f"{variant}_text",
        "multiline": f"{variant}_multiline",
        "checkbox": f"{variant}_checkbox",
    }
    fields = [
        {
            "field_id": field_ids["text"],
            "type": "text",
            "bbox": spec["text"]["bbox"],
            "confidence": 0.9,
            "label": "Name",
            "evidence": {"line_ids": []},
            "grounding_type": "text",
            "grounding_source": None,
            "qa_status": None,
            "reviewed": False,
            "font_size_pt": None,
        },
        {
            "field_id": field_ids["multiline"],
            "type": "multiline_text",
            "bbox": spec["multiline"]["bbox"],
            "confidence": 0.9,
            "label": "Notes",
            "evidence": {"line_ids": []},
            "grounding_type": "multiline_text",
            "grounding_source": None,
            "qa_status": None,
            "reviewed": False,
            "font_size_pt": None,
        },
        {
            "field_id": field_ids["checkbox"],
            "type": "checkbox",
            "bbox": spec["checkbox"]["bbox"],
            "confidence": 0.9,
            "label": "Agree",
            "evidence": {"line_ids": []},
            "grounding_type": "checkbox",
            "grounding_source": None,
            "qa_status": None,
            "reviewed": False,
            "font_size_pt": None,
        },
    ]
    grounding_payload = {
        "page_index": 0,
        "width_px": IMAGE_WIDTH_PX,
        "height_px": IMAGE_HEIGHT_PX,
        "unit": "px",
        "origin": "top-left",
        "fields": fields,
    }
    (fg_dir / "page_0001.fields.json").write_text(json.dumps(grounding_payload, indent=2) + "\n", encoding="utf-8")

    stamping_payload = {
        "values": {
            field_ids["text"]: spec["text"]["value"],
            field_ids["multiline"]: spec["multiline"]["value"],
            field_ids["checkbox"]: spec["checkbox"]["value"],
        },
        "require_all_values": False,
        "style": {"font_size_pt": font_size_pt, "text_color": "#111111"},
        "overrides": {field_ids["text"]: {"font_size_pt": font_size_pt + 3}},
    }
    (fg_dir / "stamping.json").write_text(json.dumps(stamping_payload, indent=2) + "\n", encoding="utf-8")

    manifest = new_job_manifest(job_id=job_id, source_filename=f"{variant}.pdf", dpi=int(DPI))
    manifest["page_count"] = 1
    write_job_manifest(root, manifest)
    set_job_grounding(root, provider=GROUNDING_PROVIDER, model=GROUNDING_MODEL)

    return ParityForm(job_id=job_id, root=root, output_dir=output_dir, input_pdf=input_pdf, field_ids=field_ids)
