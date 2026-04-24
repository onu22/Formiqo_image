"""Stamp a PDF using a stamp plan and a {field_id: value} payload.

Reads stamp plans from ``output/stamp_plans/{provider}_{model}/`` (produced by
:func:`app.services.stamping.run_prepare_stamp_for_job`) and writes a stamped
copy to ``output/stamped_pdfs/{provider}_{model}/stamped.pdf``.

PyMuPDF is used directly because the plan's ``pdf_rect_tl`` rectangles are
already in PyMuPDF's top-left PDF coordinate space.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz

STAMP_RESULT_SCHEMA_VERSION = "1.0"

_STAMP_PAGE_RE = re.compile(r"^page_(\d{4})\.stamp\.json$")
_MODEL_DIR_SAFE_RE = re.compile(r"[^a-z0-9._-]+")

_ALIGN_MAP = {
    "left": 0,
    "center": 1,
    "right": 2,
}

DEFAULT_FONTSIZE = 10.0
DEFAULT_FONTNAME = "helv"
DEFAULT_COLOR_RGB: tuple[float, float, float] = (0.0, 0.0, 0.0)
DEFAULT_ALIGN = "left"
DEFAULT_MIN_FONTSIZE = 6.0


@dataclass(frozen=True)
class StampOptions:
    fontsize: float = DEFAULT_FONTSIZE
    fontname: str = DEFAULT_FONTNAME
    color_rgb: tuple[float, float, float] = DEFAULT_COLOR_RGB
    align: str = DEFAULT_ALIGN
    autoshrink: bool = True
    min_fontsize: float = DEFAULT_MIN_FONTSIZE

    def __post_init__(self) -> None:
        if self.fontsize <= 0:
            raise ValueError("fontsize must be > 0")
        if self.min_fontsize <= 0 or self.min_fontsize > self.fontsize:
            raise ValueError("min_fontsize must be > 0 and <= fontsize")
        if self.align not in _ALIGN_MAP:
            raise ValueError(f"align must be one of {sorted(_ALIGN_MAP.keys())}")
        if (
            not isinstance(self.color_rgb, tuple)
            or len(self.color_rgb) != 3
            or any(c < 0 or c > 1 for c in self.color_rgb)
        ):
            raise ValueError("color_rgb must be a 3-tuple of floats in [0, 1]")
        if not self.fontname.strip():
            raise ValueError("fontname must be a non-empty string")


@dataclass
class FieldStampResult:
    field_id: str
    status: str
    final_fontsize: float | None = None
    error: str | None = None


@dataclass
class PageStampResult:
    page_index: int
    fields: list[FieldStampResult] = field(default_factory=list)

    @property
    def stamped_count(self) -> int:
        return sum(1 for f in self.fields if f.status == "stamped")


def _provider_model_dir_name(provider: str, model: str) -> str:
    safe_provider = _MODEL_DIR_SAFE_RE.sub("-", provider.lower()).strip("-")
    safe_model = _MODEL_DIR_SAFE_RE.sub("-", model.lower()).strip("-")
    return f"{safe_provider}_{safe_model}"


def _discover_stamp_plan_pages(plan_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for path in sorted(plan_dir.glob("page_*.stamp.json")):
        m = _STAMP_PAGE_RE.match(path.name)
        if not m:
            continue
        page_index = int(m.group(1)) - 1
        out.append((page_index, path))
    return out


def stamp_field_on_page(
    page: "fitz.Page",
    rect: "fitz.Rect",
    text: str,
    options: StampOptions,
) -> tuple[str, float]:
    """Insert ``text`` into ``rect`` on ``page``.

    Returns ``(status, final_fontsize)`` where ``status`` is one of
    ``"stamped"`` or ``"overflow"``. Uses PyMuPDF ``insert_textbox`` which
    returns a negative value when the text does not fit in the rect; if
    ``autoshrink`` is enabled we retry down to ``min_fontsize``.
    """
    align = _ALIGN_MAP[options.align]
    fontsize = float(options.fontsize)
    min_fs = float(options.min_fontsize)
    last_remaining = 0.0

    while True:
        remaining = page.insert_textbox(
            rect,
            text,
            fontname=options.fontname,
            fontsize=fontsize,
            color=options.color_rgb,
            align=align,
        )
        last_remaining = remaining
        if remaining >= 0:
            return "stamped", fontsize
        if not options.autoshrink or fontsize - 1.0 < min_fs:
            return "overflow", fontsize
        fontsize -= 1.0


def _resolve_options(options: StampOptions | None) -> StampOptions:
    return options if options is not None else StampOptions()


def _serialize_options(options: StampOptions) -> dict[str, Any]:
    data = asdict(options)
    data["color_rgb"] = list(options.color_rgb)
    return data


def _collect_known_field_ids(pages_plans: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for plan in pages_plans:
        for f in plan.get("fields", []):
            fid = f.get("field_id")
            if isinstance(fid, str):
                ids.add(fid)
    return ids


def run_stamp_pdf_for_job(
    *,
    job_id: str,
    input_pdf: Path,
    output_dir: Path,
    provider: str,
    model: str,
    values: dict[str, str] | None,
    strict: bool = False,
    options: StampOptions | None = None,
) -> dict[str, Any]:
    """Stamp ``input_pdf`` using the stamp plan for ``{provider}_{model}``.

    Writes ``output/stamped_pdfs/{provider}_{model}/stamped.pdf`` and
    ``stamp_result.json``. Returns a summary dict that matches the response
    schema.
    """
    provider_norm = provider.strip().lower()
    if not provider_norm:
        raise ValueError("provider must be a non-empty string.")
    if not model.strip():
        raise ValueError("model must be a non-empty string.")

    opts = _resolve_options(options)
    values_map: dict[str, str] = dict(values or {})

    run_dir_name = _provider_model_dir_name(provider_norm, model)
    plan_dir = output_dir / "stamp_plans" / run_dir_name
    if not plan_dir.is_dir():
        raise FileNotFoundError(
            f"Stamp plan run not found for provider={provider_norm}, model={model} "
            f"(expected directory: stamp_plans/{run_dir_name}). "
            f"Run POST /prepare-stamp first."
        )

    if not input_pdf.is_file():
        raise FileNotFoundError(f"Source PDF not found: {input_pdf}")

    plan_pages_paths = _discover_stamp_plan_pages(plan_dir)
    if not plan_pages_paths:
        raise FileNotFoundError(
            f"No stamp plan page files found under stamp_plans/{run_dir_name}."
        )

    pages_plans: list[dict[str, Any]] = []
    for page_index, path in plan_pages_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("page_index", -1)) != page_index:
            raise ValueError(
                f"Stamp plan file {path.name} reports page_index={data.get('page_index')} "
                f"but filename indicates {page_index}."
            )
        pages_plans.append(data)

    known_ids = _collect_known_field_ids(pages_plans)
    requested_ids = set(values_map.keys())
    unknown_ids = sorted(requested_ids - known_ids)
    missing_for_known = sorted(known_ids - requested_ids)

    if strict and (unknown_ids or missing_for_known):
        raise ValueError(
            "strict=true: "
            + (f"unknown field_ids={unknown_ids}; " if unknown_ids else "")
            + (f"missing values for field_ids={missing_for_known}" if missing_for_known else "")
        )

    stamped_out_dir = output_dir / "stamped_pdfs" / run_dir_name
    stamped_out_dir.mkdir(parents=True, exist_ok=True)
    stamped_pdf_path = stamped_out_dir / "stamped.pdf"
    result_path = stamped_out_dir / "stamp_result.json"

    page_results: list[PageStampResult] = []

    doc = fitz.open(str(input_pdf))
    try:
        for plan in pages_plans:
            page_index = int(plan["page_index"])
            if page_index < 0 or page_index >= doc.page_count:
                page_res = PageStampResult(page_index=page_index)
                for f in plan.get("fields", []):
                    page_res.fields.append(
                        FieldStampResult(
                            field_id=str(f.get("field_id", "")),
                            status="error",
                            error=f"page_index {page_index} out of range for source PDF",
                        )
                    )
                page_results.append(page_res)
                continue

            page = doc[page_index]
            page_res = PageStampResult(page_index=page_index)

            for f in plan.get("fields", []):
                field_id = str(f.get("field_id", ""))
                rect_tl = f.get("pdf_rect_tl") or {}
                try:
                    rect = fitz.Rect(
                        float(rect_tl["x0"]),
                        float(rect_tl["y0"]),
                        float(rect_tl["x1"]),
                        float(rect_tl["y1"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    page_res.fields.append(
                        FieldStampResult(field_id=field_id, status="error", error=f"invalid pdf_rect_tl: {exc}")
                    )
                    continue

                if field_id not in values_map:
                    page_res.fields.append(
                        FieldStampResult(field_id=field_id, status="no_value")
                    )
                    continue

                text = str(values_map[field_id])
                if text == "":
                    page_res.fields.append(
                        FieldStampResult(field_id=field_id, status="no_value")
                    )
                    continue

                try:
                    status, final_fs = stamp_field_on_page(page, rect, text, opts)
                    page_res.fields.append(
                        FieldStampResult(field_id=field_id, status=status, final_fontsize=final_fs)
                    )
                except Exception as exc:  # noqa: BLE001 - surface as field error
                    page_res.fields.append(
                        FieldStampResult(field_id=field_id, status="error", error=str(exc))
                    )

            page_results.append(page_res)

        try:
            doc.save(str(stamped_pdf_path), garbage=3, deflate=True)
        except Exception as exc:  # noqa: BLE001 - surface via RuntimeError to map 422
            raise RuntimeError(f"Failed to save stamped PDF: {exc}") from exc
    finally:
        doc.close()

    stamped_field_count = sum(p.stamped_count for p in page_results)
    overflow_count = sum(1 for p in page_results for f in p.fields if f.status == "overflow")
    no_value_ids = sorted(
        {f.field_id for p in page_results for f in p.fields if f.status == "no_value"}
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    result_doc = {
        "schema_version": STAMP_RESULT_SCHEMA_VERSION,
        "job_id": job_id,
        "provider": provider_norm,
        "model": model,
        "source_pdf": str(input_pdf),
        "stamp_plan_run_dir": f"stamp_plans/{run_dir_name}",
        "stamped_pdf": f"stamped_pdfs/{run_dir_name}/stamped.pdf",
        "created_at": now_iso,
        "strict": bool(strict),
        "options": _serialize_options(opts),
        "summary": {
            "pages": len(page_results),
            "stamped_fields": stamped_field_count,
            "skipped_missing_values": len(no_value_ids),
            "unknown_field_ids": len(unknown_ids),
            "overflow_fields": overflow_count,
        },
        "unknown_field_ids": unknown_ids,
        "skipped_missing_values": no_value_ids,
        "pages": [
            {
                "page_index": p.page_index,
                "stamped": p.stamped_count,
                "fields": [
                    {
                        k: v
                        for k, v in {
                            "field_id": f.field_id,
                            "status": f.status,
                            "final_fontsize": f.final_fontsize,
                            "error": f.error,
                        }.items()
                        if v is not None
                    }
                    for f in p.fields
                ],
            }
            for p in page_results
        ],
    }
    result_path.write_text(json.dumps(result_doc, indent=2) + "\n", encoding="utf-8")

    return {
        "job_id": job_id,
        "provider": provider_norm,
        "model": model,
        "run_dir": f"stamped_pdfs/{run_dir_name}",
        "stamped_pdf_path": f"stamped_pdfs/{run_dir_name}/stamped.pdf",
        "result_path": f"stamped_pdfs/{run_dir_name}/stamp_result.json",
        "page_count": len(page_results),
        "stamped_field_count": stamped_field_count,
        "skipped_missing_values": no_value_ids,
        "unknown_field_ids": unknown_ids,
        "pages": [
            {
                "page_index": p.page_index,
                "stamped": p.stamped_count,
                "fields": [
                    {
                        k: v
                        for k, v in {
                            "field_id": f.field_id,
                            "status": f.status,
                            "final_fontsize": f.final_fontsize,
                            "error": f.error,
                        }.items()
                        if v is not None
                    }
                    for f in p.fields
                ],
            }
            for p in page_results
        ],
    }


def resolve_stamped_pdf_path(
    output_dir: Path, provider: str, model: str
) -> Path:
    """Return the on-disk path to the stamped PDF for a given provider/model run."""
    run_dir_name = _provider_model_dir_name(provider.strip().lower(), model)
    return output_dir / "stamped_pdfs" / run_dir_name / "stamped.pdf"
