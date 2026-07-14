"""Shared stamping engine for preview (PNG) and export (PDF) rendering (PRD E2).

Both ``image_stamping.py`` and ``pdf_stamping.py`` import from this module for:

- bbox / page-input validation and page geometry (px<->pt scale)
- grounding-run discovery and assertion
- one font (shipped TTF) and shared text-fitting/wrapping metrics in PDF points,
  so both render paths make identical sizing/wrapping/truncation decisions
- the stamp-run loop and run-manifest scaffold

Each stamper keeps only its renderer (Pillow drawing vs PyMuPDF page ops).
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import fitz
from PIL import ImageFont

from app.services.jobs import grounding_page_px, page_manifest_image_px

_GROUNDING_PAGE_RE = re.compile(r"^page_(\d{4})\.fields\.json$")
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

DEFAULT_LINE_HEIGHT_FACTOR = 1.15

# One shipped TTF used by both render paths (PRD E2.3 font parity).
_ASSET_FONT_PATH = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "DejaVuSans.ttf"
FONT_NAME = "FormiqoSans"


def _resolve_font_path() -> str | None:
    candidates = [
        _ASSET_FONT_PATH,
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


FONT_PATH: str | None = _resolve_font_path()


@functools.lru_cache(maxsize=1)
def _metrics_font() -> fitz.Font:
    """Shared font metrics source for sizing/wrapping decisions on both paths."""
    if FONT_PATH:
        return fitz.Font(fontfile=FONT_PATH)
    return fitz.Font("helv")


@functools.lru_cache(maxsize=256)
def load_pil_font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load the shared TTF at *size_px* for Pillow previews (cached; PRD E2.8)."""
    if FONT_PATH:
        try:
            return ImageFont.truetype(FONT_PATH, size_px)
        except OSError:
            pass
    return ImageFont.load_default()


def text_width_pt(text: str, size_pt: float) -> float:
    """Measure *text* width in PDF points using the shared font metrics (same on both paths)."""
    if not text:
        return 0.0
    return _metrics_font().text_length(text, fontsize=size_pt)


def font_ascent_fraction() -> float:
    """Fraction of font size above the baseline (glyph ascent)."""
    return _metrics_font().ascender


def font_descent_fraction() -> float:
    """Fraction of font size below the baseline (glyph descent), as a positive number."""
    return abs(_metrics_font().descender)


def bottom_padding_pt(style: "StampStyle", font_size_pt: float) -> float:
    """Gap between the text baseline and the bbox bottom edge, sized to fit descenders.

    Replaces the old PDF-path approximation (``rect.y0 + font_size``) with real font
    metrics so descenders (g, y, p, ...) don't clip against the field's bottom edge
    (PRD E2.4).
    """
    return max(style.padding_pt, font_size_pt * font_descent_fraction())


# --------------------------------------------------------------------------- #
# Style model (PDF points; PRD E2.2)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StampStyle:
    """Unified stamping style shared by both render paths, defined in PDF points."""

    font_size_pt: float = 11.0
    text_color: str = "#111111"
    padding_pt: float = 2.0
    draw_debug_boxes: bool = False
    debug_box_color: str = "#ff0000"


def resolve_field_font_size_pt(
    field: dict[str, Any],
    *,
    field_id: str,
    overrides: dict[str, Any],
    default_pt: float,
) -> float:
    """Effective per-field font size: field-level override wins, then ``stamping.json``
    overrides, then the run's global style default.

    The field-level ``font_size_pt`` is what ``PATCH /jobs/{id}/fields`` writes (editor
    font-size stepper); the ``stamping.json`` ``overrides`` map is the E2 mechanism the
    PRD acceptance criteria round-trips directly on disk.
    """
    from_field = field.get("font_size_pt")
    if isinstance(from_field, (int, float)) and from_field > 0:
        return float(from_field)

    entry = overrides.get(field_id)
    if isinstance(entry, dict):
        from_override = entry.get("font_size_pt")
        if isinstance(from_override, (int, float)) and from_override > 0:
            return float(from_override)

    return default_pt


# --------------------------------------------------------------------------- #
# Text fitting / wrapping (PDF points; identical on both paths)
# --------------------------------------------------------------------------- #


def fit_text_to_width_pt(
    text: str,
    *,
    max_width_pt: float,
    preferred_size_pt: float,
    min_size_pt: float = 5.0,
) -> tuple[str, float]:
    """Shrink font, then truncate with an ellipsis if it still does not fit (single line)."""
    size = preferred_size_pt
    while size >= min_size_pt:
        if text_width_pt(text, size) <= max_width_pt:
            return text, size
        size -= 0.5
    size = min_size_pt

    ellipsis = "..."
    if text_width_pt(ellipsis, size) > max_width_pt:
        return "", size

    clipped = text
    while clipped:
        candidate = f"{clipped}{ellipsis}"
        if text_width_pt(candidate, size) <= max_width_pt:
            return candidate, size
        clipped = clipped[:-1]
    return ellipsis, size


def _hard_break_word(word: str, *, max_width_pt: float, size_pt: float) -> list[str]:
    """Break a single word that alone exceeds *max_width_pt* into fitting chunks."""
    pieces: list[str] = []
    chunk = ""
    for ch in word:
        candidate = chunk + ch
        if text_width_pt(candidate, size_pt) <= max_width_pt or not chunk:
            chunk = candidate
        else:
            pieces.append(chunk)
            chunk = ch
    if chunk:
        pieces.append(chunk)
    return pieces


def wrap_text_to_width_pt(text: str, *, max_width_pt: float, size_pt: float) -> list[str]:
    """Greedy word-wrap at *size_pt*; hard-breaks any single word wider than the box."""
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word])
        if text_width_pt(trial, size_pt) <= max_width_pt:
            current.append(word)
            continue
        if current:
            lines.append(" ".join(current))
            current = []
        if text_width_pt(word, size_pt) <= max_width_pt:
            current = [word]
        else:
            lines.extend(_hard_break_word(word, max_width_pt=max_width_pt, size_pt=size_pt))
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def fit_multiline_to_box_pt(
    text: str,
    *,
    max_width_pt: float,
    max_height_pt: float,
    preferred_size_pt: float,
    min_size_pt: float = 5.0,
    line_height_factor: float = DEFAULT_LINE_HEIGHT_FACTOR,
) -> tuple[list[str], float]:
    """Word-wrap *text* to fit width and height, shrinking font until it fits.

    When even ``min_size_pt`` overflows the box, trailing lines are dropped and the
    last retained line is ellipsis-truncated (parity with the single-line overflow
    behavior; PRD E2.5/E2.6).
    """
    size = preferred_size_pt
    lines = wrap_text_to_width_pt(text, max_width_pt=max_width_pt, size_pt=size)
    while size > min_size_pt:
        total_height = len(lines) * size * line_height_factor
        if total_height <= max_height_pt:
            break
        size = max(min_size_pt, size - 0.5)
        lines = wrap_text_to_width_pt(text, max_width_pt=max_width_pt, size_pt=size)

    max_lines = max(1, int(max_height_pt / (size * line_height_factor)))
    if len(lines) > max_lines:
        visible = lines[:max_lines]
        remainder = " ".join(lines[max_lines:])
        candidate_last = f"{visible[-1]} {remainder}".strip()
        truncated_last, _ = fit_text_to_width_pt(
            candidate_last, max_width_pt=max_width_pt, preferred_size_pt=size, min_size_pt=size
        )
        visible[-1] = truncated_last
        lines = visible
    return lines, size


# --------------------------------------------------------------------------- #
# Validation, JSON loading, page discovery (PRD E2.1)
# --------------------------------------------------------------------------- #


def validate_hex_color(value: str, *, field_name: str) -> str:
    if not _HEX_COLOR_RE.match(value):
        raise ValueError(f"{field_name} must be a hex color like #111111.")
    return value


def hex_to_rgb_tuple(value: str) -> tuple[float, float, float]:
    value_norm = validate_hex_color(value, field_name="color")
    return (
        int(value_norm[1:3], 16) / 255.0,
        int(value_norm[3:5], 16) / 255.0,
        int(value_norm[5:7], 16) / 255.0,
    )


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def rel_to_output(output_dir: Path, path: Path) -> str:
    return str(path.relative_to(output_dir)).replace("\\", "/")


def discover_grounding_pages(grounding_dir: Path) -> list[tuple[int, Path]]:
    pages: list[tuple[int, Path]] = []
    for path in sorted(grounding_dir.glob("page_*.fields.json")):
        match = _GROUNDING_PAGE_RE.match(path.name)
        if not match:
            continue
        pages.append((int(match.group(1)) - 1, path))
    if not pages:
        raise FileNotFoundError(f"No grounded page files found under: {grounding_dir}")
    return pages


def assert_grounding_run_matches(job_root_dir: Path, *, provider: str, model: str) -> None:
    from app.services.jobs import job_grounding_provider_model

    try:
        found_prov, found_model = job_grounding_provider_model(job_root_dir)
    except (FileNotFoundError, ValueError):
        return
    if found_prov != provider or found_model != model:
        raise FileNotFoundError(
            f"Field grounding run not found for provider={provider}, model={model} "
            f"(found provider={found_prov}, model={found_model})."
        )


def read_page_manifest(output_dir: Path, page_index: int) -> tuple[Path, dict[str, Any]]:
    rel = Path("converted_images") / "pages" / f"page_{page_index + 1:04d}.json"
    path = output_dir / rel
    if not path.is_file():
        raise FileNotFoundError(f"Conversion page manifest not found: {rel}")
    return rel, load_json(path)


def image_path_from_manifest(output_dir: Path, page_manifest: dict[str, Any]) -> Path:
    image_node = page_manifest.get("image")
    if not isinstance(image_node, dict):
        raise ValueError("Conversion page manifest missing image object.")
    image_rel = image_node.get("path")
    if not isinstance(image_rel, str) or not image_rel:
        raise ValueError("Conversion page manifest image.path must be a non-empty string.")
    output_root = output_dir.resolve()
    image_path = (output_root / image_rel).resolve()
    try:
        image_path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"Source image path escapes job output directory: {image_rel}") from exc
    if not image_path.is_file():
        raise FileNotFoundError(f"Source image not found: {image_rel}")
    return image_path


@dataclass(frozen=True)
class PageGeometry:
    """Validated per-page geometry shared by both stampers."""

    page_index: int
    width_px: int
    height_px: int
    pdf_width_pt: float
    pdf_height_pt: float
    scale_x: float  # PDF points per pixel
    scale_y: float
    image_rel: str


def build_page_geometry(
    *,
    page_manifest: dict[str, Any],
    grounding: dict[str, Any],
    page_index: int,
) -> PageGeometry:
    """Validate grounding dims against the page manifest and compute the px<->pt scale."""
    pdf_node = page_manifest.get("pdf")
    image_node = page_manifest.get("image")
    if not isinstance(pdf_node, dict):
        raise ValueError("Conversion page manifest missing pdf object.")
    if not isinstance(image_node, dict):
        raise ValueError("Conversion page manifest missing image object.")

    try:
        pdf_w_pt = float(pdf_node["width_pt"])
        pdf_h_pt = float(pdf_node["height_pt"])
        img_w, img_h = page_manifest_image_px(page_manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Conversion page manifest is missing required page dimensions.") from exc

    g_page_index = int(grounding.get("page_index", -1))
    try:
        g_width, g_height = grounding_page_px(grounding)
    except ValueError as exc:
        raise ValueError("Grounding page missing dimensions.") from exc
    g_unit = grounding.get("unit")
    g_origin = grounding.get("origin")

    if g_page_index != page_index:
        raise ValueError(f"Grounding page_index ({g_page_index}) does not match expected {page_index}.")
    if g_width != img_w or g_height != img_h:
        raise ValueError(
            f"Grounding dimensions ({g_width}x{g_height}) do not match converted image ({img_w}x{img_h})."
        )
    if g_unit != "px":
        raise ValueError('Grounding unit must be "px".')
    if g_origin != "top-left":
        raise ValueError('Grounding origin must be "top-left".')
    if pdf_w_pt <= 0 or pdf_h_pt <= 0:
        raise ValueError("PDF dimensions must be positive.")
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Image dimensions must be positive.")

    return PageGeometry(
        page_index=page_index,
        width_px=img_w,
        height_px=img_h,
        pdf_width_pt=pdf_w_pt,
        pdf_height_pt=pdf_h_pt,
        scale_x=pdf_w_pt / float(img_w),
        scale_y=pdf_h_pt / float(img_h),
        image_rel=str(image_node.get("path", "")),
    )


def bbox_from_field(field: dict[str, Any], *, width_px: int, height_px: int, field_index: int) -> dict[str, float]:
    bbox = field.get("bbox")
    if not isinstance(bbox, dict):
        raise ValueError(f"fields[{field_index}].bbox must be an object.")
    expected = {"x", "y", "w", "h"}
    if set(bbox.keys()) != expected:
        raise ValueError(f"fields[{field_index}].bbox keys must be exactly {sorted(expected)}.")
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["w"])
        h = float(bbox["h"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fields[{field_index}].bbox values must be numeric.") from exc

    if w <= 0 or h <= 0:
        raise ValueError(f"fields[{field_index}].bbox requires w > 0 and h > 0.")
    if x < 0 or y < 0:
        raise ValueError(f"fields[{field_index}].bbox requires x >= 0 and y >= 0.")
    if x + w > width_px or y + h > height_px:
        raise ValueError(f"fields[{field_index}].bbox exceeds image bounds ({width_px}x{height_px}).")
    return {"x": x, "y": y, "w": w, "h": h}


# --------------------------------------------------------------------------- #
# Stamp-run loop and run-manifest scaffold (PRD E2.1)
# --------------------------------------------------------------------------- #


def new_stamp_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


@dataclass(frozen=True)
class StampRunContext:
    provider: str
    model: str
    stamp_run_id: str
    run_dir: Path
    run_dir_rel: str
    grounding_pages: list[tuple[int, Path]]


def prepare_stamp_run(
    *,
    output_dir: Path,
    provider: str,
    model: str,
    subdir: Literal["stamped_images", "stamped_pdfs"],
) -> StampRunContext:
    """Validate prerequisites and allocate a new stamp run directory."""
    provider_norm = provider.strip().lower()
    if not provider_norm:
        raise ValueError("provider must be a non-empty string.")
    if not model.strip():
        raise ValueError("model must be a non-empty string.")

    converted_pages_dir = output_dir / "converted_images" / "pages"
    if not converted_pages_dir.is_dir():
        raise FileNotFoundError("Conversion page manifests not found: converted_images/pages")

    grounding_dir = output_dir / "field_grounding"
    if not grounding_dir.is_dir():
        raise FileNotFoundError(
            f"Field grounding run not found for provider={provider_norm}, model={model} "
            f"(expected directory: field_grounding)."
        )
    assert_grounding_run_matches(output_dir.parent, provider=provider_norm, model=model)

    stamp_run_id = new_stamp_run_id()
    run_dir_rel = f"{subdir}/{stamp_run_id}"
    run_dir = output_dir / run_dir_rel
    grounding_pages = discover_grounding_pages(grounding_dir)

    return StampRunContext(
        provider=provider_norm,
        model=model,
        stamp_run_id=stamp_run_id,
        run_dir=run_dir,
        run_dir_rel=run_dir_rel,
        grounding_pages=grounding_pages,
    )


def run_stamp_pages(
    *,
    output_dir: Path,
    grounding_pages: list[tuple[int, Path]],
    render_page: Callable[[int, Path], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Iterate grounding pages, isolating per-page failures into ``status: failed`` entries."""
    results: list[dict[str, Any]] = []
    for page_index, grounding_path in grounding_pages:
        try:
            results.append(render_page(page_index, grounding_path))
        except Exception as exc:  # noqa: BLE001 - preserve per-page errors for API responses
            results.append(
                {
                    "page_index": page_index,
                    "status": "failed",
                    "grounding_file": rel_to_output(output_dir, grounding_path),
                    "error": str(exc),
                }
            )
    return results


def summarize_page_results(page_results: list[dict[str, Any]]) -> tuple[int, int]:
    succeeded = sum(1 for p in page_results if p.get("status") == "succeeded")
    return succeeded, len(page_results) - succeeded


def write_run_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def finalize_job_after_stamp(
    *,
    output_dir: Path,
    subdir: Literal["stamped_images", "stamped_pdfs"],
    update_job: Callable[[Path], None],
) -> None:
    """Prune old runs and update job.json; tolerate a missing job.json (ad-hoc/legacy runs)."""
    from app.services.jobs import prune_stamp_runs, read_job_manifest

    job_root_dir = output_dir.parent
    try:
        job_manifest = read_job_manifest(job_root_dir)
        max_runs = int(job_manifest.get("retention", {}).get("max_stamp_runs", 3))
        prune_stamp_runs(output_dir, subdir, max_runs)
        update_job(job_root_dir)
    except FileNotFoundError:
        pass
