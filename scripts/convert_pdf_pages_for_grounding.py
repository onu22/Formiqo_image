#!/usr/bin/env python3
"""
Convert each PDF page to a PNG for form grounding, with JSON manifests that
preserve an exact linear map between image pixels (top-left origin) and PDF
points (bottom-left origin).

Usage:
    python scripts/convert_pdf_pages_for_grounding.py INPUT.pdf --output-dir OUT [--dpi 200] [--overwrite]

See README.md for output layout. This module exposes:
    - convert_pdf_to_images(pdf_path, output_dir, dpi=200, ...)
    - map_image_bbox_to_pdf(bbox, page_manifest)
    - scales_from_dimensions(pdf_w_pt, pdf_h_pt, img_w_px, img_h_px)

Coordinate conventions
----------------------
- Image: origin at the **top-left** of the stored bitmap. ``x`` increases to the
  right, ``y`` increases **downward**. A bounding box ``(x, y, w, h)`` uses that
  same top-left corner ``(x, y)`` with non-negative ``w`` and ``h``.
- PDF: origin at the **bottom-left** of the page (standard PDF user space for
  ``Page.rect`` in PyMuPDF). ``map_image_bbox_to_pdf`` returns the PDF rectangle
  aligned with the formulas in each manifest's ``mapping.formula``: the lower-left
  corner ``(pdf_x, pdf_y)`` in points plus ``pdf_w`` and ``pdf_h``.

Image → PDF (lower-left of the same visual box)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let ``sx = pdf_width_pt / saved_image_width_px`` and
``sy = pdf_height_pt / saved_image_height_px``::

    pdf_x = x * sx
    pdf_y = pdf_height_pt - ((y + h) * sy)
    pdf_w = w * sx
    pdf_h = h * sy

These are the authoritative scales stored as ``image_to_pdf_scale_x`` and
``image_to_pdf_scale_y`` in the manifest. Downstream grounding should treat the
saved PNG width and height as authoritative pixel extents.

Assumptions and fidelity caveats
--------------------------------
- **Page rectangle**: Sizes come from ``Page.rect`` (PyMuPDF’s effective page
  area, typically respecting CropBox / MediaBox per library rules). Unusual
  PDFs with asymmetric boxes still rasterize the full ``rect`` by default.
- **Rotation**: Non-zero ``page.rotation`` changes how the pixmap relates to
  unrotated PDF axes. By default this tool **aborts** so you never get a manifest
  that looks linear but is wrong. Use ``--allow-rotated-pages`` only if you accept
  ``mapping.status == "unsupported_simple_linear"`` and will handle rotation in
  stamping yourself.
- **Uniform scale**: Rendering uses ``fitz.Matrix(zoom, zoom)`` with
  ``zoom = dpi / 72``. For rotation == 0, we verify that ``pdf_w/px_w`` and
  ``pdf_h/px_h`` (points per pixel) agree within tolerance. MuPDF uses **integer**
  pixmap width and height, so fractional ``Page.rect`` sizes plus independent
  rounding can make those ratios differ slightly; the tolerance is set to allow
  that quantization while still rejecting true anisotropic output.
- **Rasterization**: Mapping is linear in continuous coordinates; anti-aliasing
  does not preserve sub-pixel ink boundaries.
- **No post steps**: No crop, pad, deskew, resize, or annotation overlay in the
  default pipeline (``get_pixmap(..., annots=False)``); ``saved_*`` dimensions
  equal ``rendered_*`` unless a future explicit resize path records an additional
  transform (not implemented here).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import fitz

LOG = logging.getLogger(__name__)

MANIFEST_VERSION = "1.0"
# Tolerance for ``pdf_w/img_w`` vs ``pdf_h/img_h`` (rotation == 0 only). Integer
# pixmap dimensions vs float ``page.rect`` can produce ~1e-5 relative drift; see
# module docstring. True anisotropic scaling would differ by orders of magnitude more.
# Some PDFs with fractional page sizes can produce slightly larger axis drift
# after independent integer rounding of rendered width/height.
_UNIFORM_SCALE_RTOL = 3e-4
_UNIFORM_SCALE_ATOL = 1e-12


class PageRecord(TypedDict):
    page_index: int
    image_path: str
    page_manifest_path: str


def scales_from_dimensions(
    pdf_w_pt: float,
    pdf_h_pt: float,
    img_w_px: int,
    img_h_px: int,
) -> tuple[float, float]:
    """
    Return ``(image_to_pdf_scale_x, image_to_pdf_scale_y)`` in points per pixel.

    Raises ``ValueError`` if dimensions are non-positive.
    """
    if pdf_w_pt <= 0 or pdf_h_pt <= 0:
        raise ValueError("PDF width and height in points must be positive.")
    if img_w_px <= 0 or img_h_px <= 0:
        raise ValueError("Image width and height in pixels must be positive.")
    return (pdf_w_pt / img_w_px, pdf_h_pt / img_h_px)


def _assert_uniform_scale(
    pdf_w_pt: float,
    pdf_h_pt: float,
    img_w_px: int,
    img_h_px: int,
) -> None:
    """Reject clearly anisotropic renders; allow small drift from integer pixmap size."""
    sx, sy = scales_from_dimensions(pdf_w_pt, pdf_h_pt, img_w_px, img_h_px)
    if not math.isclose(sx, sy, rel_tol=_UNIFORM_SCALE_RTOL, abs_tol=_UNIFORM_SCALE_ATOL):
        raise RuntimeError(
            "Image-to-PDF scale mismatch after render; refusing to emit manifest. "
            f"pdf=({pdf_w_pt:g}x{pdf_h_pt:g})pt, image=({img_w_px}x{img_h_px})px, "
            f"sx={sx:.12g}, sy={sy:.12g}. "
            "Expected near-equal scales with a diagonal DPI matrix and rotation == 0; "
            "large gaps suggest anisotropic scaling or a renderer issue, not only "
            "integer-rounding noise."
        )


def map_image_bbox_to_pdf(
    bbox: tuple[float, float, float, float],
    page_manifest: dict[str, Any],
) -> dict[str, float]:
    """
    Map a top-left image bounding box ``(x, y, w, h)`` to PDF points.

    Returns keys ``pdf_x``, ``pdf_y``, ``pdf_w``, ``pdf_h`` (lower-left corner
    plus width and height in PDF space, origin bottom-left).
    """
    x, y, w, h = bbox
    pdf_meta = page_manifest["pdf"]
    img_meta = page_manifest["image"]
    mapping = page_manifest["mapping"]

    pdf_w_pt = float(pdf_meta["width_pt"])
    pdf_h_pt = float(pdf_meta["height_pt"])
    saved_w = int(img_meta["saved_image_width_px"])
    saved_h = int(img_meta["saved_image_height_px"])

    sx = float(mapping["image_to_pdf_scale_x"])
    sy = float(mapping["image_to_pdf_scale_y"])

    # Cross-check manifest against stored dimensions (catches hand-edited JSON).
    ex_sx, ex_sy = scales_from_dimensions(pdf_w_pt, pdf_h_pt, saved_w, saved_h)
    if not math.isclose(sx, ex_sx, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"image_to_pdf_scale_x {sx} != expected {ex_sx} from dimensions.")
    if not math.isclose(sy, ex_sy, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"image_to_pdf_scale_y {sy} != expected {ex_sy} from dimensions.")

    pdf_x = x * sx
    pdf_y = pdf_h_pt - ((y + h) * sy)
    pdf_w = w * sx
    pdf_h = h * sy
    return {"pdf_x": pdf_x, "pdf_y": pdf_y, "pdf_w": pdf_w, "pdf_h": pdf_h}


def _pymupdf_version() -> str:
    try:
        return importlib.metadata.version("pymupdf")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _render_page_pixmap(page: fitz.Page, matrix: fitz.Matrix) -> fitz.Pixmap:
    """Rasterize ``page`` without alpha; omit PDF annotations when supported."""
    try:
        return page.get_pixmap(matrix=matrix, alpha=False, annots=False)
    except TypeError:
        # Older PyMuPDF builds may not support the ``annots`` keyword.
        return page.get_pixmap(matrix=matrix, alpha=False)


def _validate_pdf_path(pdf_path: Path) -> Path:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF does not exist: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"Input path is not a file: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Input file must have a .pdf extension: {pdf_path}")
    return pdf_path.resolve()


def _ensure_output_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    images_dir = output_dir / "converted_images"
    pages_dir = images_dir / "pages"
    images_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, images_dir, pages_dir


def _page_image_stem(page_index: int) -> str:
    return f"page_{page_index + 1:04d}"


def _build_page_manifest(
    *,
    page_index: int,
    pdf_width_pt: float,
    pdf_height_pt: float,
    rendered_w: int,
    rendered_h: int,
    saved_w: int,
    saved_h: int,
    image_relative_path: str,
    dpi: float,
    zoom: float,
    rotation_deg: int,
    allow_rotated_pages: bool,
) -> dict[str, Any]:
    sx, sy = scales_from_dimensions(pdf_width_pt, pdf_height_pt, saved_w, saved_h)
    mapping: dict[str, Any] = {
        "image_to_pdf_scale_x": sx,
        "image_to_pdf_scale_y": sy,
        "formula": {
            "pdf_x": "x * image_to_pdf_scale_x",
            "pdf_y": "pdf_height_pt - ((y + h) * image_to_pdf_scale_y)",
            "pdf_w": "w * image_to_pdf_scale_x",
            "pdf_h": "h * image_to_pdf_scale_y",
        },
    }
    if rotation_deg != 0 and allow_rotated_pages:
        mapping["status"] = "unsupported_simple_linear"

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "page_index": page_index,
        "pdf": {
            "width_pt": pdf_width_pt,
            "height_pt": pdf_height_pt,
            "origin": "bottom-left",
        },
        "image": {
            "path": image_relative_path,
            "format": "png",
            "width_px": saved_w,
            "height_px": saved_h,
            "origin": "top-left",
            "rendered_image_width_px": rendered_w,
            "rendered_image_height_px": rendered_h,
            "saved_image_width_px": saved_w,
            "saved_image_height_px": saved_h,
        },
        "rendering": {
            "dpi": dpi,
            "zoom": zoom,
            "library": "pymupdf",
            "pymupdf_version": _pymupdf_version(),
        },
        "mapping": mapping,
    }
    if rotation_deg != 0:
        manifest["rotation_deg"] = rotation_deg
    return manifest


def convert_pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: float = 200.0,
    *,
    overwrite: bool = False,
    allow_rotated_pages: bool = False,
) -> dict[str, Any]:
    """
    Render every page of ``pdf_path`` to PNG under ``output_dir/converted_images``
    and write per-page JSON plus folder and document manifests.

    Returns a dict with absolute paths and a ``pages`` list of :class:`PageRecord`.
    """
    pdf_file = _validate_pdf_path(Path(pdf_path))
    out_root = Path(output_dir).expanduser().resolve()
    _, images_dir, pages_dir = _ensure_output_dirs(out_root)

    if dpi <= 0:
        raise ValueError("--dpi must be positive.")

    # Sample output collision check (lightweight).
    if not overwrite:
        existing = list(images_dir.glob("page_*.png"))
        if existing:
            raise FileExistsError(
                f"Output directory already contains page images under {images_dir}. "
                "Pass overwrite=True or use --overwrite."
            )

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    page_summaries: list[PageRecord] = []
    doc = fitz.open(pdf_file)

    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            rotation_deg = int(page.rotation)
            if rotation_deg != 0 and not allow_rotated_pages:
                raise ValueError(
                    f"Page {page_index} has rotation={rotation_deg} degrees. "
                    "Simple linear image↔PDF mapping may be invalid for stamping. "
                    "Normalize rotation in the PDF or pass allow_rotated_pages=True / "
                    "--allow-rotated-pages (manifest will mark mapping.status)."
                )

            rect = page.rect
            pdf_width_pt = float(rect.width)
            pdf_height_pt = float(rect.height)

            pix = _render_page_pixmap(page, matrix)
            rendered_w = int(pix.width)
            rendered_h = int(pix.height)

            if rotation_deg == 0:
                _assert_uniform_scale(pdf_width_pt, pdf_height_pt, rendered_w, rendered_h)

            stem = _page_image_stem(page_index)
            image_name = f"{stem}.png"
            image_abs_path = images_dir / image_name
            # Path relative to output_dir root for portability.
            image_rel_to_out = f"converted_images/{image_name}"

            pix.save(str(image_abs_path))

            saved_w = rendered_w
            saved_h = rendered_h

            page_manifest = _build_page_manifest(
                page_index=page_index,
                pdf_width_pt=pdf_width_pt,
                pdf_height_pt=pdf_height_pt,
                rendered_w=rendered_w,
                rendered_h=rendered_h,
                saved_w=saved_w,
                saved_h=saved_h,
                image_relative_path=image_rel_to_out,
                dpi=dpi,
                zoom=zoom,
                rotation_deg=rotation_deg,
                allow_rotated_pages=allow_rotated_pages,
            )

            page_json_name = f"{stem}.json"
            page_json_abs = pages_dir / page_json_name
            page_manifest_rel = f"converted_images/pages/{page_json_name}"

            page_json_abs.write_text(json.dumps(page_manifest, indent=2) + "\n", encoding="utf-8")

            page_summaries.append(
                {
                    "page_index": page_index,
                    "image_path": image_rel_to_out,
                    "page_manifest_path": page_manifest_rel,
                }
            )

            LOG.info(
                "Page %d: %gx%gpt → pixmap %dx%d → %s",
                page_index,
                pdf_width_pt,
                pdf_height_pt,
                rendered_w,
                rendered_h,
                image_abs_path.name,
            )
    finally:
        doc.close()

    images_manifest_path = images_dir / "manifest.json"
    document_manifest_path = out_root / "document_manifest.json"

    images_manifest = {
        "manifest_version": MANIFEST_VERSION,
        "source_pdf": str(pdf_file),
        "output_dir": str(out_root),
        "dpi": dpi,
        "pages": page_summaries,
    }
    images_manifest_path.write_text(json.dumps(images_manifest, indent=2) + "\n", encoding="utf-8")

    document_manifest = {
        "manifest_version": MANIFEST_VERSION,
        "source_pdf": str(pdf_file),
        "output_dir": str(out_root),
        "dpi": dpi,
        "converted_images_manifest": "converted_images/manifest.json",
        "pages": [
            {
                "page_index": p["page_index"],
                "page_manifest_path": p["page_manifest_path"],
                "image_path": p["image_path"],
            }
            for p in page_summaries
        ],
    }
    document_manifest_path.write_text(
        json.dumps(document_manifest, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "output_dir": str(out_root),
        "source_pdf": str(pdf_file),
        "document_manifest": str(document_manifest_path),
        "images_manifest": str(images_manifest_path),
        "pages": page_summaries,
    }


def run_self_check() -> None:
    """Create a one-page PDF in a temp directory, convert, and verify bbox mapping."""
    dpi = 72.0
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "self_check.pdf"
        out_dir = Path(tmp) / "out"

        doc = fitz.open()
        try:
            page = doc.new_page(width=612, height=792)
            # Filled rectangle in PDF space (origin bottom-left).
            page.draw_rect(fitz.Rect(100, 100, 200, 200), color=(0, 0, 0), fill=(0.2, 0.4, 0.8))
            doc.save(str(pdf_path))
        finally:
            doc.close()

        result = convert_pdf_to_images(str(pdf_path), str(out_dir), dpi=dpi, overwrite=True)
        page_manifest_path = out_dir / result["pages"][0]["page_manifest_path"]
        manifest = json.loads(page_manifest_path.read_text(encoding="utf-8"))

        pix_w = manifest["image"]["saved_image_width_px"]
        pix_h = manifest["image"]["saved_image_height_px"]

        # Full-page bbox should map to full-page PDF rectangle (within pixel quantization).
        full = map_image_bbox_to_pdf((0.0, 0.0, float(pix_w), float(pix_h)), manifest)
        assert math.isclose(full["pdf_x"], 0.0, abs_tol=1e-6)
        assert math.isclose(full["pdf_y"], 0.0, abs_tol=1e-6)
        assert math.isclose(full["pdf_w"], 612.0, rel_tol=1e-5, abs_tol=0.05)
        assert math.isclose(full["pdf_h"], 792.0, rel_tol=1e-5, abs_tol=0.05)

        # Inverse map: PDF lower-left (100, 100) with 10x10 pt box → image top-left bbox.
        sx = manifest["mapping"]["image_to_pdf_scale_x"]
        sy = manifest["mapping"]["image_to_pdf_scale_y"]
        pdf_ll_x, pdf_ll_y = 100.0, 100.0
        pdf_w_pt, pdf_h_pt = 10.0, 10.0
        img_x = pdf_ll_x / sx
        img_y = (792.0 - pdf_ll_y - pdf_h_pt) / sy
        img_w = pdf_w_pt / sx
        img_h = pdf_h_pt / sy
        inner = map_image_bbox_to_pdf((img_x, img_y, img_w, img_h), manifest)
        assert math.isclose(inner["pdf_x"], pdf_ll_x, rel_tol=0, abs_tol=0.05)
        assert math.isclose(inner["pdf_y"], pdf_ll_y, rel_tol=0, abs_tol=0.05)
        assert math.isclose(inner["pdf_w"], pdf_w_pt, rel_tol=0, abs_tol=0.05)
        assert math.isclose(inner["pdf_h"], pdf_h_pt, rel_tol=0, abs_tol=0.05)

    LOG.info("Self-check passed.")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render PDF pages to PNG with manifests for exact image↔PDF mapping.",
    )
    p.add_argument("pdf_path", nargs="?", help="Path to the input PDF.")
    p.add_argument(
        "--output-dir",
        "-o",
        required=False,
        help="Output root (required unless --self-check).",
    )
    p.add_argument("--dpi", type=float, default=200.0, help="Rasterization DPI (default 200).")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a directory that already has page_*.png files.",
    )
    p.add_argument(
        "--allow-rotated-pages",
        action="store_true",
        help="Allow non-zero page.rotation (manifest marks linear mapping as unsupported).",
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="Run internal round-trip check and exit (does not require pdf_path).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.self_check:
        run_self_check()
        return 0

    if not args.pdf_path:
        LOG.error("pdf_path is required unless --self-check is set.")
        return 2
    if not args.output_dir:
        LOG.error("--output-dir is required unless --self-check is set.")
        return 2

    try:
        convert_pdf_to_images(
            args.pdf_path,
            args.output_dir,
            dpi=args.dpi,
            overwrite=args.overwrite,
            allow_rotated_pages=args.allow_rotated_pages,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        LOG.error("%s", exc)
        return 1
    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
