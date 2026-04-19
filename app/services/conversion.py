"""Bridge to the PDF→image conversion library (scripts on ``sys.path``)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def get_converter_module() -> ModuleType:
    """Import ``convert_pdf_pages_for_grounding`` from ``scripts/``."""
    import convert_pdf_pages_for_grounding as conv  # noqa: PLC0415

    return conv


def run_convert_pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: float,
    *,
    overwrite: bool,
    allow_rotated_pages: bool,
) -> dict[str, Any]:
    """Synchronous wrapper used from ``asyncio.to_thread``."""
    conv = get_converter_module()
    return conv.convert_pdf_to_images(
        pdf_path,
        output_dir,
        dpi=dpi,
        overwrite=overwrite,
        allow_rotated_pages=allow_rotated_pages,
    )
