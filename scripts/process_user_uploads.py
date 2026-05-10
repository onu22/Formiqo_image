#!/usr/bin/env python3
"""Scan ``data/user-uploads`` and process each PDF through the pipeline router (CLI)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Run from repo root: python scripts/process_user_uploads.py
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.dependencies import get_settings  # noqa: E402
from app.services.pdf_pipeline import scan_and_process_user_uploads  # noqa: E402


def main() -> None:
    settings = get_settings()
    results = scan_and_process_user_uploads(settings=settings)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
