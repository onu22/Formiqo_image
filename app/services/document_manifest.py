"""Write provider/model metadata onto ``document_manifest.json`` after conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.jobs import provider_model_dir_name, read_document_manifest


def write_provider_metadata_to_document_manifest(
    *,
    output_dir: Path,
    provider: str,
    model: str,
) -> dict[str, Any]:
    manifest_path = output_dir / "document_manifest.json"
    doc = read_document_manifest(output_dir)
    doc["provider"] = provider
    doc["model"] = model
    doc["provider_model"] = provider_model_dir_name(provider, model)
    manifest_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc
