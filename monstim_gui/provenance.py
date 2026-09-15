"""Citation and machine-readable provenance for MonStim outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from monstim_gui.plugins import PLUGIN_API_VERSION, installed_manifests
from monstim_gui.version import VERSION

CITATION_URL = "https://github.com/AEWorthy/MonStim-Analyzer"


def citation_text() -> str:
    return f"Worthy, A. ({datetime.now(UTC).year}). MonStim Analyzer (Version {VERSION}) [Computer software]. {CITATION_URL}"


def provenance_payload(context: dict[str, Any] | None = None) -> dict[str, Any]:
    plugins = [{"id": manifest.id, "version": manifest.version} for manifest, _path, compatible in installed_manifests() if compatible]
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "software": {"name": "MonStim Analyzer", "version": VERSION, "citation": citation_text(), "citation_url": CITATION_URL},
        "plugin_api_version": PLUGIN_API_VERSION,
        "installed_importers": plugins,
        "analysis_context": context or {},
    }


def write_provenance_sidecar(output: str | Path, context: dict[str, Any] | None = None) -> Path:
    output_path = Path(output)
    sidecar = output_path.with_suffix(output_path.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(provenance_payload(context), indent=2), encoding="utf-8")
    return sidecar
