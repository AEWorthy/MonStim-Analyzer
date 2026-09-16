"""Citation helpers for MonStim outputs and reports."""

from __future__ import annotations

from datetime import UTC, datetime

from monstim_gui.version import VERSION

CITATION_URL = "https://github.com/AEWorthy/MonStim-Analyzer"


def citation_text() -> str:
    return f"Worthy, A. ({datetime.now(UTC).year}). MonStim Analyzer (Version {VERSION}) [Computer software]. {CITATION_URL}"
