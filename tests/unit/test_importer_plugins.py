from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from monstim_gui.plugins import (
    PluginError,
    catalog_check_due,
    compatibility_problems,
    install_pack,
    read_pack_manifest,
    verify_catalog,
)
from monstim_gui.provenance import write_provenance_sidecar


def _manifest(**updates):
    manifest = {
        "id": "example-importer",
        "name": "Example importer",
        "version": "0.1.0",
        "entry_point": "importer.py:PLUGIN",
        "app_requires": ">=0.6,<0.8",
        "api_requires": ">=2,<3",
        "source_kind": "file",
        "file_patterns": ["*.csv"],
        "runtime_requirements": ["numpy", "monstim-plugin-api"],
    }
    manifest.update(updates)
    return manifest


def _pack(path: Path, manifest: dict, extra: dict[str, str] | None = None) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("importer.py", "PLUGIN = object()")
        for name, content in (extra or {}).items():
            archive.writestr(name, content)
    return path


def test_installs_compatible_pack_in_user_plugin_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MONSTIM_PLUGIN_DIR", str(tmp_path / "user"))
    pack = _pack(tmp_path / "example.zip", _manifest())

    manifest = install_pack(pack)

    assert manifest.id == "example-importer"
    assert (tmp_path / "user" / "plugins" / "example-importer" / "0.1.0" / "manifest.json").is_file()


def test_rejects_runtime_dependency_missing_from_frozen_capabilities(tmp_path):
    pack = _pack(tmp_path / "unsupported.zip", _manifest(runtime_requirements=["vendor-native-sdk"]))

    manifest = read_pack_manifest(pack)

    assert "vendor-native-sdk" in " ".join(compatibility_problems(manifest))


def test_rejects_path_traversal_archive(tmp_path):
    pack = _pack(tmp_path / "unsafe.zip", _manifest(), {"../outside.py": "unsafe"})

    with pytest.raises(PluginError, match="unsafe path"):
        read_pack_manifest(pack)


def test_export_provenance_is_a_separate_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("MONSTIM_PLUGIN_DIR", str(tmp_path / "user"))
    output = tmp_path / "results.csv"
    output.write_text("value\n1\n", encoding="utf-8")

    sidecar = write_provenance_sidecar(output, {"plot_type": "Example"})

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sidecar.name == "results.csv.provenance.json"
    assert payload["analysis_context"]["plot_type"] == "Example"
    assert payload["software"]["name"] == "MonStim Analyzer"


def test_catalog_rejects_invalid_signature():
    with pytest.raises(PluginError, match="signature"):
        verify_catalog(b'{"plugins": []}', "not-a-valid-signature")


def test_catalog_check_is_due_once_daily():
    now = datetime(2026, 9, 15, tzinfo=UTC)

    assert catalog_check_due(None, now=now)
    assert not catalog_check_due((now - timedelta(hours=23)).isoformat(), now=now)
    assert catalog_check_due((now - timedelta(days=1)).isoformat(), now=now)
    assert catalog_check_due("not-a-time", now=now)
