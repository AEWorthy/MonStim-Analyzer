from __future__ import annotations

import base64
import json
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools.catalog_signing import signed_wrapper
from tools.publish_release_catalog import update_catalog, validate_archive


def _release_zip(path: Path, version: str = "0.7.0") -> None:
    root = f"MonStim_Analyzer_v{version}-WIN/"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(root + "MonStim Analyzer.exe", b"application")
        archive.writestr(root + "MonStim Updater.exe", b"updater")
        archive.writestr(root + "monstim-release.json", json.dumps({"version": version, "executable": "MonStim Analyzer.exe"}))


def test_validate_archive_requires_matching_manifest_and_updater(tmp_path):
    archive = tmp_path / "release.zip"
    _release_zip(archive)

    validate_archive(archive, "0.7.0")

    with pytest.raises(ValueError, match=r"not '0\.8\.0'"):
        validate_archive(archive, "0.8.0")


def test_update_catalog_replaces_matching_release_and_sorts_versions():
    catalog = {
        "schema_version": "1.0",
        "releases": [
            {"version": "0.6.0", "channel": "beta", "platform": "windows-x86_64"},
            {"version": "0.7.0", "channel": "stable", "platform": "windows-x86_64"},
        ],
    }
    entry = {"version": "0.7.0", "channel": "beta", "platform": "windows-x86_64", "sha256": "a" * 64}

    updated = update_catalog(catalog, entry)

    assert next(item for item in updated["releases"] if item["channel"] == "beta" and item["version"] == "0.7.0") == entry
    assert [item["version"] for item in updated["releases"]].count("0.7.0") == 2
    assert next(item for item in updated["releases"] if item["channel"] == "beta" and item["version"] == "0.7.0") == entry


def test_signed_wrapper_is_verifiable_and_contains_no_private_key(tmp_path):
    private_key = Ed25519PrivateKey.generate()
    key_path = tmp_path / "release.key"
    key_path.write_bytes(private_key.private_bytes_raw())
    catalog = {"schema_version": "1.0", "releases": []}

    wrapper = signed_wrapper(catalog, key_path)

    payload = base64.b64decode(wrapper["payload"])
    private_key.public_key().verify(base64.b64decode(wrapper["signature"]), payload)
    assert key_path.read_bytes().hex() not in json.dumps(wrapper)
