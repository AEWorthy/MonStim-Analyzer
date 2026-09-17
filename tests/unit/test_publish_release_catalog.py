from __future__ import annotations

import base64
import json
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools.catalog_signing import signed_wrapper
from tools.publish_release_catalog import DEMO_ARCHIVE_MEMBER, update_catalog, validate_archive


def _release_zip(path: Path, version: str = "0.7.1") -> None:
    root = f"MonStim_Analyzer_v{version}-WIN/"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(root + "MonStim Analyzer.exe", b"application")
        archive.writestr(root + "_internal/MonStim Updater.exe", b"updater")
        archive.writestr(root + "_internal/LICENSE", b"license")
        archive.writestr(root + "_internal/monstim-release.json", json.dumps({"version": version, "executable": "MonStim Analyzer.exe"}))
        archive.writestr(root + DEMO_ARCHIVE_MEMBER, b"fictional demo data")


def test_validate_archive_requires_matching_manifest_and_updater(tmp_path):
    archive = tmp_path / "release.zip"
    _release_zip(archive)

    validate_archive(archive, "0.7.1")

    with pytest.raises(ValueError, match=r"not '0\.8\.0'"):
        validate_archive(archive, "0.8.0")


def test_validate_archive_rejects_root_level_updater(tmp_path):
    archive = tmp_path / "release-with-root-updater.zip"
    _release_zip(archive)
    with zipfile.ZipFile(archive, "a") as zip_file:
        zip_file.writestr("MonStim_Analyzer_v0.7.1-WIN/MonStim Updater.exe", b"duplicate updater")

    with pytest.raises(ValueError, match="must not contain a root-level"):
        validate_archive(archive, "0.7.1")


def test_validate_archive_requires_internal_license(tmp_path):
    archive = tmp_path / "release-without-license.zip"
    root = "MonStim_Analyzer_v0.7.1-WIN/"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr(root + "MonStim Analyzer.exe", b"application")
        zip_file.writestr(root + "_internal/MonStim Updater.exe", b"updater")
        zip_file.writestr(root + "_internal/monstim-release.json", json.dumps({"version": "0.7.1", "executable": "MonStim Analyzer.exe"}))
        zip_file.writestr(root + DEMO_ARCHIVE_MEMBER, b"fictional demo data")

    with pytest.raises(ValueError, match="internal LICENSE"):
        validate_archive(archive, "0.7.1")


def test_validate_archive_requires_bundled_demo_data(tmp_path):
    archive = tmp_path / "release-without-demos.zip"
    root = "MonStim_Analyzer_v0.7.1-WIN/"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr(root + "MonStim Analyzer.exe", b"application")
        zip_file.writestr(root + "_internal/MonStim Updater.exe", b"updater")
        zip_file.writestr(root + "_internal/LICENSE", b"license")
        zip_file.writestr(root + "_internal/monstim-release.json", json.dumps({"version": "0.7.1", "executable": "MonStim Analyzer.exe"}))

    with pytest.raises(ValueError, match="bundled synthetic-demo archive"):
        validate_archive(archive, "0.7.1")


def test_validate_archive_rejects_demo_data_outside_internal_bundle(tmp_path):
    archive = tmp_path / "release-with-root-demo-data.zip"
    root = "MonStim_Analyzer_v0.7.1-WIN/"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr(root + "MonStim Analyzer.exe", b"application")
        zip_file.writestr(root + "_internal/MonStim Updater.exe", b"updater")
        zip_file.writestr(root + "_internal/LICENSE", b"license")
        zip_file.writestr(root + "_internal/monstim-release.json", json.dumps({"version": "0.7.1", "executable": "MonStim Analyzer.exe"}))
        zip_file.writestr(root + DEMO_ARCHIVE_MEMBER.removeprefix("_internal/"), b"fictional demo data")

    with pytest.raises(ValueError, match="bundled synthetic-demo archive"):
        validate_archive(archive, "0.7.1")


def test_update_catalog_replaces_matching_release_and_sorts_versions():
    catalog = {
        "schema_version": "1.0",
        "releases": [
            {"version": "0.6.0", "channel": "beta", "platform": "windows-x86_64"},
            {"version": "0.7.1", "channel": "stable", "platform": "windows-x86_64"},
        ],
    }
    entry = {"version": "0.7.1", "channel": "beta", "platform": "windows-x86_64", "sha256": "a" * 64}

    updated = update_catalog(catalog, entry)

    assert next(item for item in updated["releases"] if item["channel"] == "beta" and item["version"] == "0.7.1") == entry
    assert [item["version"] for item in updated["releases"]].count("0.7.1") == 2
    assert next(item for item in updated["releases"] if item["channel"] == "beta" and item["version"] == "0.7.1") == entry


def test_signed_wrapper_is_verifiable_and_contains_no_private_key(tmp_path):
    private_key = Ed25519PrivateKey.generate()
    key_path = tmp_path / "release.key"
    key_path.write_bytes(private_key.private_bytes_raw())
    catalog = {"schema_version": "1.0", "releases": []}

    wrapper = signed_wrapper(catalog, key_path)

    payload = base64.b64decode(wrapper["payload"])
    private_key.public_key().verify(base64.b64decode(wrapper["signature"]), payload)
    assert key_path.read_bytes().hex() not in json.dumps(wrapper)
