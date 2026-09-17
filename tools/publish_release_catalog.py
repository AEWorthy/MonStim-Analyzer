"""Prepare, checksum, and sign a MonStim Windows-release catalog entry.

Run this only after building the final ZIP.  It validates the package manifest,
computes the checksum itself, updates the unsigned source catalog, and writes
the signed Pages catalog.  The private key must stay outside the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from packaging.version import InvalidVersion, Version

from tools.catalog_signing import atomic_write_text, sha256_file, signed_wrapper

REPOSITORY = "AEWorthy/MonStim-Analyzer"
VALID_CHANNELS = {"beta", "stable"}
VALID_PLATFORMS = {"windows-x86_64"}
# PyInstaller places application data files beneath ``_internal`` in the
# one-folder Windows bundle. The synthetic demo archive is collected as part
# of ``docs``, so this is its path relative to the release directory.
DEMO_ARCHIVE_MEMBER = "_internal/docs/resources/demo_experiments/monstim-synthetic-protocol-demos.zip"


def _https_url(value: str, name: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{name} must be a complete HTTPS URL")
    return value


def _github_release_url(value: str, name: str) -> str:
    value = _https_url(value, name)
    required_prefix = f"https://github.com/{REPOSITORY}/releases/"
    if not value.startswith(required_prefix):
        raise ValueError(f"{name} must point to an official {REPOSITORY} GitHub Release")
    return value


def validate_archive(archive: Path, version: str) -> None:
    """Check that a release ZIP contains required updater and demo resources."""
    try:
        with zipfile.ZipFile(archive) as zip_file:
            bad_member = zip_file.testzip()
            if bad_member:
                raise ValueError(f"ZIP integrity check failed at {bad_member!r}")
            manifests = [name for name in zip_file.namelist() if name.rstrip("/").endswith("monstim-release.json")]
            if len(manifests) != 1:
                raise ValueError("release ZIP must contain exactly one monstim-release.json manifest")
            manifest_name = manifests[0]
            manifest_suffix = "_internal/monstim-release.json"
            if not manifest_name.endswith(manifest_suffix):
                raise ValueError("release ZIP manifest must be stored in _internal")
            manifest = json.loads(zip_file.read(manifest_name).decode("utf-8"))
            if str(manifest.get("version")) != version:
                raise ValueError(f"release manifest version is {manifest.get('version')!r}, not {version!r}")
            executable = manifest.get("executable")
            if not isinstance(executable, str) or not executable.endswith(".exe"):
                raise ValueError("release manifest must name a Windows executable")
            prefix = manifest_name.removesuffix(manifest_suffix)
            if prefix + executable not in zip_file.namelist():
                raise ValueError(f"release ZIP is missing the manifest executable {executable!r}")
            updater_member = prefix + "_internal/MonStim Updater.exe"
            if updater_member not in zip_file.namelist():
                raise ValueError("release ZIP is missing the internal MonStim Updater.exe helper")
            if prefix + "_internal/LICENSE" not in zip_file.namelist():
                raise ValueError("release ZIP is missing the internal LICENSE file")
            if prefix + "monstim-release.json" in zip_file.namelist():
                raise ValueError("release ZIP must not contain a root-level monstim-release.json")
            if prefix + "MonStim Updater.exe" in zip_file.namelist():
                raise ValueError("release ZIP must not contain a root-level MonStim Updater.exe")
            if prefix + DEMO_ARCHIVE_MEMBER not in zip_file.namelist():
                raise ValueError("release ZIP is missing the bundled synthetic-demo archive")
    except zipfile.BadZipFile as exc:
        raise ValueError("release archive is not a valid ZIP file") from exc


def update_catalog(catalog: dict, entry: dict) -> dict:
    if catalog.get("schema_version") != "1.0" or not isinstance(catalog.get("releases"), list):
        raise ValueError("update catalog must have schema_version '1.0' and a releases list")
    key = (entry["version"], entry["channel"], entry["platform"])
    releases = [
        candidate for candidate in catalog["releases"] if (candidate.get("version"), candidate.get("channel"), candidate.get("platform")) != key
    ]
    releases.append(entry)
    releases.sort(key=lambda candidate: Version(str(candidate["version"])), reverse=True)
    return {"schema_version": "1.0", "releases": releases}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and sign one official MonStim application-update release.")
    parser.add_argument("--version", required=True, help="Release version without a leading v, for example 0.7.0")
    parser.add_argument("--archive", type=Path, required=True, help="Final Windows release ZIP")
    parser.add_argument("--asset-url", required=True, help="HTTPS URL of that ZIP in the official GitHub Release")
    parser.add_argument("--notes-url", required=True, help="HTTPS URL of the official GitHub Release notes")
    parser.add_argument("--private-key", type=Path, required=True, help="Application-update private key outside the repository")
    parser.add_argument("--channel", choices=sorted(VALID_CHANNELS), default="beta")
    parser.add_argument("--platform", choices=sorted(VALID_PLATFORMS), default="windows-x86_64")
    parser.add_argument("--minimum-updater-version", default="0")
    parser.add_argument("--catalog", type=Path, default=Path("tools/update_catalog.template.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/updates.json"))
    parser.add_argument("--checksums-output", type=Path, help="Output SHA256SUMS.txt (defaults beside the ZIP)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and display changes without writing any files")
    args = parser.parse_args()

    try:
        version = str(Version(args.version))
    except InvalidVersion:
        parser.error("--version must be a valid PEP 440 version")
    if version != args.version:
        parser.error("--version must be normalized already (for example, use 0.7.0, not v0.7.0)")
    if not args.archive.is_file():
        parser.error(f"release archive does not exist: {args.archive}")
    try:
        asset_url = _github_release_url(args.asset_url, "--asset-url")
        notes_url = _github_release_url(args.notes_url, "--notes-url")
        validate_archive(args.archive, version)
        catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
        digest = sha256_file(args.archive)
        entry = {
            "version": version,
            "channel": args.channel,
            "platform": args.platform,
            "asset_url": asset_url,
            "sha256": digest,
            "notes_url": notes_url,
            "minimum_updater_version": args.minimum_updater_version,
        }
        updated_catalog = update_catalog(catalog, entry)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    checksums_output = args.checksums_output or args.archive.with_name("SHA256SUMS.txt")
    checksum_text = f"{digest} *{args.archive.name}\n"
    if args.dry_run:
        print(json.dumps(entry, indent=2))
        print(f"Would write {args.catalog}, {args.output}, and {checksums_output}.")
        return 0

    try:
        # Validate the key and generate the signature before changing either
        # checked-in catalog file. A missing/wrong key must leave the source
        # catalog untouched.
        wrapper = signed_wrapper(updated_catalog, args.private_key)
        atomic_write_text(args.catalog, json.dumps(updated_catalog, indent=2) + "\n")
        atomic_write_text(checksums_output, checksum_text)
        atomic_write_text(args.output, json.dumps(wrapper, indent=2) + "\n")
    except (OSError, ValueError) as exc:
        print(f"error: could not publish release catalog: {exc}", file=sys.stderr)
        return 1
    print(f"Prepared {version} ({args.channel}, {args.platform}).")
    print(f"SHA-256: {digest}")
    print(f"Checksum file: {checksums_output}")
    print(f"Unsigned catalog: {args.catalog}")
    print(f"Signed Pages catalog: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
