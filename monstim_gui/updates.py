"""Signed, data-preserving update discovery and side-by-side staging.

This module never edits experiment data, settings, or plug-in directories.
It only writes a separately configurable application-version cache.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from packaging.version import Version

from monstim_gui.version import VERSION

UPDATE_MANIFEST_URL = "https://AEWorthy.github.io/MonStim-Analyzer/updates.json"
# Dedicated offline application-update catalog verification key. The matching
# private key is maintained outside this repository.
UPDATE_PUBLIC_KEY_B64 = "8eski4IH85R06ucPskWrInV08UeFt87nGMA2rw/0uo0="
UPDATE_CHECK_INTERVAL = timedelta(days=1)
MAX_RELEASE_BYTES = 2 * 1024 * 1024 * 1024


class UpdateError(RuntimeError):
    """An update could not be safely checked, staged, or activated."""


@dataclass(frozen=True)
class UpdateRelease:
    version: str
    channel: str
    platform: str
    asset_url: str
    sha256: str
    notes_url: str
    minimum_updater_version: str = "0"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> UpdateRelease:
        required = {"version", "channel", "platform", "asset_url", "sha256", "notes_url"}
        missing = sorted(required - value.keys())
        if missing:
            raise UpdateError("Update manifest entry is missing: " + ", ".join(missing))
        try:
            Version(str(value["version"]))
        except ValueError as exc:
            raise UpdateError("Update manifest contains an invalid version.") from exc
        digest = str(value["sha256"]).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise UpdateError("Update manifest contains an invalid SHA-256 digest.")
        return cls(
            version=str(value["version"]),
            channel=str(value["channel"]),
            platform=str(value["platform"]),
            asset_url=str(value["asset_url"]),
            sha256=digest,
            notes_url=str(value["notes_url"]),
            minimum_updater_version=str(value.get("minimum_updater_version", "0")),
        )


def update_root() -> Path:
    """Return the isolated per-user location for application binaries only."""
    override = os.environ.get("MONSTIM_UPDATE_ROOT")
    base = Path(override) if override else Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "MonStim Analyzer" / "updates"
    root = base.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_update_catalog(payload: bytes, signature_b64: str) -> dict[str, Any]:
    if not UPDATE_PUBLIC_KEY_B64:
        raise UpdateError("Application update checks are not configured in this build.")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        key = Ed25519PublicKey.from_public_bytes(base64.b64decode(UPDATE_PUBLIC_KEY_B64))
        key.verify(base64.b64decode(signature_b64), payload)
        catalog = json.loads(payload)
    except UpdateError:
        raise
    except Exception as exc:
        raise UpdateError("The application update catalog signature could not be verified.") from exc
    if catalog.get("schema_version") != "1.0" or not isinstance(catalog.get("releases"), list):
        raise UpdateError("The application update catalog has an unsupported schema.")
    return catalog


def fetch_update_catalog(url: str = UPDATE_MANIFEST_URL) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            wrapper = json.loads(response.read())
        return verify_update_catalog(base64.b64decode(wrapper["payload"]), wrapper["signature"])
    except UpdateError:
        raise
    except Exception as exc:
        raise UpdateError("Could not check for application updates. You may be offline, or the update catalog is unavailable.") from exc


def update_check_due(last_checked: str | None, *, now: datetime | None = None) -> bool:
    if not last_checked:
        return True
    try:
        checked = datetime.fromisoformat(last_checked)
        if checked.tzinfo is None:
            checked = checked.replace(tzinfo=UTC)
    except ValueError:
        return True
    return (now or datetime.now(UTC)) - checked.astimezone(UTC) >= UPDATE_CHECK_INTERVAL


def available_update(catalog: dict[str, Any], *, channel: str = "beta", platform: str = "windows-x86_64") -> UpdateRelease | None:
    candidates = []
    for raw in catalog["releases"]:
        try:
            release = UpdateRelease.from_dict(raw)
            if release.channel == channel and release.platform == platform and Version(release.version) > Version(VERSION):
                candidates.append(release)
        except TypeError, UpdateError:
            continue
    return max(candidates, key=lambda release: Version(release.version), default=None)


def download_update(release: UpdateRelease) -> Path:
    """Download a release to a temporary ZIP after verifying its advertised digest."""
    with urllib.request.urlopen(release.asset_url, timeout=30) as response, tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as stream:
        shutil.copyfileobj(response, stream)
        path = Path(stream.name)
    if path.stat().st_size > MAX_RELEASE_BYTES or _sha256(path).lower() != release.sha256:
        path.unlink(missing_ok=True)
        raise UpdateError("Downloaded release did not match the signed catalog checksum.")
    return path


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    members = archive.infolist()
    if sum(member.file_size for member in members) > MAX_RELEASE_BYTES:
        raise UpdateError("Update archive is too large to extract safely.")
    for member in members:
        member_path = Path(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise UpdateError("Update archive contains an unsafe path.")
    archive.extractall(destination)


def _release_root(staging: Path) -> Path:
    marker = list(staging.rglob("monstim-release.json"))
    if len(marker) != 1:
        raise UpdateError("Update archive must contain exactly one monstim-release.json manifest.")
    root = marker[0].parent
    try:
        manifest = json.loads(marker[0].read_text(encoding="utf-8"))
        executable = root / str(manifest["executable"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise UpdateError("Update archive release manifest is invalid.") from exc
    if not executable.is_file():
        raise UpdateError("Update archive does not contain its declared executable.")
    return root


def stage_update(release: UpdateRelease, archive: Path) -> Path:
    """Extract a verified update beside existing versions without activating it."""
    root = update_root()
    versions = root / "versions"
    destination = versions / release.version
    if destination.exists():
        return destination
    staging = versions / f".{release.version}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        staging.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(archive) as zip_file:
            _safe_extract(zip_file, staging)
        release_root = _release_root(staging)
        if json.loads((release_root / "monstim-release.json").read_text(encoding="utf-8"))["version"] != release.version:
            raise UpdateError("Update archive version does not match the signed catalog.")
        os.replace(release_root, destination)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def activate_update(version: str) -> Path:
    """Atomically select a staged version; this cannot alter research data."""
    root = update_root()
    destination = root / "versions" / version
    if not destination.is_dir():
        raise UpdateError("Requested update version has not been staged.")
    current = root / "current.json"
    if current.exists():
        shutil.copy2(current, root / "rollback.json")
    _atomic_json(current, {"version": version, "activated_at": datetime.now(UTC).isoformat()})
    return destination


def rollback_update() -> str:
    root = update_root()
    rollback = root / "rollback.json"
    if not rollback.is_file():
        raise UpdateError("No prior application version is available for rollback.")
    payload = json.loads(rollback.read_text(encoding="utf-8"))
    version = str(payload["version"])
    activate_update(version)
    return version


def staged_executable(version: str) -> Path:
    """Return the release-declared executable for a staged version."""
    release_root = update_root() / "versions" / version
    try:
        manifest = json.loads((release_root / "monstim-release.json").read_text(encoding="utf-8"))
        executable = release_root / str(manifest["executable"])
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise UpdateError("Staged update is missing its release manifest.") from exc
    if not executable.is_file():
        raise UpdateError("Staged update is missing its declared executable.")
    return executable


def launch_staged_update(version: str, *, wait_pid: int) -> None:
    """Start the bundled helper; it activates a staged version only after exit."""
    if not getattr(sys, "frozen", False):
        raise UpdateError("Install-on-restart is available only in the packaged Windows application.")
    helper = Path(sys.executable).parent / "MonStim Updater.exe"
    if not helper.is_file():
        raise UpdateError("This MonStim installation does not include the update helper.")
    subprocess.Popen([str(helper), "--version", version, "--wait-pid", str(wait_pid), "--restart"], close_fds=True)


def launch_selected_update() -> bool:
    """Launch a selected newer managed version when an older shortcut starts.

    This is intentionally a read-only bootstrap until the new executable has
    been verified and activated by the updater helper.
    """
    try:
        current = json.loads((update_root() / "current.json").read_text(encoding="utf-8"))
        version = str(current["version"])
        if Version(version) <= Version(VERSION):
            return False
        subprocess.Popen([str(staged_executable(version))], close_fds=True)
        return True
    except OSError, KeyError, TypeError, ValueError, UpdateError, json.JSONDecodeError:
        return False
