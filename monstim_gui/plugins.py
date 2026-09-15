"""Official importer add-on discovery, validation, and installation.

Add-ons are deliberately installed outside a PyInstaller distribution.  A pack
is executable Python, so this module treats installation as a trust boundary:
only a signed official catalog may offer automatic downloads, while local ZIP
files receive the same structural and compatibility checks before installation.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from packaging.specifiers import SpecifierSet
from packaging.version import Version
from PySide6.QtCore import QStandardPaths, QThread, Signal

from monstim_gui.version import VERSION

logger = logging.getLogger(__name__)

PLUGIN_API_VERSION = "2.0"
MIN_AUTO_DETECTION_CONFIDENCE = 80
OFFICIAL_CATALOG_URL = "https://AEWorthy.github.io/MonStim-Analyzer/plugins/catalog.json"
CATALOG_PUBLIC_KEY_B64 = "ssPrk5L4AQ4xq294lLyPJKHmvd8Awb6f03gYWxyrMW0="
MAX_PACK_BYTES = 50 * 1024 * 1024
CATALOG_CHECK_INTERVAL = timedelta(days=1)


class PluginError(RuntimeError):
    """A user-actionable add-on error."""


@dataclass(frozen=True)
class ImporterManifest:
    id: str
    name: str
    version: str
    entry_point: str
    app_requires: str
    api_requires: str
    source_kind: str
    file_patterns: tuple[str, ...]
    runtime_requirements: tuple[str, ...] = ()
    description: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ImporterManifest:
        required = {"id", "name", "version", "entry_point", "app_requires", "api_requires", "source_kind", "file_patterns"}
        missing = sorted(required - payload.keys())
        if missing:
            raise PluginError(f"Add-on manifest is missing: {', '.join(missing)}")
        if payload["source_kind"] not in {"file", "directory"}:
            raise PluginError("Add-on source_kind must be 'file' or 'directory'.")
        if not str(payload["id"]).replace("-", "").replace("_", "").isalnum():
            raise PluginError("Add-on id may contain only letters, numbers, hyphens, and underscores.")
        try:
            Version(str(payload["version"]))
            SpecifierSet(str(payload["app_requires"]))
            SpecifierSet(str(payload["api_requires"]))
        except Exception as exc:
            raise PluginError(f"Add-on manifest has an invalid version requirement: {exc}") from exc
        patterns = payload["file_patterns"]
        if not isinstance(patterns, list) or not all(isinstance(item, str) and item for item in patterns):
            raise PluginError("Add-on file_patterns must be a non-empty list of patterns.")
        runtime = payload.get("runtime_requirements", [])
        if not isinstance(runtime, list) or not all(isinstance(item, str) for item in runtime):
            raise PluginError("Add-on runtime_requirements must be a list of strings.")
        return cls(
            id=str(payload["id"]),
            name=str(payload["name"]),
            version=str(payload["version"]),
            entry_point=str(payload["entry_point"]),
            app_requires=str(payload["app_requires"]),
            api_requires=str(payload["api_requires"]),
            source_kind=str(payload["source_kind"]),
            file_patterns=tuple(patterns),
            runtime_requirements=tuple(runtime),
            description=str(payload.get("description", "")),
        )


class ImporterPlugin(Protocol):
    """Stable contract implemented by an importer pack's entry point."""

    def probe_source(self, source: Path) -> int: ...

    def validate_source(self, source: Path) -> None: ...

    def normalize_experiment(self, source: Path, is_canceled) -> Any: ...


def plugin_root() -> Path:
    override = os.environ.get("MONSTIM_PLUGIN_DIR")
    candidates = [override] if override else []
    candidates.extend(
        [
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation),
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation),
        ]
    )
    for base in candidates:
        if not base:
            continue
        root = Path(base) / "plugins"
        try:
            root.mkdir(parents=True, exist_ok=True)
            return root
        except OSError as exc:
            logger.warning("Could not create plugin directory %s: %s", root, exc)
    raise PluginError(
        "MonStim could not create a writable per-user plugin directory. Choose an approved folder through MONSTIM_PLUGIN_DIR or contact support."
    )


def available_runtime_capabilities() -> set[str]:
    """Dependencies guaranteed by the packaged application/plugin API."""
    return {"monstim-plugin-api", "numpy", "pandas", "scipy", "h5py", "pyside6", "pyqtgraph"}


def compatibility_problems(manifest: ImporterManifest) -> list[str]:
    problems = []
    if Version(VERSION) not in SpecifierSet(manifest.app_requires):
        problems.append(f"Requires MonStim {manifest.app_requires}; this installation is {VERSION}.")
    if Version(PLUGIN_API_VERSION) not in SpecifierSet(manifest.api_requires):
        problems.append(f"Requires plugin API {manifest.api_requires}; this installation provides {PLUGIN_API_VERSION}.")
    missing = sorted({item.lower() for item in manifest.runtime_requirements} - available_runtime_capabilities())
    if missing:
        problems.append("Requires runtime capability not bundled with this MonStim release: " + ", ".join(missing) + ".")
    return problems


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if sum(member.file_size for member in members) > MAX_PACK_BYTES:
        raise PluginError("Add-on archive is too large to install safely.")
    for member in members:
        path = Path(member.filename)
        if path.is_absolute() or ".." in path.parts or (member.is_dir() and not member.filename):
            raise PluginError("Add-on archive contains an unsafe path.")
    return members


def read_pack_manifest(pack: Path) -> ImporterManifest:
    try:
        with zipfile.ZipFile(pack) as archive:
            _safe_members(archive)
            names = {member.filename for member in archive.infolist()}
            if "manifest.json" not in names:
                raise PluginError("Add-on archive must contain manifest.json at its root.")
            return ImporterManifest.from_dict(json.loads(archive.read("manifest.json")))
    except zipfile.BadZipFile as exc:
        raise PluginError("The selected add-on is not a valid ZIP archive.") from exc


def install_pack(pack: Path) -> ImporterManifest:
    manifest = read_pack_manifest(pack)
    problems = compatibility_problems(manifest)
    if problems:
        raise PluginError("\n".join(problems))
    destination = plugin_root() / manifest.id / manifest.version
    temporary = destination.with_name(destination.name + ".installing")
    shutil.rmtree(temporary, ignore_errors=True)
    try:
        with zipfile.ZipFile(pack) as archive:
            _safe_members(archive)
            archive.extractall(temporary)
        if not (temporary / "manifest.json").is_file():
            raise PluginError("Add-on extraction did not produce a manifest.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        (destination.parent / "active.json").write_text(json.dumps({"version": manifest.version}, indent=2), encoding="utf-8")
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def installed_manifests() -> list[tuple[ImporterManifest, Path, bool]]:
    found = []
    for plugin_dir in plugin_root().iterdir():
        active_path = plugin_dir / "active.json"
        if not active_path.is_file():
            continue
        try:
            active_version = json.loads(active_path.read_text(encoding="utf-8"))["version"]
            path = plugin_dir / active_version
            manifest = ImporterManifest.from_dict(json.loads((path / "manifest.json").read_text(encoding="utf-8")))
            found.append((manifest, path, not compatibility_problems(manifest)))
        except Exception as exc:
            logger.warning("Ignoring invalid installed add-on in %s: %s", plugin_dir, exc)
    return found


def load_plugin(manifest: ImporterManifest, path: Path) -> ImporterPlugin:
    if compatibility_problems(manifest):
        raise PluginError("This add-on is incompatible with the installed MonStim release.")
    module_name, separator, attribute = manifest.entry_point.partition(":")
    if not separator or not module_name or not attribute:
        raise PluginError("Add-on entry_point must use 'module.py:object' format.")
    module_path = path / module_name
    if not module_path.is_file() or module_path.suffix != ".py":
        raise PluginError("Add-on entry point does not refer to a bundled Python file.")
    spec = importlib.util.spec_from_file_location(f"monstim_plugin_{manifest.id}_{manifest.version}", module_path)
    if spec is None or spec.loader is None:
        raise PluginError("MonStim could not load the add-on module.")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        plugin = getattr(module, attribute)
        return plugin() if isinstance(plugin, type) else plugin
    except Exception as exc:
        raise PluginError(f"Add-on failed to load: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_pack(url: str, expected_sha256: str) -> Path:
    """Download an official asset to a temporary ZIP after checksum validation."""
    with urllib.request.urlopen(url, timeout=15) as response, tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as stream:
        shutil.copyfileobj(response, stream)
        downloaded = Path(stream.name)
    if sha256(downloaded).lower() != expected_sha256.lower():
        downloaded.unlink(missing_ok=True)
        raise PluginError("Downloaded add-on checksum did not match the official catalog.")
    return downloaded


def verify_catalog(payload: bytes, signature_b64: str) -> dict[str, Any]:
    """Verify an Ed25519-signed official catalog before automatic discovery."""
    if not CATALOG_PUBLIC_KEY_B64:
        raise PluginError("Automatic add-on discovery is not configured in this MonStim release.")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        key = Ed25519PublicKey.from_public_bytes(base64.b64decode(CATALOG_PUBLIC_KEY_B64))
        key.verify(base64.b64decode(signature_b64), payload)
        return json.loads(payload)
    except PluginError:
        raise
    except Exception as exc:
        raise PluginError("Official add-on catalog signature could not be verified.") from exc


def fetch_official_catalog(url: str = OFFICIAL_CATALOG_URL) -> dict[str, Any]:
    """Fetch and verify the signed official catalog without accepting unsigned data.

    The published JSON wrapper contains base64 ``payload`` and ``signature``
    fields.  Keeping the signature outside the payload permits deterministic
    catalog generation while the application verifies the exact bytes.
    """
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            wrapper = json.loads(response.read())
        payload = base64.b64decode(wrapper["payload"])
        catalog = verify_catalog(payload, wrapper["signature"])
        if not isinstance(catalog.get("plugins", []), list):
            raise PluginError("Official add-on catalog has no plugin list.")
        return catalog
    except PluginError:
        raise
    except Exception as exc:
        raise PluginError("Could not check official add-ons. You may be offline, or the catalog is unavailable.") from exc


def catalog_check_due(last_checked: str | None, *, now: datetime | None = None) -> bool:
    """Return whether the daily background catalog check is due.

    A malformed or absent timestamp deliberately results in a check. Network
    failures are contained by the worker and do not affect normal startup.
    """
    if not last_checked:
        return True
    try:
        checked_at = datetime.fromisoformat(last_checked)
        if checked_at.tzinfo is None:
            checked_at = checked_at.replace(tzinfo=UTC)
    except ValueError:
        return True
    return (now or datetime.now(UTC)) - checked_at.astimezone(UTC) >= CATALOG_CHECK_INTERVAL


class OfficialCatalogCheckThread(QThread):
    """Fetch a signed catalog without delaying or destabilizing application startup."""

    catalog_available = Signal(dict)
    catalog_unavailable = Signal(str)

    def run(self) -> None:
        try:
            self.catalog_available.emit(fetch_official_catalog())
        except PluginError as exc:
            self.catalog_unavailable.emit(str(exc))


class OfficialUpdateThread(QThread):
    """Download, revalidate, and atomically install user-approved automatic updates."""

    update_installed = Signal(str)
    update_failed = Signal(str)

    def __init__(self, updates: list[dict[str, Any]], parent=None):
        super().__init__(parent)
        self.updates = updates

    def run(self) -> None:
        for item in self.updates:
            downloaded = None
            try:
                downloaded = download_pack(str(item["asset_url"]), str(item["sha256"]))
                manifest = install_pack(downloaded)
                self.update_installed.emit(f"Updated {manifest.name} to {manifest.version}.")
            except (PluginError, OSError, KeyError) as exc:
                self.update_failed.emit(f"Could not install official add-on {item.get('name', item.get('id', 'update'))}: {exc}")
            finally:
                if downloaded is not None:
                    downloaded.unlink(missing_ok=True)


def diagnostic_report() -> str:
    records = []
    for manifest, path, compatible in installed_manifests():
        records.append({"manifest": asdict(manifest), "path": str(path), "compatible": compatible})
    return json.dumps(
        {"app_version": VERSION, "plugin_api_version": PLUGIN_API_VERSION, "plugin_root": str(plugin_root()), "plugins": records}, indent=2
    )
