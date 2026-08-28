"""Storage and validation for built-in and user analysis-profile libraries."""

from __future__ import annotations

import glob
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PySide6.QtCore import QStandardPaths

from monstim_gui.io.config_repository import ConfigRepository
from monstim_signals.core import get_docs_path
from monstim_signals.core.configuration import GLOBAL_ONLY_PROFILE_KEYS

logger = logging.getLogger(__name__)

# These filenames are part of the distributed profile library. Older versions
# stored user-created profiles beside them, so unrecognised legacy files can be
# moved into the writable library without copying the shipped examples.
BUILTIN_PROFILE_FILENAMES = frozenset(
    {
        "classic_emg.yml",
        "emg_force_stretch.yml",
        "emg_force_vibration.yml",
        "optical-long.yml",
        "optical-short.yml",
        "pre-stimulus_view.yml",
    }
)


def get_bundled_profile_dir() -> str:
    """Return the shipped, read-only analysis-profile directory."""
    return os.path.join(get_docs_path(), "resources", "analysis_profiles")


def get_user_profile_dir() -> str:
    """Return the per-user library, kept outside installed application files."""
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    if not base:
        base = os.getenv("APPDATA", os.getcwd())
    return os.path.join(base, "analysis_profiles")


@dataclass(frozen=True)
class ProfileRecord:
    name: str
    path: str
    data: dict[str, Any]
    source: str
    read_only: bool


class ProfileManager:
    """Manage a read-only built-in library and a writable user library.

    Passing ``profile_dir`` retains the historic single-library behavior for
    callers and tests that explicitly own a profile directory.
    """

    def __init__(self, profile_dir=None, reference_config=None, *, builtin_dir=None, user_dir=None):
        self.reference_config = reference_config
        self._legacy_single_library = profile_dir is not None and builtin_dir is None and user_dir is None
        if self._legacy_single_library:
            self.profile_dir = str(profile_dir)
            self.builtin_dir = None
            self.user_dir = self.profile_dir
        else:
            self.builtin_dir = str(builtin_dir or get_bundled_profile_dir())
            self.user_dir = str(user_dir or get_user_profile_dir())
            self.profile_dir = self.user_dir

    def _load_path(self, path: str) -> dict[str, Any]:
        with open(path, encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Profile '{path}' must contain a YAML mapping.")
        if self.reference_config:
            data = ConfigRepository.coerce_types(data, self.reference_config)
        return data

    def list_profile_records(self) -> list[ProfileRecord]:
        records: list[ProfileRecord] = []
        libraries = []
        if self.builtin_dir:
            libraries.append((self.builtin_dir, "Built-in", True))
        libraries.append((self.user_dir, "User", False))
        for directory, source, read_only in libraries:
            for path in sorted(glob.glob(os.path.join(directory, "*.yml"))):
                data = self._load_path(path)
                records.append(ProfileRecord(str(data.get("name", Path(path).stem)), path, data, source, read_only))
        return records

    def migrate_legacy_profiles(self) -> list[str]:
        """Copy older user profiles out of the formerly writable bundled folder.

        This is deliberately invoked during Apply, rather than while opening
        Settings, so browsing preferences never creates user files.
        """
        if not self.builtin_dir:
            return []
        migrated = []
        for source in glob.glob(os.path.join(self.builtin_dir, "*.yml")):
            if Path(source).name.casefold() in BUILTIN_PROFILE_FILENAMES:
                continue
            data = self._load_path(source)
            destination = self._user_path(str(data.get("name", Path(source).stem)))
            if not os.path.exists(destination):
                os.makedirs(self.user_dir, exist_ok=True)
                shutil.copy2(source, destination)
                migrated.append(destination)
        return migrated

    def list_profiles(self) -> list[tuple[str, str, dict]]:
        """Historic tuple API used by the main profile selector."""
        return [(record.name, record.path, record.data) for record in self.list_profile_records()]

    def load_profile(self, filename):
        return self._load_path(filename)

    @staticmethod
    def _filename_for(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_") or "profile"
        return f"{slug}.yml"

    def _user_path(self, name: str) -> str:
        return os.path.join(self.user_dir, self._filename_for(name))

    def is_read_only(self, filename: str) -> bool:
        if not self.builtin_dir:
            return False
        try:
            return Path(filename).resolve().is_relative_to(Path(self.builtin_dir).resolve())
        except OSError, ValueError:
            return False

    def validate_profile(self, data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("A profile must be a YAML mapping.")
        name = str(data.get("name", "")).strip()
        if not name:
            raise ValueError("A profile needs a name.")
        # ``stimuli_to_plot`` never affected the bundled plotting path; drop
        # the obsolete profile key when a profile is next saved or imported.
        data.pop("stimuli_to_plot", None)
        parameters = data.get("analysis_parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("analysis_parameters must be a mapping.")
        forbidden = sorted(GLOBAL_ONLY_PROFILE_KEYS & parameters.keys())
        if forbidden:
            raise ValueError(f"Global-only analysis profile keys: {', '.join(forbidden)}")
        return data

    def save_profile(self, data, filename=None):
        data = self.validate_profile(dict(data))
        if filename and self.is_read_only(filename):
            raise ValueError("Built-in profiles are read-only. Duplicate one to edit it.")
        filename = filename or self._user_path(data["name"])
        if not self._legacy_single_library and not Path(filename).resolve().is_relative_to(Path(self.user_dir).resolve()):
            filename = self._user_path(data["name"])
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        ordered = {}
        for key in ("name", "description", "latency_window_preset", "analysis_parameters"):
            if key in data:
                ordered[key] = data[key]
        for key, value in data.items():
            if key not in ordered:
                ordered[key] = value
        with open(filename, "w", encoding="utf-8") as fp:
            yaml.safe_dump(ordered, fp, sort_keys=False)
        return filename

    def duplicate_profile(self, filename: str, name: str) -> str:
        data = self.load_profile(filename)
        data["name"] = name
        return self.save_profile(data)

    def delete_profile(self, filename):
        if self.is_read_only(filename):
            raise ValueError("Built-in profiles cannot be deleted.")
        if os.path.exists(filename):
            os.remove(filename)

    def export_profile(self, filename: str, destination: str) -> None:
        data = self.validate_profile(self.load_profile(filename))
        with open(destination, "w", encoding="utf-8") as fp:
            yaml.safe_dump(data, fp, sort_keys=False)

    def import_profile(self, source: str, *, conflict: str = "error") -> str:
        data = self.validate_profile(self._load_path(source))
        destination = self._user_path(data["name"])
        if os.path.exists(destination):
            if conflict == "replace":
                pass
            elif conflict == "keep_both":
                base = data["name"]
                index = 2
                while os.path.exists(destination):
                    data["name"] = f"{base} {index}"
                    destination = self._user_path(data["name"])
                    index += 1
            else:
                raise FileExistsError(f"A user profile named '{data['name']}' already exists.")
        return self.save_profile(data, destination)
