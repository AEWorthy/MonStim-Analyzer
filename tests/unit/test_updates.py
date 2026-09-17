from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime, timedelta

import pytest

import monstim_gui.updates as updates
from monstim_gui.updates import UpdateError, UpdateRelease, activate_update, launch_staged_update, stage_update, staged_executable, update_check_due


def test_update_check_is_due_once_daily():
    now = datetime(2026, 9, 15, tzinfo=UTC)
    assert update_check_due(None, now=now)
    assert not update_check_due((now - timedelta(hours=23)).isoformat(), now=now)
    assert update_check_due((now - timedelta(days=1)).isoformat(), now=now)


def test_activation_only_changes_isolated_update_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MONSTIM_UPDATE_ROOT", str(tmp_path / "updates"))
    version_dir = tmp_path / "updates" / "versions" / "9.9.9"
    version_dir.mkdir(parents=True)
    user_data = tmp_path / "research" / "sentinel.txt"
    settings = tmp_path / "settings" / "MonStim.ini"
    plugin = tmp_path / "plugins" / "official.zip"
    for path in (user_data, settings, plugin):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("do not modify", encoding="utf-8")

    activate_update("9.9.9")

    assert all(path.read_text(encoding="utf-8") == "do not modify" for path in (user_data, settings, plugin))
    assert (tmp_path / "updates" / "current.json").is_file()


def test_stage_rejects_missing_release_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("MONSTIM_UPDATE_ROOT", str(tmp_path / "updates"))
    archive = tmp_path / "invalid.zip"
    archive.write_bytes(b"not a zip")
    release = UpdateRelease("9.9.9", "beta", "windows-x86_64", "https://invalid", "0" * 64, "https://notes")

    with pytest.raises((UpdateError, Exception)):
        stage_update(release, archive)


def test_stage_update_keeps_user_data_outside_version_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MONSTIM_UPDATE_ROOT", str(tmp_path / "updates"))
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("MonStim_Analyzer_v9.9.9-WIN/MonStim Analyzer v9.9.9.exe", b"test executable")
        zip_file.writestr(
            "MonStim_Analyzer_v9.9.9-WIN/_internal/monstim-release.json",
            json.dumps({"version": "9.9.9", "executable": "MonStim Analyzer v9.9.9.exe"}),
        )
    release = UpdateRelease("9.9.9", "beta", "windows-x86_64", "https://invalid", hashlib.sha256(archive.read_bytes()).hexdigest(), "https://notes")
    user_data = tmp_path / "research" / "sentinel.txt"
    user_data.parent.mkdir()
    user_data.write_text("do not modify", encoding="utf-8")

    staged = stage_update(release, archive)

    assert staged_executable("9.9.9").is_file()
    assert staged.name == "9.9.9"
    assert user_data.read_text(encoding="utf-8") == "do not modify"


def test_launch_staged_update_uses_internal_helper(tmp_path, monkeypatch):
    executable = tmp_path / "MonStim Analyzer.exe"
    helper = tmp_path / "_internal" / "MonStim Updater.exe"
    helper.parent.mkdir()
    helper.write_bytes(b"updater")
    executable.write_bytes(b"application")
    launched: list[list[str]] = []

    monkeypatch.setattr(updates.sys, "frozen", True, raising=False)
    monkeypatch.setattr(updates.sys, "executable", str(executable))
    monkeypatch.setattr(updates.subprocess, "Popen", lambda args, **_kwargs: launched.append(args))

    launch_staged_update("9.9.9", wait_pid=123)

    assert launched == [[str(helper), "--version", "9.9.9", "--wait-pid", "123", "--restart"]]
