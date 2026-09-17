from __future__ import annotations

from pathlib import Path


def test_internal_updater_is_a_self_contained_onefile_binary():
    """Do not recreate the nested-_internal DLL layout that breaks the helper."""
    spec = Path("win-main.spec").read_text(encoding="utf-8")
    updater_section = spec.split("updater_exe = EXE", maxsplit=1)[1].split("coll = COLLECT", maxsplit=1)[0]

    assert "updater_a.binaries" in updater_section
    assert "updater_a.datas" in updater_section
    assert "exclude_binaries=True" not in updater_section
    assert "updater_internal_file" in spec
    assert "*updater_exe.dependencies" not in spec
