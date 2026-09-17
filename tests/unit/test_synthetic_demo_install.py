from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

from monstim_gui.managers.data_manager import (
    SYNTHETIC_DEMO_ARCHIVE,
    SYNTHETIC_DEMO_EXPERIMENT_IDS,
    DataManager,
)


def _write_demo_archive(path: Path, *, unsafe_member: bool = False) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for experiment_id in SYNTHETIC_DEMO_EXPERIMENT_IDS:
            archive.writestr(f"{experiment_id}/experiment.annot.json", "{}")
            archive.writestr(f"{experiment_id}/example.txt", "fictional")
        if unsafe_member:
            archive.writestr("../outside.txt", "must not escape")


def _manager(tmp_path: Path, monkeypatch):
    docs_path = tmp_path / "docs"
    archive_path = docs_path / SYNTHETIC_DEMO_ARCHIVE
    archive_path.parent.mkdir(parents=True)
    _write_demo_archive(archive_path)
    gui = SimpleNamespace(output_path=str(tmp_path / "data"), status_bar=SimpleNamespace(showMessage=lambda *_args: None))
    manager = DataManager(gui)
    monkeypatch.setattr("monstim_gui.managers.data_manager.get_docs_path", lambda: str(docs_path))
    return manager, gui, archive_path


def test_install_synthetic_demos_extracts_to_managed_data_and_refreshes_catalogs(tmp_path, monkeypatch):
    manager, gui, _archive_path = _manager(tmp_path, monkeypatch)
    refreshed = []
    monkeypatch.setattr(manager, "refresh_data_views", lambda *paths: refreshed.extend(paths))

    installed = manager.install_synthetic_demo_experiments(show_dialogs=False)

    expected = [Path(gui.output_path) / experiment_id for experiment_id in SYNTHETIC_DEMO_EXPERIMENT_IDS]
    assert installed == expected
    assert refreshed == expected
    assert all((path / "experiment.annot.json").is_file() for path in expected)


def test_install_synthetic_demos_never_overwrites_an_existing_experiment(tmp_path, monkeypatch):
    manager, gui, _archive_path = _manager(tmp_path, monkeypatch)
    existing = Path(gui.output_path) / SYNTHETIC_DEMO_EXPERIMENT_IDS[0]
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("user data", encoding="utf-8")

    assert manager.install_synthetic_demo_experiments(show_dialogs=False) == []
    assert marker.read_text(encoding="utf-8") == "user data"


def test_install_synthetic_demos_rejects_archive_path_traversal(tmp_path, monkeypatch):
    manager, gui, archive_path = _manager(tmp_path, monkeypatch)
    _write_demo_archive(archive_path, unsafe_member=True)

    assert manager.install_synthetic_demo_experiments(show_dialogs=False) == []
    assert not (tmp_path / "outside.txt").exists()
    assert not any((Path(gui.output_path) / experiment_id).exists() for experiment_id in SYNTHETIC_DEMO_EXPERIMENT_IDS)
