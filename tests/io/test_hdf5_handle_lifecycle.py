import json
from dataclasses import asdict
from pathlib import Path

import pytest

import monstim_signals.io.repositories as repo_mod
from monstim_signals.core import DatasetAnnot, ExperimentAnnot
from tests.helpers import create_minimal_dataset_folder, create_minimal_session_folder

pytestmark = pytest.mark.unit


def _assert_renameable(path: Path) -> None:
    moved = path.with_name(f"{path.name}.renamed")
    path.rename(moved)
    moved.rename(path)


def _assert_raw_files_renameable(root: Path) -> None:
    raw_files = sorted(root.rglob("*.raw.h5"))
    assert raw_files, f"No raw HDF5 files found under {root}"
    for raw_file in raw_files:
        _assert_renameable(raw_file)


def test_session_close_releases_eager_raw_h5_handles(tmp_path: Path):
    session_dir = create_minimal_session_folder(tmp_path, num_recordings=2)

    session = repo_mod.SessionRepository(session_dir).load(lazy_open_h5=False)
    try:
        assert session.get_all_recordings(include_excluded=True)
    finally:
        session.close()

    _assert_raw_files_renameable(session_dir)


def test_dataset_close_releases_excluded_session_handles(tmp_path: Path):
    dataset_dir = tmp_path / "250916 C554.1 post-dec vibes1"
    dataset_dir.mkdir()
    create_minimal_session_folder(dataset_dir, session_name="RX01", num_recordings=1)
    create_minimal_session_folder(dataset_dir, session_name="RX02", num_recordings=1)

    annot = DatasetAnnot.create_empty()
    annot.excluded_sessions = ["RX02"]
    (dataset_dir / "dataset.annot.json").write_text(json.dumps(asdict(annot), indent=2))

    dataset = repo_mod.DatasetRepository(dataset_dir).load(lazy_open_h5=False)
    try:
        assert [session.id for session in dataset.sessions] == ["RX01"]
        assert [session.id for session in dataset.get_all_sessions(include_excluded=True)] == ["RX01", "RX02"]
    finally:
        dataset.close()

    _assert_raw_files_renameable(dataset_dir)


def test_experiment_close_releases_excluded_dataset_handles(tmp_path: Path):
    exp_dir = tmp_path / "exp1"
    exp_dir.mkdir()
    create_minimal_dataset_folder(exp_dir, dataset_name="dsA", num_recordings=1)
    create_minimal_dataset_folder(exp_dir, dataset_name="dsB", num_recordings=1)

    annot = ExperimentAnnot.create_empty()
    annot.excluded_datasets = ["dsB"]
    (exp_dir / "experiment.annot.json").write_text(json.dumps(asdict(annot), indent=2))

    experiment = repo_mod.ExperimentRepository(exp_dir).load(lazy_open_h5=False)
    try:
        assert [dataset.id for dataset in experiment.datasets] == ["dsA"]
        assert [dataset.id for dataset in experiment._all_datasets] == ["dsA", "dsB"]
    finally:
        experiment.close()

    _assert_raw_files_renameable(exp_dir)


def test_session_repository_closes_recordings_when_session_constructor_fails(tmp_path: Path, monkeypatch):
    session_dir = create_minimal_session_folder(tmp_path, num_recordings=2)

    def fail_session_constructor(*args, **kwargs):
        raise RuntimeError("session constructor failed")

    monkeypatch.setattr(repo_mod, "Session", fail_session_constructor)

    with pytest.raises(RuntimeError, match="session constructor failed"):
        repo_mod.SessionRepository(session_dir).load(lazy_open_h5=False)

    _assert_raw_files_renameable(session_dir)


def test_dataset_repository_closes_sessions_when_dataset_constructor_fails(tmp_path: Path, monkeypatch):
    dataset_dir = create_minimal_dataset_folder(tmp_path, num_recordings=2)

    def fail_dataset_constructor(*args, **kwargs):
        raise RuntimeError("dataset constructor failed")

    monkeypatch.setattr(repo_mod, "Dataset", fail_dataset_constructor)

    with pytest.raises(RuntimeError, match="dataset constructor failed"):
        repo_mod.DatasetRepository(dataset_dir).load(lazy_open_h5=False)

    _assert_raw_files_renameable(dataset_dir)


def test_experiment_repository_closes_datasets_when_experiment_constructor_fails(tmp_path: Path, monkeypatch):
    exp_dir = tmp_path / "exp1"
    exp_dir.mkdir()
    create_minimal_dataset_folder(exp_dir, dataset_name="dsA", num_recordings=1)
    create_minimal_dataset_folder(exp_dir, dataset_name="dsB", num_recordings=1)

    def fail_experiment_constructor(*args, **kwargs):
        raise RuntimeError("experiment constructor failed")

    monkeypatch.setattr(repo_mod, "Experiment", fail_experiment_constructor)

    with pytest.raises(RuntimeError, match="experiment constructor failed"):
        repo_mod.ExperimentRepository(exp_dir).load(lazy_open_h5=False)

    _assert_raw_files_renameable(exp_dir)
