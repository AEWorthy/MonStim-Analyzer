import json
from pathlib import Path

from monstim_signals.core.utils import load_config
from monstim_signals.io.experiment_catalog import (
    CATALOG_FILENAME,
    build_catalog,
    ensure_catalog,
    invalidate_catalogs,
    recording_stem,
    relocate_catalog_paths,
)
from monstim_signals.io.repositories import DatasetRepository, ExperimentRepository, RecordingRepository, SessionRepository
from tests.helpers import create_minimal_dataset_folder, create_minimal_session_folder


def test_catalog_records_exact_raw_h5_stems_and_removes_legacy_index(tmp_path: Path):
    experiment = tmp_path / "Experiment"
    experiment.mkdir()
    session = create_minimal_dataset_folder(experiment, dataset_name="Dataset", num_recordings=2) / "RX02"
    (experiment / ".index.json").write_text("obsolete")

    catalog = build_catalog(experiment)

    assert (experiment / CATALOG_FILENAME).is_file()
    assert not (experiment / ".index.json").exists()
    records = catalog.recordings(session)
    assert [record.stem.name for record in records] == ["WT00-0000", "WT00-0001"]
    assert all(record.raw_path == record.stem.with_suffix(".raw.h5") for record in records)
    assert recording_stem(records[0].raw_path) == records[0].stem


def test_experiment_repository_uses_catalog_without_json_index(tmp_path: Path):
    experiment = tmp_path / "Experiment"
    experiment.mkdir()
    create_minimal_dataset_folder(experiment, dataset_name="Dataset", num_recordings=2)

    first_catalog = ensure_catalog(experiment)
    catalog_mtime = first_catalog.path.stat().st_mtime
    config = load_config()
    config.update({"lazy_open_h5": True, "load_workers": 2})
    loaded = ExperimentRepository(experiment).load(config=config)
    try:
        assert [dataset.id for dataset in loaded.datasets] == ["Dataset"]
        assert loaded.datasets[0].sessions[0].num_all_recordings == 2
        assert first_catalog.path.stat().st_mtime == catalog_mtime
        assert not (experiment / ".index.json").exists()
    finally:
        loaded.close()


def test_saved_recording_annotation_updates_existing_catalog(tmp_path: Path):
    experiment = tmp_path / "Experiment"
    experiment.mkdir()
    session = create_minimal_dataset_folder(experiment, dataset_name="Dataset", num_recordings=1) / "RX02"
    catalog = build_catalog(experiment)
    record = catalog.recordings(session)[0]

    repository = RecordingRepository(record.stem)
    recording = repository.load(lazy_open_h5=True)
    recording.annot.cache["catalog_test"] = True
    repository.save(recording)

    refreshed = ensure_catalog(experiment).recordings(session)[0]
    assert json.loads(refreshed.annot_json)["cache"]["catalog_test"] is True


def test_catalog_relocates_a_renamed_dataset_without_rebuild(tmp_path: Path):
    experiment = tmp_path / "Experiment"
    experiment.mkdir()
    old_dataset = create_minimal_dataset_folder(experiment, dataset_name="Dataset", num_recordings=1)
    catalog = build_catalog(experiment)
    new_dataset = experiment / "RenamedDataset"
    old_dataset.rename(new_dataset)

    assert relocate_catalog_paths(experiment, old_dataset, new_dataset)
    assert catalog.dataset_paths() == [new_dataset]
    assert catalog.session_paths(new_dataset) == [new_dataset / "RX02"]
    assert catalog.recordings(new_dataset / "RX02")[0].stem.parent == new_dataset / "RX02"


def test_catalog_invalidation_removes_cache_and_forces_rebuild(tmp_path: Path):
    experiment = tmp_path / "Experiment"
    experiment.mkdir()
    create_minimal_dataset_folder(experiment, dataset_name="Dataset", num_recordings=1)
    build_catalog(experiment)

    invalidate_catalogs(experiment)

    assert not (experiment / CATALOG_FILENAME).exists()
    rebuilt = ensure_catalog(experiment)
    assert rebuilt.dataset_paths() == [experiment / "Dataset"]


def test_catalog_rebuilds_when_dataset_or_session_directories_change(tmp_path: Path):
    experiment = tmp_path / "Experiment"
    experiment.mkdir()
    dataset = create_minimal_dataset_folder(experiment, dataset_name="Dataset", num_recordings=1)
    build_catalog(experiment)

    (dataset / "RX02").rename(dataset / "RX03")
    rebuilt = ensure_catalog(experiment)

    assert rebuilt.session_paths(dataset) == [dataset / "RX03"]


def test_catalog_rebuilds_when_a_dataset_is_removed_externally(tmp_path: Path):
    experiment = tmp_path / "Experiment"
    experiment.mkdir()
    create_minimal_dataset_folder(experiment, dataset_name="First", num_recordings=1)
    second = create_minimal_dataset_folder(experiment, dataset_name="Second", num_recordings=1)
    build_catalog(experiment)

    import shutil

    shutil.rmtree(second)
    rebuilt = ensure_catalog(experiment)

    assert rebuilt.dataset_paths() == [experiment / "First"]


def test_session_save_does_not_rewrite_recording_annotations(tmp_path: Path, monkeypatch):
    experiment = tmp_path / "Experiment"
    experiment.mkdir()
    session_path = create_minimal_dataset_folder(experiment, dataset_name="Dataset", num_recordings=2) / "RX02"
    ensure_catalog(experiment)
    session = SessionRepository(session_path).load(lazy_open_h5=True)
    calls = []

    def record_save_should_not_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(RecordingRepository, "save", record_save_should_not_run)
    session.annot.excluded_recordings.append(session.all_recordings[0].id)
    session.repo.save(session)

    assert calls == []
    persisted = json.loads((session_path / "session.annot.json").read_text())
    assert persisted["excluded_recordings"] == [session.all_recordings[0].id]


def test_batch_session_save_persists_each_session_and_refreshes_the_catalog(tmp_path: Path):
    experiment = tmp_path / "Experiment"
    experiment.mkdir()
    dataset = create_minimal_dataset_folder(experiment, dataset_name="Dataset", num_recordings=1)
    second_session = create_minimal_session_folder(dataset, session_name="RX03", num_recordings=1)
    catalog = ensure_catalog(experiment)
    sessions = [
        SessionRepository(dataset / "RX02").load(lazy_open_h5=True),
        SessionRepository(second_session).load(lazy_open_h5=True),
    ]
    for session in sessions:
        session.annot.excluded_recordings.append(session.all_recordings[0].id)

    SessionRepository.save_many(sessions)

    for session in sessions:
        persisted = json.loads((session.repo.session_js).read_text())
        with catalog.connect() as connection:
            cached = json.loads(
                connection.execute("SELECT annot_json FROM sessions WHERE path = ?", (str(session.repo.folder.resolve()),)).fetchone()["annot_json"]
            )
        assert persisted["excluded_recordings"] == [session.all_recordings[0].id]
        assert cached["excluded_recordings"] == [session.all_recordings[0].id]


def test_dataset_latency_window_change_persists_every_child_session(tmp_path: Path):
    experiment = tmp_path / "Experiment"
    experiment.mkdir()
    dataset_path = create_minimal_dataset_folder(experiment, dataset_name="Dataset", num_recordings=1)
    create_minimal_session_folder(dataset_path, session_name="RX03", num_recordings=1)
    ensure_catalog(experiment)
    dataset = DatasetRepository(dataset_path).load(lazy_open_h5=True)

    dataset.add_latency_window("Dataset window", [5.0, 5.0], [2.0, 2.0])

    for session_path in (dataset_path / "RX02", dataset_path / "RX03"):
        persisted = json.loads((session_path / "session.annot.json").read_text())
        assert [window["name"] for window in persisted["latency_windows"]] == ["Dataset window"]
