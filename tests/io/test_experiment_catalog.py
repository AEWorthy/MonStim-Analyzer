import json
from pathlib import Path

from monstim_signals.core.utils import load_config
from monstim_signals.io.experiment_catalog import (
    CATALOG_FILENAME,
    build_catalog,
    ensure_catalog,
    recording_stem,
    relocate_catalog_paths,
)
from monstim_signals.io.repositories import ExperimentRepository, RecordingRepository
from tests.helpers import create_minimal_dataset_folder


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
