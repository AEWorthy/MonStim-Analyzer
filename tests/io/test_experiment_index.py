from pathlib import Path

from monstim_signals.io.experiment_index import (
    build_experiment_index,
    ensure_fresh_index,
    find_session_index,
    is_index_stale,
    load_experiment_index,
    save_experiment_index,
)
from monstim_signals.io.repositories import DatasetRepository


def test_build_and_save_index(tmp_path: Path):
    # Create fake experiment structure
    exp = tmp_path / "ExpA"
    ds = exp / "Dataset1"
    sess = ds / "SessionX"
    exp.mkdir()
    ds.mkdir()
    sess.mkdir()

    # Minimal annot files
    (exp / "experiment.annot.json").write_text("{}")
    (ds / "dataset.annot.json").write_text("{}")
    (sess / "session.annot.json").write_text("{}")

    # Add a recording file
    rec = sess / "RX00_0000.raw.h5"
    rec.write_bytes(b"h5mock")

    # Build index
    idx = build_experiment_index("ExpA", exp)
    save_experiment_index(idx)

    # Load and validate
    loaded = load_experiment_index(exp)
    assert loaded is not None
    assert loaded.id == "ExpA"
    assert len(loaded.datasets) == 1
    assert len(loaded.datasets[0].sessions) == 1
    assert len(loaded.datasets[0].sessions[0].recordings) == 1


def test_stale_detection(tmp_path: Path):
    # Setup experiment
    exp = tmp_path / "ExpB"
    ds = exp / "Dataset1"
    sess = ds / "SessionX"
    exp.mkdir()
    ds.mkdir()
    sess.mkdir()
    (exp / "experiment.annot.json").write_text("{}")
    (ds / "dataset.annot.json").write_text("{}")
    (sess / "session.annot.json").write_text("{}")

    # Build index
    idx = build_experiment_index("ExpB", exp)
    save_experiment_index(idx)

    # Initially not stale
    loaded = load_experiment_index(exp)
    assert loaded is not None
    assert is_index_stale(loaded) is False

    # Modify session annot to change mtime/size
    (sess / "session.annot.json").write_text('{"changed":true}')

    loaded2 = load_experiment_index(exp)
    assert loaded2 is not None
    assert is_index_stale(loaded2) is True


def test_ensure_fresh_index_builds_when_missing(tmp_path: Path):
    exp = tmp_path / "ExpC"
    ds = exp / "DatasetA"
    sess = ds / "S1"
    rec = sess / "R1_0000.raw.h5"
    exp.mkdir()
    ds.mkdir()
    sess.mkdir()
    (exp / "experiment.annot.json").write_text("{}")
    (ds / "dataset.annot.json").write_text("{}")
    (sess / "session.annot.json").write_text("{}")
    rec.write_bytes(b"h5mock")

    # No index present initially
    ensure_fresh_index("ExpC", exp)
    loaded = load_experiment_index(exp)
    assert loaded is not None
    assert loaded.id == "ExpC"
    assert loaded.datasets[0].id == "DatasetA"
    assert loaded.datasets[0].sessions[0].id == "S1"
    first_rec_path = Path(loaded.datasets[0].sessions[0].recordings[0].path)
    assert first_rec_path.name.startswith("R1")


def test_find_session_index_maps_recordings(tmp_path: Path):
    exp = tmp_path / "ExpD"
    ds = exp / "DatasetA"
    sess = ds / "S1"
    exp.mkdir()
    ds.mkdir()
    sess.mkdir()
    (exp / "experiment.annot.json").write_text("{}")
    (ds / "dataset.annot.json").write_text("{}")
    (sess / "session.annot.json").write_text("{}")
    # Multiple recordings
    for rid in ("RX00_0000.raw.h5", "RX00_0001.raw.h5"):
        (sess / rid).write_bytes(b"h5mock")
    idx = build_experiment_index("ExpD", exp)
    save_experiment_index(idx)
    loaded = load_experiment_index(exp)
    assert loaded is not None

    # Resolve session index
    si = find_session_index(exp, sess)
    assert si is not None
    assert si.id == "S1"
    assert len(si.recordings) == 2


def test_ensure_fresh_index_updates_on_dataset_rename(tmp_path: Path):
    exp = tmp_path / "ExpE"
    ds = exp / "DatasetA"
    sess = ds / "S1"
    exp.mkdir()
    ds.mkdir()
    sess.mkdir()
    (exp / "experiment.annot.json").write_text("{}")
    (ds / "dataset.annot.json").write_text("{}")
    (sess / "session.annot.json").write_text("{}")
    (sess / "RX00_0000.raw.h5").write_bytes(b"h5mock")
    ensure_fresh_index("ExpE", exp)
    loaded = load_experiment_index(exp)
    assert loaded is not None
    assert loaded.datasets[0].id == "DatasetA"

    # Rename dataset folder
    new_ds = exp / "DatasetRenamed"
    ds.rename(new_ds)
    # Ensure index refresh picks up rename
    ensure_fresh_index("ExpE", exp)
    loaded2 = load_experiment_index(exp)
    assert loaded2 is not None
    ids = [d.id for d in loaded2.datasets]
    assert "DatasetRenamed" in ids
    assert "DatasetA" not in ids


def test_dataset_repository_rename_refreshes_index(tmp_path: Path):
    exp = tmp_path / "ExpRepo"
    ds = exp / "DatasetA"
    sess = ds / "S1"
    exp.mkdir()
    ds.mkdir()
    sess.mkdir()
    (exp / "experiment.annot.json").write_text("{}")
    (ds / "dataset.annot.json").write_text("{}")
    (sess / "session.annot.json").write_text("{}")
    (sess / "RX00_0000.raw.h5").write_bytes(b"h5mock")

    ensure_fresh_index("ExpRepo", exp)
    repo = DatasetRepository(ds)

    new_ds = exp / "DatasetB"
    repo.rename(new_ds)

    loaded = load_experiment_index(exp)
    assert loaded is not None
    ids = [d.id for d in loaded.datasets]
    assert ids == ["DatasetB"]


def test_stale_detection_after_new_recording_addition(tmp_path: Path):
    exp = tmp_path / "ExpF"
    ds = exp / "DatasetA"
    sess = ds / "S1"
    exp.mkdir()
    ds.mkdir()
    sess.mkdir()
    (exp / "experiment.annot.json").write_text("{}")
    (ds / "dataset.annot.json").write_text("{}")
    (sess / "session.annot.json").write_text("{}")
    (sess / "RX00_0000.raw.h5").write_bytes(b"h5mock")
    idx = build_experiment_index("ExpF", exp)
    save_experiment_index(idx)
    loaded = load_experiment_index(exp)
    assert loaded is not None
    assert is_index_stale(loaded) is False

    # Add new recording file
    (sess / "RX00_0001.raw.h5").write_bytes(b"h5mock")
    loaded2 = load_experiment_index(exp)
    assert loaded2 is not None
    # New recording addition should mark index as stale
    assert is_index_stale(loaded2) is True


def test_index_persists_and_recovers(tmp_path: Path):
    exp = tmp_path / "ExpG"
    ds = exp / "DatasetA"
    sess = ds / "S1"
    exp.mkdir()
    ds.mkdir()
    sess.mkdir()
    for p in (exp / "experiment.annot.json", ds / "dataset.annot.json", sess / "session.annot.json"):
        p.write_text("{}")
    (sess / "RX00_0000.raw.h5").write_bytes(b"h5mock")

    idx = build_experiment_index("ExpG", exp)
    save_experiment_index(idx)

    # Simulate accidental deletion of index file and rebuild
    index_file = exp / ".index.json"
    assert index_file.exists()
    index_file.unlink()
    ensure_fresh_index("ExpG", exp)
    assert index_file.exists()

    loaded = load_experiment_index(exp)
    assert loaded is not None
    assert loaded.id == "ExpG"
