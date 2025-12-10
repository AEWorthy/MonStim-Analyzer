from pathlib import Path

from monstim_signals.io.experiment_index import (
    build_experiment_index,
    is_index_stale,
    load_experiment_index,
    save_experiment_index,
)


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
