"""Test index metadata extraction and staleness detection for meta.json changes."""

import json
import time
from pathlib import Path

import pytest  # noqa: F401

from monstim_signals.io.experiment_index import (
    build_experiment_index,
    is_index_stale,
    load_experiment_index,
    save_experiment_index,
)


def _create_minimal_meta_json(meta_path: Path, recording_id: str, stim_v: float):
    """Create a minimal valid meta.json file for testing."""
    meta_data = {
        "recording_id": recording_id,
        "num_channels": 3,
        "scan_rate": 10000,
        "pre_stim_acquired": 50,
        "post_stim_acquired": 200,
        "recording_interval": 5.0,
        "channel_types": ["EMG", "EMG", "Force"],
        "emg_amp_gains": [1000, 1000, 1],
        "stim_clusters": [
            {
                "stim_delay": 0.0,
                "stim_duration": 10.0,
                "stim_type": "Electrical",
                "stim_v": stim_v,
                "stim_min_v": 0.0,
                "stim_max_v": 50.0,
                "pulse_shape": "Square",
                "num_pulses": 1,
                "pulse_period": 10.0,
                "peak_duration": 0.5,
                "ramp_duration": 0.0,
            }
        ],
        "primary_stim": 1,  # 1-based index
        "num_samples": 2500,
        "data_version": "0.0.1",
    }
    meta_path.write_text(json.dumps(meta_data, indent=2))


def test_index_includes_recording_metadata(tmp_path):
    """Verify that build_experiment_index extracts and stores recording metadata."""
    exp_path = tmp_path / "Exp1"
    exp_path.mkdir()

    # Create a minimal session with one recording
    ds_path = exp_path / "DS1"
    ds_path.mkdir()
    sess_path = ds_path / "S1"
    sess_path.mkdir()

    # Create minimal annotation files
    (exp_path / "experiment.annot.json").write_text("{}")
    (ds_path / "dataset.annot.json").write_text("{}")
    (sess_path / "session.annot.json").write_text("{}")

    # Create recording with meta.json
    rec_stem = sess_path / "S1_0000"
    (rec_stem.with_suffix(".raw.h5")).write_bytes(b"h5mock")
    _create_minimal_meta_json(rec_stem.with_suffix(".meta.json"), "S1_0000", 10.0)

    # Build index
    idx = build_experiment_index("Exp1", exp_path)

    # Verify index structure
    assert len(idx.datasets) == 1
    assert len(idx.datasets[0].sessions) == 1
    assert len(idx.datasets[0].sessions[0].recordings) == 1

    # Verify metadata was extracted
    recording_idx = idx.datasets[0].sessions[0].recordings[0]
    assert recording_idx.recording_id == "S1_0000"
    assert recording_idx.num_channels == 3
    assert recording_idx.num_samples == 2500
    assert recording_idx.scan_rate == 10000
    assert recording_idx.primary_stim_v == 10.0  # Should match the stim_v we set

    # Verify meta.json tracking for staleness detection
    assert recording_idx.meta_path is not None
    assert recording_idx.meta_size is not None
    assert recording_idx.meta_mtime is not None


def test_index_stale_on_meta_json_change(tmp_path):
    """Verify that modifying meta.json triggers index staleness detection."""
    exp_path = tmp_path / "Exp2"
    exp_path.mkdir()

    # Create session with recording
    ds_path = exp_path / "DS1"
    ds_path.mkdir()
    sess_path = ds_path / "S1"
    sess_path.mkdir()

    # Create minimal annotation files
    (exp_path / "experiment.annot.json").write_text("{}")
    (ds_path / "dataset.annot.json").write_text("{}")
    (sess_path / "session.annot.json").write_text("{}")

    # Create recording with meta.json
    rec_stem = sess_path / "S1_0000"
    (rec_stem.with_suffix(".raw.h5")).write_bytes(b"h5mock")
    meta_file = rec_stem.with_suffix(".meta.json")
    _create_minimal_meta_json(meta_file, "S1_0000", 15.0)

    # Build and save index
    idx = build_experiment_index("Exp2", exp_path)
    save_experiment_index(idx)

    # Verify index is fresh
    loaded_idx = load_experiment_index(exp_path)
    assert loaded_idx is not None
    assert not is_index_stale(loaded_idx)

    # Wait a moment to ensure mtime changes
    time.sleep(0.1)

    # Modify meta.json file (change content to trigger size/mtime change)
    assert meta_file.exists()

    # Read existing meta, modify it, and write back
    meta_content = json.loads(meta_file.read_text())
    meta_content["recording_id"] = "MODIFIED_ID"
    meta_file.write_text(json.dumps(meta_content, indent=2))

    # Verify index is now stale due to meta.json change
    loaded_idx_after = load_experiment_index(exp_path)
    assert loaded_idx_after is not None
    assert is_index_stale(loaded_idx_after), "Index should be stale after meta.json modification"


def test_index_stale_on_meta_json_deletion(tmp_path):
    """Verify that deleting meta.json triggers index staleness detection."""
    exp_path = tmp_path / "Exp3"
    exp_path.mkdir()

    # Create session with recording
    ds_path = exp_path / "DS1"
    ds_path.mkdir()
    sess_path = ds_path / "S1"
    sess_path.mkdir()

    # Create minimal annotation files
    (exp_path / "experiment.annot.json").write_text("{}")
    (ds_path / "dataset.annot.json").write_text("{}")
    (sess_path / "session.annot.json").write_text("{}")

    # Create recording with meta.json
    rec_stem = sess_path / "S1_0000"
    (rec_stem.with_suffix(".raw.h5")).write_bytes(b"h5mock")
    meta_file = rec_stem.with_suffix(".meta.json")
    _create_minimal_meta_json(meta_file, "S1_0000", 20.0)

    # Build and save index
    idx = build_experiment_index("Exp3", exp_path)
    save_experiment_index(idx)

    # Verify index is fresh
    loaded_idx = load_experiment_index(exp_path)
    assert loaded_idx is not None
    assert not is_index_stale(loaded_idx)

    # Delete meta.json file
    assert meta_file.exists()
    meta_file.unlink()

    # Verify index is now stale due to meta.json deletion
    loaded_idx_after = load_experiment_index(exp_path)
    assert loaded_idx_after is not None
    assert is_index_stale(loaded_idx_after), "Index should be stale after meta.json deletion"


def test_session_load_uses_index_sorting(tmp_path):
    """Verify that index extracts metadata for pre-sorting recordings."""
    exp_path = tmp_path / "Exp4"
    exp_path.mkdir()

    # Create session with multiple recordings at different stim voltages
    ds_path = exp_path / "DS1"
    ds_path.mkdir()
    sess_path = ds_path / "S1"
    sess_path.mkdir()

    # Create minimal annotation files
    (exp_path / "experiment.annot.json").write_text("{}")
    (ds_path / "dataset.annot.json").write_text("{}")
    (sess_path / "session.annot.json").write_text("{}")

    # Create recordings with voltages in non-sorted order
    stim_values = [30.0, 10.0, 20.0, 5.0, 15.0]
    for i, stim_v in enumerate(stim_values):
        rec_stem = sess_path / f"S1_{i:04d}"
        (rec_stem.with_suffix(".raw.h5")).write_bytes(b"h5mock")
        _create_minimal_meta_json(rec_stem.with_suffix(".meta.json"), f"S1_{i:04d}", stim_v)
        (rec_stem.with_suffix(".annot.json")).write_text("{}")

    # Build and save index
    idx = build_experiment_index("Exp4", exp_path)
    save_experiment_index(idx)

    # Verify index metadata is present and can be used for sorting
    assert len(idx.datasets) == 1
    assert len(idx.datasets[0].sessions) == 1
    recordings = idx.datasets[0].sessions[0].recordings
    assert len(recordings) == len(stim_values)

    # Verify all recordings have primary_stim_v extracted
    for rec_idx in recordings:
        assert rec_idx.primary_stim_v is not None

    # Simulate the pre-sorting logic from SessionRepository
    sorted_by_index = sorted(recordings, key=lambda r: r.primary_stim_v if r.primary_stim_v is not None else float("inf"))
    sorted_stim_values = [r.primary_stim_v for r in sorted_by_index]

    # Verify the index-based sorting produces correct order
    assert sorted_stim_values == sorted(
        stim_values
    ), f"Index-based sorting incorrect: {sorted_stim_values} != {sorted(stim_values)}"


def test_index_handles_missing_metadata_gracefully(tmp_path):
    """Verify that index building continues when metadata extraction fails."""
    exp_path = tmp_path / "Exp5"
    exp_path.mkdir()

    ds_path = exp_path / "DS1"
    ds_path.mkdir()
    sess_path = ds_path / "S1"
    sess_path.mkdir()

    # Create minimal annotation files
    (exp_path / "experiment.annot.json").write_text("{}")
    (ds_path / "dataset.annot.json").write_text("{}")
    (sess_path / "session.annot.json").write_text("{}")

    # Create recording with corrupt meta.json
    rec_stem = sess_path / "S1_0000"
    (rec_stem.with_suffix(".raw.h5")).write_bytes(b"h5mock")
    meta_file = rec_stem.with_suffix(".meta.json")
    meta_file.write_text("{invalid json content}")

    # Build index - should succeed despite corrupt meta.json
    idx = build_experiment_index("Exp5", exp_path)

    # Verify index was built
    assert len(idx.datasets) == 1
    assert len(idx.datasets[0].sessions) == 1
    assert len(idx.datasets[0].sessions[0].recordings) == 1

    # Verify metadata fields are None due to parsing failure
    recording_idx = idx.datasets[0].sessions[0].recordings[0]
    assert recording_idx.primary_stim_v is None
    assert recording_idx.recording_id is None
    # But file tracking should still work
    assert recording_idx.path is not None
