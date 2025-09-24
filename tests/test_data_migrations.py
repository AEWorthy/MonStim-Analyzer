import json
import logging
from pathlib import Path

from monstim_signals.io.data_migrations import (
    FutureVersionError,
    InvalidVersionStringError,
    migrate_annotation_dict,
    needs_migration,
    scan_annotation_versions,
)
from monstim_signals.io.repositories import ExperimentRepository
from monstim_signals.version import DATA_VERSION


def test_migrate_legacy_session_annotation(tmp_path: Path):
    # Simulate legacy annotation missing is_completed and with old version 1.0.0
    raw = {
        "excluded_sessions": [],  # dataset-level style key (still fine)
        "data_version": "1.0.0",
    }
    assert needs_migration(raw["data_version"]) is True
    report = migrate_annotation_dict(raw)
    assert report.changed is True
    assert report.final_version == DATA_VERSION
    assert raw["data_version"] == DATA_VERSION
    # Migration added default is_completed
    assert raw.get("is_completed") is False


def test_missing_version_assumed_bootstrap(tmp_path: Path):
    raw = {"excluded_sessions": []}
    report = migrate_annotation_dict(raw)
    # Should set version to current
    assert raw["data_version"] == DATA_VERSION
    assert report.final_version == DATA_VERSION


def test_future_version_error():
    raw = {"data_version": "999.0.0"}
    try:
        migrate_annotation_dict(raw, strict_version=True)
    except FutureVersionError:
        return
    assert False, "FutureVersionError not raised"


def test_idempotent_second_run():
    raw = {"data_version": "1.0.0"}
    report1 = migrate_annotation_dict(raw, in_place=True)
    assert report1.changed is True
    # Second run should detect no change
    report2 = migrate_annotation_dict(raw, in_place=True)
    assert report2.changed is False
    assert report2.steps_applied == []


def test_invalid_version_string():
    raw = {"data_version": "2..0"}
    try:
        migrate_annotation_dict(raw)
    except InvalidVersionStringError:
        return
    assert False, "InvalidVersionStringError not raised for malformed version"


def test_lenient_major_minor_parsing(tmp_path: Path):
    raw = {"data_version": "1.0"}  # Should be interpreted as 1.0.0 legacy
    report = migrate_annotation_dict(raw, in_place=True)
    assert report.final_version == DATA_VERSION
    assert raw["data_version"] == DATA_VERSION


def test_non_strict_future_version_allowed(tmp_path: Path):
    # In non-strict mode, future versions should not raise but return unchanged
    raw = {"data_version": "999.0.0"}
    try:
        report = migrate_annotation_dict(raw, strict_version=False, dry_run=True)
    except FutureVersionError:  # pragma: no cover - shouldn't happen
        assert False, "Should not raise in non-strict mode"
    assert report.original_version == "999.0.0"
    assert report.final_version == "999.0.0"


def test_scan_annotation_versions(tmp_path: Path):
    # Create simple experiment structure
    exp_root = tmp_path / "EXP"
    ds_folder = exp_root / "DS1"
    sess_folder = ds_folder / "S1"
    sess_folder.mkdir(parents=True)
    # Recording files
    (sess_folder / "R000.raw.h5").write_bytes(b"dummy")
    # Minimal annot files with legacy version
    (exp_root / "experiment.annot.json").write_text('{"data_version":"1.0.0"}')
    (ds_folder / "dataset.annot.json").write_text('{"data_version":"1.0"}')  # lenient form
    (sess_folder / "session.annot.json").write_text('{"data_version":"1.0.0"}')
    (sess_folder / "R000.annot.json").write_text('{"data_version":"1.0.0"}')

    results = scan_annotation_versions(exp_root)
    # Expect four entries needing migration
    legacy = [r for r in results if r["needs_migration"]]
    assert len(legacy) == 4
    # Each should have at least one planned step (since current version > 1.0.0)
    assert all(r["planned_steps"] for r in legacy)


def test_dry_run_no_mutation():
    legacy = {"data_version": "1.0.0", "foo": 1}
    snapshot = dict(legacy)
    report = migrate_annotation_dict(legacy, dry_run=True)
    assert report.dry_run is True
    assert legacy == snapshot  # unchanged
    assert report.changed is True
    # Actually migrate now
    migrate_annotation_dict(legacy)
    assert legacy["data_version"] == DATA_VERSION


def test_experiment_preflight_scan_logs(tmp_path, caplog):
    # Build minimal experiment tree with legacy versions so scan finds something
    exp = tmp_path / "EXP"
    ds = exp / "DS1"
    sess = ds / "S1"
    sess.mkdir(parents=True)

    # Create dummy recording trio (only annot needed for scan completeness)
    # Create minimal valid HDF5 file with a 'raw' dataset (num_samples x channels)
    import h5py
    import numpy as np

    with h5py.File(sess / "R000.raw.h5", "w") as h5:
        h5.create_dataset("raw", data=np.zeros((10, 1), dtype="float32"))
    minimal_meta = {
        "recording_id": "R000",
        "num_channels": 1,
        "scan_rate": 1000,
        "pre_stim_acquired": 10,
        "post_stim_acquired": 10,
        "recording_interval": 1.0,
        "channel_types": ["EMG"],
        "emg_amp_gains": [1000],
        "stim_clusters": [
            {
                "stim_delay": 0.0,
                "stim_duration": 10.0,
                "stim_type": "Electrical",
                "stim_v": 0.5,
                "stim_min_v": 0.0,
                "stim_max_v": 1.0,
                "pulse_shape": "Square",
                "num_pulses": 1,
                "pulse_period": 10.0,
                "peak_duration": 1.0,
                "ramp_duration": 0.0,
            }
        ],
        "primary_stim": 1,
        "data_version": "0.0.0",
    }
    (sess / "R000.meta.json").write_text(json.dumps(minimal_meta))
    (sess / "R000.annot.json").write_text(json.dumps({"data_version": "1.0.0"}))

    # Legacy annotation files missing / outdated
    (exp / "experiment.annot.json").write_text(json.dumps({"data_version": "1.0.0"}))
    (ds / "dataset.annot.json").write_text(json.dumps({"data_version": "1.0"}))  # lenient form
    (sess / "session.annot.json").write_text(json.dumps({"data_version": "1.0.0"}))

    caplog.set_level(logging.INFO)

    # Perform explicit scan (now handled at GUI layer in production)
    scan_results = scan_annotation_versions(exp)
    legacy = [r for r in scan_results if r["needs_migration"]]
    assert len(legacy) >= 1

    repo = ExperimentRepository(exp)
    repo.load(preflight_scan=False)  # repository no longer runs scan by default

    # After load, experiment annot should now be current version
    updated = json.loads((exp / "experiment.annot.json").read_text())
    assert updated["data_version"] == DATA_VERSION
