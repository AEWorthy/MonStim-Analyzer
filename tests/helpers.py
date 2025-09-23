from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

from monstim_signals.core import DatasetAnnot, RecordingAnnot, SessionAnnot


# ---------------------------------------------------------
# Golden fixtures discovery helpers (tests/fixtures/golden)
# ---------------------------------------------------------
def get_golden_root() -> Path:
    """Return the path to the golden fixtures directory used in tests.

    Prefers tests/fixtures/golden; if not found, tries tests/golden.
    Raises AssertionError if neither exists.
    """
    base = Path(__file__).resolve().parent
    candidates = [base / "fixtures" / "golden", base / "golden"]
    for p in candidates:
        if p.exists():
            return p
    raise AssertionError("Missing golden dataset in tests/fixtures/golden or tests/golden")


def _write_bytes(path: Path, seed: str, size: int = 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(seed.encode("utf-8")).digest()
    # repeat digest to desired size
    data = (h * (size // len(h) + 1))[:size]
    path.write_bytes(data)


def make_experiment_and_dataset_with_nested_content(
    root: Path,
    exp_name: str,
    ds_name: str,
    sessions: List[str] | None = None,
) -> Tuple[Path, Path, Dict]:
    """Create an experiment/dataset with nested files for integrity tests.

    Returns: (experiment_path, dataset_path, dataset_metadata_dict)
    """
    sessions = sessions or ["S01", "S02", "S03"]
    exp_path = root / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)

    # Minimal, but present, experiment annot
    (exp_path / "experiment.annot.json").write_text(json.dumps({"data_version": "1.0", "excluded_datasets": []}, indent=2))

    ds_path = exp_path / ds_name
    ds_path.mkdir(parents=True, exist_ok=True)

    # Dataset metadata with specific fields to verify preservation
    ds_meta = {
        "is_completed": True,
        "data_version": "2025.09-test",
        "animal_id": "A1",
        "date": "250101",
        "condition": "cond-X",
        "excluded_sessions": [sessions[1]] if len(sessions) > 1 else [],
    }
    (ds_path / "dataset.annot.json").write_text(json.dumps(ds_meta, indent=2))

    # Create nested session content
    for s in sessions:
        sdir = ds_path / s
        sdir.mkdir(parents=True, exist_ok=True)
        # session annot
        (sdir / "session.annot.json").write_text(json.dumps({"notes": f"session {s}"}, indent=2))
        # raw data placeholder(s)
        _write_bytes(sdir / f"{s}.raw.h5", seed=f"raw-{s}", size=2048)
        # nested auxiliary dirs/files
        _write_bytes(sdir / "aux" / "log.txt", seed=f"log-{s}", size=256)
        _write_bytes(sdir / "deep" / "n1" / "n2" / "blob.bin", seed=f"blob-{s}", size=1536)

    return exp_path, ds_path, ds_meta


def list_files_with_hashes(root: Path) -> Dict[str, Tuple[int, str]]:
    """Return mapping of relative file path -> (size, md5) for all files under root."""
    result: Dict[str, Tuple[int, str]] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            rel = str(fpath.relative_to(root)).replace("\\", "/")
            b = fpath.read_bytes()
            md5 = hashlib.md5(b).hexdigest()
            result[rel] = (len(b), md5)
    return result


# ---------------------------------------------------------------------------
# Minimal on-disk fixtures for repositories (when real golden data is missing)
# ---------------------------------------------------------------------------
def create_minimal_stim_cluster_dict(v: float) -> dict:
    return {
        "stim_delay": 2.0,
        "stim_duration": 1.0,
        "stim_type": "Electrical",
        "stim_v": float(v),
        "stim_min_v": float(v),
        "stim_max_v": float(v),
        "pulse_shape": "Square",
        "num_pulses": 1,
        "pulse_period": 1.0,
        "peak_duration": 0.1,
        "ramp_duration": 0.0,
    }


def create_minimal_session_folder(
    root: Path,
    session_name: str = "RX02",
    num_recordings: int = 3,
    num_channels: int = 2,
    num_samples: int = 1000,
    scan_rate: int = 10_000,
) -> Path:
    """Create a minimal session folder with HDF5 raw, meta, and annot files.

    Returns the session folder path.
    """
    session_dir = root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    # Session annot
    sess_annot = SessionAnnot.create_empty(num_channels)
    # Ensure default channel names are unique and predictable (Ch0, Ch1, ...)
    for i, ch in enumerate(sess_annot.channels):
        ch.name = f"Ch{i}"
        ch.unit = "V"
    (session_dir / "session.annot.json").write_text(json.dumps(asdict(sess_annot), indent=2))

    # Create recordings
    for i in range(num_recordings):
        stem = session_dir / f"WT00-{i:04d}"
        raw_path = stem.with_suffix(".raw.h5")
        meta_path = stem.with_suffix(".meta.json")
        annot_path = stem.with_suffix(".annot.json")

        # Raw HDF5 with a simple signal
        rng = np.random.default_rng(seed=42 + i)
        data = rng.normal(0, 1, size=(num_samples, num_channels)).astype(np.float32)
        with h5py.File(raw_path, "w") as h5:
            h5.create_dataset("raw", data=data, compression="gzip")

        stim = create_minimal_stim_cluster_dict(0.5 + i * 0.5)
        meta = {
            "recording_id": f"WT00-{i:04d}",
            "num_channels": num_channels,
            "scan_rate": scan_rate,
            "pre_stim_acquired": 20,
            "post_stim_acquired": 20,
            "recording_interval": 1.0,
            "channel_types": ["EMG"] * num_channels,
            "emg_amp_gains": [1000] * num_channels,
            "stim_clusters": [stim],
            "primary_stim": 1,  # 1-based index for first cluster
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        rec_annot = RecordingAnnot.create_empty()
        annot_path.write_text(json.dumps(asdict(rec_annot), indent=2))

    return session_dir


def create_minimal_dataset_folder(
    root: Path,
    dataset_name: str = "250916 C554.1 post-dec vibes1",
    session_name: str = "RX02",
    **session_kwargs,
) -> Path:
    """Create a minimal dataset folder containing one session.

    Returns the dataset folder path.
    """
    ds_dir = root / dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)

    ds_annot = DatasetAnnot.create_empty()
    (ds_dir / "dataset.annot.json").write_text(json.dumps(asdict(ds_annot), indent=2))

    create_minimal_session_folder(ds_dir, session_name=session_name, **session_kwargs)
    return ds_dir
