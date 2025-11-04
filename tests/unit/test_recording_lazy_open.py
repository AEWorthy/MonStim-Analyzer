import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from monstim_signals.core import RecordingAnnot
from monstim_signals.io.repositories import RecordingRepository

pytestmark = pytest.mark.unit


def _write_recording_files(stem: Path, data: np.ndarray) -> None:
    # 1) raw.h5
    with h5py.File(stem.with_suffix(".raw.h5"), "w") as f:
        f.create_dataset("raw", data=data)

    # 2) meta.json
    stim = {
        "stim_delay": 2.0,
        "stim_duration": 1.0,
        "stim_type": "Electrical",
        "stim_v": 1.0,
        "stim_min_v": 0.0,
        "stim_max_v": 5.0,
        "pulse_shape": "Square",
        "num_pulses": 1,
        "pulse_period": 1.0,
        "peak_duration": 0.1,
        "ramp_duration": 0.0,
    }
    meta = {
        "recording_id": stem.name,
        "num_channels": int(data.shape[1]),
        "scan_rate": 1000,
        "pre_stim_acquired": 20,
        "post_stim_acquired": 20,
        "recording_interval": 1.0,
        "channel_types": ["EMG"] * int(data.shape[1]),
        "emg_amp_gains": [1000] * int(data.shape[1]),
        "stim_clusters": [stim],
        "primary_stim": 1,  # 1-based index per data model allowance
        "num_samples": int(data.shape[0]),
    }
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    # 3) annot.json
    annot = RecordingAnnot.create_empty()
    stem.with_suffix(".annot.json").write_text(json.dumps({"cache": annot.cache, "data_version": annot.data_version}))


def test_lazy_reopen_after_close(tmp_path: Path):
    # Arrange: create small 2-channel dataset and associated files
    n, c = 100, 2
    data = np.arange(n * c, dtype=np.float32).reshape(n, c)
    stem = tmp_path / "AA00_0000"
    _write_recording_files(stem, data)

    # Load via repository (provides repo.raw_h5 path to Recording)
    repo = RecordingRepository(stem)
    rec = repo.load()

    # Sanity: initial access works and uses lazy slicing
    np.testing.assert_allclose(rec.raw_view(t=slice(0, 5)), data[0:5, :])

    # Act: close underlying handles, then access again (should reopen lazily)
    rec.close()
    assert rec._raw is None

    sl = rec.raw_view(t=slice(10, 15), ch=slice(None))

    # Assert: slice matches original data and metadata updated
    np.testing.assert_allclose(sl, data[10:15, :])
    assert rec.meta.num_samples == n
