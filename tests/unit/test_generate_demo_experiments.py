from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from monstim_signals.io.repositories import ExperimentRepository
from tools.generate_demo_experiments import _h_reflex_recording, build_demo_archive


def test_synthetic_h_reflex_condition_scale_does_not_change_m_response():
    time_ms = np.arange(400) * 1_000 / 8_000 - 10
    reduced = _h_reflex_recording(time_ms, 4.2, np.random.default_rng(123), h_reflex_scale=0.70)
    enhanced = _h_reflex_recording(time_ms, 4.2, np.random.default_rng(123), h_reflex_scale=1.30)

    m_response = (time_ms >= 4.0) & (time_ms <= 9.5)
    h_response = (time_ms >= 24.0) & (time_ms <= 32.0)
    np.testing.assert_array_equal(reduced[m_response], enhanced[m_response])
    assert np.max(np.abs(enhanced[h_response])) > np.max(np.abs(reduced[h_response]))


def test_synthetic_protocol_demo_archive_contains_loadable_native_experiments(tmp_path: Path):
    archive_path = build_demo_archive(tmp_path / "synthetic-demos.zip")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(tmp_path / "unpacked")

    h_reflex = ExperimentRepository(tmp_path / "unpacked" / "Synthetic H-reflex Recruitment").load(allow_write=False)
    vibration = ExperimentRepository(tmp_path / "unpacked" / "Synthetic 100Hz Vibration").load(allow_write=False)
    stretch = ExperimentRepository(tmp_path / "unpacked" / "Synthetic Stretch Ramp Hold Release").load(allow_write=False)
    try:
        assert len(h_reflex.datasets) == len(vibration.datasets) == len(stretch.datasets) == 3
        assert [dataset.id for dataset in h_reflex.datasets] == [
            "Condition A - Reduced response",
            "Condition B - Reference response",
            "Condition C - Enhanced response",
        ]
        assert [dataset.id for dataset in vibration.datasets] == [
            "Condition A - Reduced response",
            "Condition B - Reference response",
            "Condition C - Enhanced response",
        ]
        assert [dataset.id for dataset in stretch.datasets] == [
            "Condition A - Reduced response",
            "Condition B - Reference response",
            "Condition C - Enhanced response",
        ]
        assert [session.id for session in h_reflex.datasets[0].sessions] == ["Trial 1", "Trial 2", "Trial 3"]
        assert [session.id for session in vibration.datasets[0].sessions] == ["Trial 1", "Trial 2", "Trial 3"]
        assert [session.id for session in stretch.datasets[0].sessions] == ["Trial 1", "Trial 2", "Trial 3"]
        assert vibration.datasets[0].sessions[0].channel_names == ["TA", "LG", "Force", "Length"]
        assert stretch.datasets[0].sessions[0].channel_names == ["TA", "LG", "Force", "Length"]
        assert vibration.datasets[0].sessions[0].recordings[0].meta.primary_stim.stim_type == "Vibration"
        assert stretch.datasets[0].sessions[0].recordings[0].meta.primary_stim.stim_type == "Motor Length"
    finally:
        h_reflex.close()
        vibration.close()
        stretch.close()
