"""
Domain Annotations and Overlay Behavior

Purpose: Verify annotation overlay operations on Session/Dataset without touching GUI.
Markers: integration (repo I/O), slow (file copy), no PyQt imports in domain.
Notes: Operates on a temporary copy of a real session directory to avoid mutating repo data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monstim_signals.io.repositories import SessionRepository

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def temp_copied_session(tmp_path: Path) -> Path:
    """Import golden CSVs into a temp directory and return a valid session path."""
    from monstim_signals.io.csv_importer import import_experiment

    from .helpers import get_golden_root

    out_expt = tmp_path / "GoldenExp"
    import_experiment(get_golden_root(), out_expt, overwrite=True, max_workers=1)

    # Pick first dataset/session
    ds_dirs = [p for p in out_expt.iterdir() if p.is_dir()]
    assert ds_dirs, "No datasets were imported from golden fixtures"
    sess_dirs = [p for p in ds_dirs[0].iterdir() if p.is_dir()]
    assert sess_dirs, f"No sessions found in imported dataset {ds_dirs[0]}"
    return sess_dirs[0]


class TestSessionAnnotationOverlay:
    def test_exclude_and_restore_recording(self, temp_copied_session: Path):
        session = SessionRepository(temp_copied_session).load()

        # Precondition: at least one recording
        assert session.num_recordings > 0
        all_recs = session.get_all_recordings(include_excluded=True)
        first_id = all_recs[0].id

        # Count before
        before_filtered = len(session.recordings_filtered)

        # Exclude one recording via annot overlay and reset caches
        session.annot.excluded_recordings.append(first_id)
        session.reset_all_caches()
        after_filtered = len(session.recordings_filtered)

        assert after_filtered == max(0, before_filtered - 1)
        assert first_id in session.excluded_recordings

        # Restore and verify back to original count
        session.restore_recording(first_id)
        session.reset_all_caches()
        assert len(session.recordings_filtered) == before_filtered
        assert first_id not in session.excluded_recordings

    def test_channel_rename_persists_in_overlay(self, temp_copied_session: Path):
        session = SessionRepository(temp_copied_session).load()

        orig_names = list(session.channel_names)
        # Map each name to name+'_x'
        mapping = {name: f"{name}_x" for name in orig_names}
        session.rename_channels(mapping)

        # Session should reflect new names
        got = session.channel_names
        assert all(name.endswith("_x") for name in got)

        # Overlay file should exist and include channel names
        annot_path = temp_copied_session / "session.annot.json"
        assert annot_path.exists()
        data = json.loads(annot_path.read_text())
        # Channels may be saved as list of dicts; verify presence of renamed names
        channel_names_from_file = [ch.get("name") for ch in data.get("channels", [])]
        if channel_names_from_file:
            assert channel_names_from_file == got

    def test_add_and_remove_latency_window(self, temp_copied_session: Path):
        session = SessionRepository(temp_copied_session).load()

        n_channels = session.num_channels
        assert n_channels > 0

        # Add a synthetic latency window and verify
        session.add_latency_window(
            name="TestWindow",
            start_times=[5.0] * n_channels,
            durations=[10.0] * n_channels,
            color="orange",
        )
        assert session.get_latency_window("TestWindow") is not None

        # Removing should update overlays & cache
        session.remove_latency_window("TestWindow")
        assert session.get_latency_window("TestWindow") is None
