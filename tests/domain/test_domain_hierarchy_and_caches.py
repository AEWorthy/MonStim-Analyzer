"""
Domain Hierarchy and Cache Behavior

Purpose: Validate Dataset/Session properties, hierarchy navigation, and cache reset semantics.
Markers: unit (where in-memory), integration (when loading from repo); fast by default.
Notes: Avoid PySide6 in domain; use create_empty annotations and small stubs.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from monstim_signals.core import RecordingAnnot, RecordingMeta, SessionAnnot, StimCluster
from monstim_signals.domain import Recording, Session

pytestmark = pytest.mark.unit


def make_dummy_session(num_recs: int = 3, num_channels: int = 2) -> Session:
    recs: List[Recording] = []
    for i in range(num_recs):
        stim = StimCluster(
            stim_delay=2.0,
            stim_duration=1.0,
            stim_type="Electrical",
            stim_v=float(i + 1),
            stim_min_v=float(i + 1),
            stim_max_v=float(i + 1),
            pulse_shape="Square",
            num_pulses=1,
            pulse_period=1.0,
            peak_duration=0.1,
            ramp_duration=0.0,
        )
        meta = RecordingMeta(
            recording_id=f"rec_{i:02d}",
            num_samples=100,
            num_channels=num_channels,
            scan_rate=10000,
            recording_interval=1.0,
            channel_types=["EMG"] * num_channels,
            emg_amp_gains=[1000] * num_channels,
            stim_clusters=[stim],
            primary_stim=stim,
            pre_stim_acquired=10,
            post_stim_acquired=10,
        )
        raw = np.random.randn(100, num_channels)
        recs.append(Recording(meta=meta, annot=RecordingAnnot.create_empty(), raw=raw))

    annot = SessionAnnot.create_empty(num_channels)
    # Construct Session using public initializer
    sess = Session(session_id="S01", recordings=recs, annot=annot)
    return sess


class TestHierarchyAndCaches:
    def test_session_basic_properties_and_caches(self):
        sess = make_dummy_session()

        assert sess.id == "S01"
        assert sess.num_recordings == 3
        assert sess.num_channels == 2

        # Cached properties (all_*) should memoize
        all_raw1 = sess.all_recordings_raw
        all_raw2 = sess.all_recordings_raw
        assert all_raw1 is all_raw2

        # dynamic properties (recordings_*) should return new lists but same elements (since all_* is cached)
        raw1 = sess.recordings_raw
        raw2 = sess.recordings_raw
        assert raw1 is not raw2  # List object is new
        assert raw1[0] is raw2[0]  # Element is cached

        filt1 = sess.recordings_filtered
        filt2 = sess.recordings_filtered
        assert filt1 is not filt2
        assert filt1[0] is filt2[0]

        # Reset caches should invalidate
        sess.reset_all_caches()
        # After reset, all_recordings_raw should be new
        assert sess.all_recordings_raw is not all_raw1
        # And elements of new recordings_raw should be different from old ones (recomputed)
        assert sess.recordings_raw[0] is not raw1[0]

    def test_excluding_recording_affects_filtered_only(self):
        sess = make_dummy_session()

        before_filtered = len(sess.recordings_filtered)
        all_ids = [r.id for r in sess.get_all_recordings(include_excluded=True)]
        sess.exclude_recording(all_ids[0])
        sess.reset_all_caches()

        after_filtered = len(sess.recordings_filtered)
        assert after_filtered == before_filtered - 1
        # Raw may include all depending on implementation; at least filtered changed
