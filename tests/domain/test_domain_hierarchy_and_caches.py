"""
Domain Hierarchy and Cache Behavior

Purpose: Validate Dataset/Session properties, hierarchy navigation, and cache reset semantics.
Markers: unit (where in-memory), integration (when loading from repo); fast by default.
Notes: Avoid PySide6 in domain; use create_empty annotations and small stubs.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from monstim_signals.core import DatasetAnnot, ExperimentAnnot, LatencyWindow, RecordingAnnot, RecordingMeta, SessionAnnot, StimCluster
from monstim_signals.domain import Dataset, Experiment, Recording, Session

pytestmark = pytest.mark.unit


def make_dummy_session(num_recs: int = 3, num_channels: int = 2) -> Session:
    recs: list[Recording] = []
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
    def test_parent_aggregate_cache_tracks_child_revision_and_returns_copies(self, monkeypatch):
        sess = make_dummy_session()
        sess.annot.latency_windows = [
            LatencyWindow(name="M-wave", start_times=[-11.0, -11.0], durations=[1.0, 1.0], color="red"),
        ]
        sess.invalidate_window_results()
        dataset = Dataset("D01", [sess], DatasetAnnot.create_empty(), config=sess._config)
        calls = 0
        original = sess.get_m_wave_amplitudes

        def counted(method, channel_index):
            nonlocal calls
            calls += 1
            return original(method, channel_index)

        monkeypatch.setattr(sess, "get_m_wave_amplitudes", counted)
        first, _ = dataset.get_avg_m_wave_amplitudes("rms", 0)
        expected = list(first)
        first[0] = -1
        second, _ = dataset.get_avg_m_wave_amplitudes("rms", 0)
        assert second == expected
        assert calls == 1

        sess.invalidate_selection_results()
        dataset.get_avg_m_wave_amplitudes("rms", 0)
        assert calls == 2

    def test_plotter_recreation_is_limited_to_construction_options(self):
        sess = make_dummy_session()
        original = sess.plotter
        sess.set_config(sess._config)
        assert sess.plotter is original
        style = sess._config.to_dict()
        style["time_window"] += 1
        sess.set_config(style)
        assert sess.plotter is original
        construction = sess._config.to_dict()
        construction["plotting"] = {"enable_decimation": False}
        sess.set_config(construction)
        assert sess.plotter is not original

    def test_distribution_results_are_cached_but_returned_as_copies(self):
        sess = make_dummy_session()
        sess.annot.latency_windows = [
            LatencyWindow(name="M-wave", start_times=[-11.0, -11.0], durations=[1.0, 1.0], color="red"),
            LatencyWindow(name="H-reflex", start_times=[-9.0, -9.0], durations=[1.0, 1.0], color="blue"),
        ]
        sess.invalidate_window_results()
        first = sess.get_lw_distribution("rms", 0, bins=4)
        assert first["bin_edges"].size == 5
        label = next(iter(first["values"]))
        expected = first["values"][label].copy()
        first["values"][label][:] = -1
        second = sess.get_lw_distribution("rms", 0, bins=4)
        np.testing.assert_array_equal(second["values"][label], expected)

    def test_concurrent_requests_share_one_recording_calculation(self, monkeypatch):
        sess = make_dummy_session(num_recs=1)
        recording = sess.all_recordings[0]
        started = Event()
        release = Event()
        calls = 0
        original = sess._compute_signal_recording

        def blocked(rec, kind):
            nonlocal calls
            if kind == "filtered":
                calls += 1
                started.set()
                assert release.wait(2)
            return original(rec, kind)

        monkeypatch.setattr(sess, "_compute_signal_recording", blocked)
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(sess._get_signal_recording, recording, "filtered")
            assert started.wait(2)
            second = executor.submit(sess._get_signal_recording, recording, "filtered")
            release.set()
            first_value = first.result()
            second_value = second.result()
        assert calls == 1
        assert first_value is second_value

    def test_root_config_propagation_visits_each_child_once(self, monkeypatch):
        first = make_dummy_session()
        second = make_dummy_session()
        second.id = "S02"
        dataset = Dataset("D01", [first, second], DatasetAnnot.create_empty(), config=first._config)
        experiment = Experiment("E01", [dataset], ExperimentAnnot.create_empty(), config=first._config)
        calls = {first.id: 0, second.id: 0}
        for session in (first, second):
            original = session.set_config

            def counted(config, _session=session, _original=original):
                calls[_session.id] += 1
                return _original(config)

            monkeypatch.setattr(session, "set_config", counted)

        changed = experiment._config.to_dict()
        changed["time_window"] += 1
        experiment.set_config(changed)
        assert calls == {"S01": 1, "S02": 1}
        experiment.set_config(experiment._config)
        assert calls == {"S01": 1, "S02": 1}

    def test_narrow_invalidations_preserve_filtered_recording_cache(self, monkeypatch):
        sess = make_dummy_session()
        calls = dict.fromkeys((recording.id for recording in sess.all_recordings), 0)
        for recording in sess.all_recordings:
            original = recording.raw_view

            def counted(*args, _recording=recording, _original=original, **kwargs):
                calls[_recording.id] += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(recording, "raw_view", counted)

        first = sess.all_recordings_filtered
        assert set(calls.values()) == {1}
        assert sess.all_recordings_filtered is not first
        assert sess.all_recordings_filtered[0] is first[0]
        assert all(not array.flags.writeable for array in first)
        assert sess.all_recordings_raw
        assert set(calls.values()) == {1}

        sess.invalidate_window_results()
        sess.invalidate_selection_results()
        sess.invalidate_analysis_results()
        assert sess.all_recordings_filtered[0] is first[0]
        assert set(calls.values()) == {1}

        sess.set_config(sess._config)
        assert sess.all_recordings_filtered[0] is first[0]
        assert set(calls.values()) == {1}

        plot_config = sess._config.to_dict()
        plot_config["time_window"] += 1
        sess.set_config(plot_config)
        assert sess.all_recordings_filtered[0] is first[0]
        assert set(calls.values()) == {1}

        analysis_config = sess._config.to_dict()
        analysis_config["bin_size"] *= 2
        sess.set_config(analysis_config)
        assert sess.all_recordings_filtered[0] is first[0]
        assert set(calls.values()) == {1}

        signal_config = sess._config.to_dict()
        signal_config["butter_filter_args"]["order"] += 1
        sess.set_config(signal_config)
        assert sess.all_recordings_filtered[0] is not first[0]
        assert set(calls.values()) == {2}

    def test_session_basic_properties_and_caches(self):
        sess = make_dummy_session()

        assert sess.id == "S01"
        assert sess.num_recordings == 3
        assert sess.num_channels == 2

        # Per-recording arrays memoize while list containers remain safe to mutate.
        all_raw1 = sess.all_recordings_raw
        all_raw2 = sess.all_recordings_raw
        assert all_raw1 is not all_raw2
        assert all_raw1[0] is all_raw2[0]
        all_raw1.pop()
        assert len(sess.all_recordings_raw) == 3

        # dynamic properties (recordings_*) should return new lists but same elements (since all_* is cached)
        raw1 = sess.recordings_raw
        raw2 = sess.recordings_raw
        assert raw1 is not raw2  # list object is new
        assert raw1[0] is raw2[0]  # Element is cached

        filt1 = sess.recordings_filtered
        filt2 = sess.recordings_filtered
        assert filt1 is not filt2
        assert filt1[0] is filt2[0]

        # Signal invalidation should replace source-dependent entries.
        sess.invalidate_signal_data()
        # The list and its per-recording arrays should be new.
        assert sess.all_recordings_raw is not all_raw1
        # And elements of new recordings_raw should be different from old ones (recomputed)
        assert sess.recordings_raw[0] is not raw1[0]

    def test_excluding_recording_affects_filtered_only(self):
        sess = make_dummy_session()

        before_filtered = len(sess.recordings_filtered)
        all_ids = [r.id for r in sess.get_all_recordings(include_excluded=True)]
        sess.exclude_recording(all_ids[0])
        sess.invalidate_selection_results()

        after_filtered = len(sess.recordings_filtered)
        assert after_filtered == before_filtered - 1
        # Raw may include all depending on implementation; at least filtered changed
