import numpy as np
import pytest

from monstim_signals.core import DatasetAnnot, LatencyWindow, SessionAnnot
from monstim_signals.domain.dataset import Dataset
from monstim_signals.domain.session import Session

pytestmark = pytest.mark.unit


class DummyRepo:
    def save(self, obj):
        pass


def _make_basic_session(session_id: str, windows=None, n_recordings=2):
    if windows is None:
        windows = []
    annot = SessionAnnot.create_empty()
    annot.latency_windows = windows

    # fabricate minimal required properties using a real Session initialization path
    # We'll build minimal Recording objects with synthetic meta
    from monstim_signals.core import RecordingAnnot, RecordingMeta, StimCluster
    from monstim_signals.domain.recording import Recording

    recs = []
    n_samples = 200
    n_channels = 1
    stim = StimCluster(
        stim_delay=1.0,
        stim_duration=1.0,
        stim_type="Electrical",
        stim_v=0.5,
        stim_min_v=0.5,
        stim_max_v=0.5,
        pulse_shape="Square",
        num_pulses=1,
        pulse_period=1.0,
        peak_duration=0.1,
        ramp_duration=0.0,
    )
    for i in range(n_recordings):
        meta = RecordingMeta(
            recording_id=f"r{i}",
            num_channels=n_channels,
            scan_rate=1000,
            pre_stim_acquired=10,
            post_stim_acquired=10,
            recording_interval=1.0,
            channel_types=["EMG"],
            emg_amp_gains=[1000],
            stim_clusters=[stim],
            primary_stim=stim,
            num_samples=n_samples,
        )
        raw = np.zeros((n_samples, n_channels))
        recs.append(Recording(meta=meta, annot=RecordingAnnot.create_empty(), raw=raw))

    sess = Session(session_id=session_id, recordings=recs, annot=annot)
    return sess


def test_dataset_notice_codes():
    # Sessions with differing window sets to trigger heterogeneity and churn
    sess1 = _make_basic_session(
        "S1",
        windows=[
            LatencyWindow(name="M-wave", color="blue", start_times=[5.0], durations=[5.0]),
            LatencyWindow(name="H-reflex", color="red", start_times=[15.0], durations=[5.0]),
            LatencyWindow(name="Late1", color="green", start_times=[30.0], durations=[5.0]),
        ],
    )
    sess2 = _make_basic_session(
        "S2",
        windows=[
            LatencyWindow(name="M-wave", color="blue", start_times=[6.0], durations=[4.0]),
            LatencyWindow(name="AltReflex", color="purple", start_times=[18.0], durations=[4.0]),
            LatencyWindow(name="Late2", color="green", start_times=[32.0], durations=[5.0]),
        ],
    )

    ds = Dataset(
        dataset_id="DS-N1",
        sessions=[sess1, sess2],
        annot=DatasetAnnot.create_empty(),
        repo=DummyRepo(),
        config={
            "bin_size": 0.5,
            "default_method": "average_rectified",
            "m_color": "blue",
            "h_color": "red",
            "title_font_size": 10,
            "axis_label_font_size": 10,
            "tick_font_size": 8,
            "subplot_adjust_args": {},
        },
    )

    notices = ds.collect_notices()
    codes = {n["code"] for n in notices}

    assert "heterogeneous_latency_windows" in codes
    assert "missing_m_wave_window" not in codes  # M-wave present in both sessions
    assert "single_session_only" not in codes  # there are two sessions

    # high churn likely present due to diverse names
    # Do not assert mandatory churn to avoid brittleness; if present it's correct


def test_dataset_notices_single_session_and_missing_m():
    # Single session lacking M-wave
    sess = _make_basic_session(
        "S_only",
        windows=[
            LatencyWindow(name="H-reflex", color="red", start_times=[15.0], durations=[5.0]),
        ],
    )
    ds = Dataset(
        dataset_id="DS-single",
        sessions=[sess],
        annot=DatasetAnnot.create_empty(),
        repo=DummyRepo(),
        config={
            "bin_size": 0.5,
            "default_method": "average_rectified",
            "m_color": "blue",
            "h_color": "red",
            "title_font_size": 10,
            "axis_label_font_size": 10,
            "tick_font_size": 8,
            "subplot_adjust_args": {},
        },
    )
    codes = {n["code"] for n in ds.collect_notices()}
    assert "single_session_only" in codes
    assert "missing_m_wave_window" in codes


def test_dataset_no_active_sessions_notice():
    ds = Dataset(
        dataset_id="DS-empty",
        sessions=[],
        annot=DatasetAnnot.create_empty(),
        repo=DummyRepo(),
        config={
            "bin_size": 0.5,
            "default_method": "average_rectified",
            "m_color": "blue",
            "h_color": "red",
            "title_font_size": 10,
            "axis_label_font_size": 10,
            "tick_font_size": 8,
            "subplot_adjust_args": {},
        },
    )
    codes = {n["code"] for n in ds.collect_notices()}
    assert "no_active_session" in codes
    # Missing M-wave also expected because there are no sessions with that window
    assert "missing_m_wave_window" in codes
