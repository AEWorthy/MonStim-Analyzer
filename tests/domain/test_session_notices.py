import numpy as np
import pytest

from monstim_signals.core import LatencyWindow, RecordingAnnot, RecordingMeta, SessionAnnot, StimCluster
from monstim_signals.domain.recording import Recording
from monstim_signals.domain.session import Session

pytestmark = pytest.mark.unit


def _make_recording(recording_id: str, stim_v: float, scan_rate=1000, stim_delay=1.0):
    stim = StimCluster(
        stim_delay=stim_delay,
        stim_duration=1.0,
        stim_type="Electrical",
        stim_v=stim_v,
        stim_min_v=stim_v,
        stim_max_v=stim_v,
        pulse_shape="Square",
        num_pulses=1,
        pulse_period=1.0,
        peak_duration=0.1,
        ramp_duration=0.0,
    )
    meta = RecordingMeta(
        recording_id=recording_id,
        num_channels=1,
        scan_rate=scan_rate,
        pre_stim_acquired=10,
        post_stim_acquired=10,
        recording_interval=1.0,
        channel_types=["EMG"],
        emg_amp_gains=[1000],
        stim_clusters=[stim],
        primary_stim=stim,
        num_samples=200,
    )
    raw = np.zeros((200, 1))
    return Recording(meta=meta, annot=RecordingAnnot.create_empty(), raw=raw)


def test_session_notice_codes():
    # Windows with overlap and out-of-bounds + zero duration
    windows = [
        LatencyWindow(name="M-wave", color="blue", start_times=[5.0], durations=[5.0]),
        LatencyWindow(name="Overlap1", color="green", start_times=[7.0], durations=[5.0]),
        LatencyWindow(name="ZeroDur", color="purple", start_times=[12.0], durations=[0.0]),
        LatencyWindow(name="OutOfBounds", color="orange", start_times=[30.0], durations=[100.0]),
    ]
    recs = [
        _make_recording("r0", 0.5, stim_delay=1.0),
        _make_recording("r1", 1.0, stim_delay=1.0),
    ]
    sess = Session(session_id="S-notices", recordings=recs, annot=SessionAnnot.create_empty())
    # Inject custom windows after construction to avoid default parameter effects
    sess.annot.latency_windows = windows
    sess.update_latency_window_parameters()

    notices = sess.collect_notices()
    codes = {n["code"] for n in notices}

    assert "missing_m_wave_window" not in codes  # we have an M-wave
    assert "zero_or_negative_window" in codes
    assert "window_out_of_bounds" in codes
    assert "excessive_window_overlap" in codes


def test_session_missing_m_wave_and_no_recordings():
    # Session with all recordings excluded to trigger no_active_recordings
    recs = [
        _make_recording("r0", 0.5),
    ]
    sess = Session(session_id="S-missing", recordings=recs, annot=SessionAnnot.create_empty())
    # Exclude the only recording
    sess.annot.excluded_recordings.append(recs[0].id)
    sess.annot.latency_windows = [LatencyWindow(name="Other", color="grey", start_times=[5.0], durations=[5.0])]
    sess.update_latency_window_parameters()
    notices = sess.collect_notices()
    codes = {n["code"] for n in notices}
    assert "missing_m_wave_window" in codes
    assert "no_active_recordings" in codes
