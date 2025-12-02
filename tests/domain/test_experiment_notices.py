import numpy as np
import pytest

from monstim_signals.core import DatasetAnnot, ExperimentAnnot, LatencyWindow, SessionAnnot
from monstim_signals.domain.dataset import Dataset
from monstim_signals.domain.experiment import Experiment
from monstim_signals.domain.session import Session

pytestmark = pytest.mark.unit


class DummyRepo:
    def save(self, obj):
        pass


def _make_session(session_id: str, windows=None, scan_rate=1000):
    if windows is None:
        windows = []
    annot = SessionAnnot.create_empty()
    annot.latency_windows = windows

    from monstim_signals.core import RecordingAnnot, RecordingMeta, StimCluster
    from monstim_signals.domain.recording import Recording

    recs = []
    n_samples = 100
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
    for i in range(2):
        meta = RecordingMeta(
            recording_id=f"r{i}",
            num_channels=n_channels,
            scan_rate=scan_rate,
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

    return Session(session_id=session_id, recordings=recs, annot=annot)


def _make_dataset(ds_id: str, sessions):
    return Dataset(
        dataset_id=ds_id,
        sessions=sessions,
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


def test_experiment_notice_codes():
    # Dataset 1 with M-wave/H-reflex
    sess1 = _make_session(
        "S1",
        windows=[
            LatencyWindow(name="M-wave", color="blue", start_times=[5.0], durations=[5.0]),
            LatencyWindow(name="H-reflex", color="red", start_times=[15.0], durations=[5.0]),
        ],
        scan_rate=1000,
    )
    ds1 = _make_dataset("DS1", [sess1])

    # Dataset 2 missing M-wave, different scan_rate to trigger mixed_scan_rates
    sess2 = _make_session(
        "S2",
        windows=[LatencyWindow(name="H-reflex", color="red", start_times=[16.0], durations=[5.0])],
        scan_rate=1200,
    )
    ds2 = _make_dataset("DS2", [sess2])

    exp = Experiment(
        expt_id="EXP-N1",
        datasets=[ds1, ds2],
        annot=ExperimentAnnot.create_empty(),
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
    codes = {n["code"] for n in exp.collect_notices()}
    assert "heterogeneous_latency_windows" in codes
    assert "mixed_scan_rates" in codes
    # At least one dataset provides an M-wave so experiment should NOT claim it's missing
    assert "missing_m_wave_window" not in codes


def test_experiment_no_active_datasets():
    exp = Experiment(
        expt_id="EXP-empty",
        datasets=[],
        annot=ExperimentAnnot.create_empty(),
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
    codes = {n["code"] for n in exp.collect_notices()}
    assert "no_active_datasets" in codes
    assert "missing_m_wave_window" in codes  # trivially missing
