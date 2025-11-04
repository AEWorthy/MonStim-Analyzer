import numpy as np

from monstim_signals.core import DatasetAnnot, LatencyWindow, SessionAnnot
from monstim_signals.domain.dataset import Dataset
from monstim_signals.domain.session import Session


class DummyRepo:
    def save(self, obj):
        pass


def _make_session(session_id: str, scan_rate=1000, stim_start=0.0, windows=None, n_recordings=3, n_channels=2):
    if windows is None:
        windows = []
    # Minimal channel structure via annot
    annot = SessionAnnot.create_empty()
    annot.latency_windows = windows
    # Fake required session attributes
    session = Session.__new__(Session)  # bypass __init__ (complex real signature)
    session.id = session_id
    session.annot = annot
    session.repo = DummyRepo()
    session.scan_rate = scan_rate
    session.stim_start = stim_start
    session.parent_dataset = None
    session._config = {
        "bin_size": 0.5,
        "default_method": "average_rectified",
        "m_color": "blue",
        "h_color": "red",
        "title_font_size": 10,
        "axis_label_font_size": 10,
        "tick_font_size": 8,
        "subplot_adjust_args": {},
    }
    session.m_max_args = {"validation_tolerance": 1.05}
    # Provide minimal properties used downstream
    session.num_channels = n_channels
    session.channel_names = [f"Ch{i+1}" for i in range(n_channels)]
    session.primary_stim = type("Stim", (), {"stim_type": "square"})()

    # Minimal dummy recordings so Session.stimulus_voltages property works
    class _DummyRec:
        def __init__(self, rid: str, v: float):
            self.id = rid
            self.meta = type(
                "Meta",
                (),
                {
                    "primary_stim": type("Stim", (), {"stim_v": v})(),
                    "scan_rate": scan_rate,
                    "num_channels": n_channels,
                },
            )()

        def raw_view(self):  # not used directly but kept for compatibility
            return np.zeros((1000, n_channels), dtype=float)

    session._all_recordings = [_DummyRec(f"rec{i}", float(i + 1)) for i in range(n_recordings)]
    # Provide fake filtered recordings for amplitude calculations
    rng = np.random.default_rng(0)
    filtered = [rng.normal(0, 0.1, size=(1000, n_channels)) for _ in range(n_recordings)]
    session.recordings_filtered = filtered  # monkey-patch attribute used by synthetic get_lw_reflex_amplitudes

    # Methods required by dataset aggregation
    def reset_all_caches():
        pass

    session.reset_all_caches = reset_all_caches

    def update_latency_window_parameters():
        # Derive m/h arrays based on window names
        session.m_start = [0.0] * n_channels
        session.m_duration = [0.0] * n_channels
        session.h_start = [0.0] * n_channels
        session.h_duration = [0.0] * n_channels
        for w in session.annot.latency_windows:
            if w.name.lower().startswith("m"):
                session.m_start = w.start_times
                session.m_duration = w.durations
            if w.name.lower().startswith("h"):
                session.h_start = w.start_times
                session.h_duration = w.durations

    session.update_latency_window_parameters = update_latency_window_parameters
    session.update_latency_window_parameters()
    # Minimal amplitude calc dependencies
    from monstim_signals.domain.session import calculate_emg_amplitude

    session.get_lw_reflex_amplitudes = lambda method, channel_index, window: np.array(
        [
            calculate_emg_amplitude(
                rec[:, channel_index],
                (next(w for w in session.annot.latency_windows if w.name == window).start_times[channel_index])
                + session.stim_start,
                (next(w for w in session.annot.latency_windows if w.name == window).end_times[channel_index])
                + session.stim_start,
                session.scan_rate,
                method=method,
            )
            for rec in filtered
        ]
    )
    session.get_latency_window = lambda name: next((w for w in session.annot.latency_windows if w.name == name), None)
    return session


def test_dataset_latency_window_heterogeneity():
    # Session A has M-wave, H-reflex, Late window
    sess_a = _make_session(
        "S_A",
        windows=[
            LatencyWindow(name="M-wave", start_times=[5.0, 5.0], durations=[5.0, 5.0], color="blue"),
            LatencyWindow(name="H-reflex", start_times=[15.0, 15.0], durations=[5.0, 5.0], color="red"),
            LatencyWindow(name="Late", start_times=[30.0, 30.0], durations=[10.0, 10.0], color="green"),
        ],
    )
    # Session B lacks the Late window
    sess_b = _make_session(
        "S_B",
        windows=[
            LatencyWindow(name="M-wave", start_times=[6.0, 6.0], durations=[4.0, 4.0], color="blue"),
            LatencyWindow(name="H-reflex", start_times=[16.0, 16.0], durations=[5.0, 5.0], color="red"),
        ],
    )

    ds_annot = DatasetAnnot.create_empty()
    dataset = Dataset(
        dataset_id="DS1",
        sessions=[sess_a, sess_b],
        annot=ds_annot,
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

    # Validate union window names
    names = dataset.unique_latency_window_names()
    assert names == ["H-reflex", "Late", "M-wave"], f"Unexpected union names: {names}"

    # Heterogeneity flag should be true
    assert dataset.has_heterogeneous_latency_windows is True

    presence = dataset.window_presence_map()
    assert set(presence.keys()) == set(names)
    assert presence["Late"] == ["S_A"], "Late window should only be in S_A"

    # Average curve for Late should only include one session's contribution
    late_curve = dataset.get_average_lw_reflex_curve(method="average_rectified", channel_index=0, window="Late")
    assert late_curve["voltages"].size > 0
    assert np.all(late_curve["n_sessions"] <= 1)

    # Notices should include heterogeneity
    notices = dataset.collect_notices()
    codes = {n["code"] for n in notices}
    assert "heterogeneous_latency_windows" in codes, f"Expected heterogeneity notice, got {codes}"
    # Should NOT flag missing M-wave since both sessions have an M-wave window
    assert "missing_m_wave_window" not in codes, "Unexpected missing_m_wave_window notice when M-wave present"


def test_experiment_latency_window_heterogeneity_curve():
    # Reuse sessions from the dataset test (one with Late window, one without)
    from monstim_signals.core import ExperimentAnnot
    from monstim_signals.domain.experiment import Experiment

    sess_a = _make_session(
        "S_A2",
        windows=[
            LatencyWindow(name="M-wave", start_times=[5.0, 5.0], durations=[5.0, 5.0], color="blue"),
            LatencyWindow(name="H-reflex", start_times=[15.0, 15.0], durations=[5.0, 5.0], color="red"),
            LatencyWindow(name="Late", start_times=[30.0, 30.0], durations=[10.0, 10.0], color="green"),
        ],
    )
    sess_b = _make_session(
        "S_B2",
        windows=[
            LatencyWindow(name="M-wave", start_times=[6.0, 6.0], durations=[4.0, 4.0], color="blue"),
            LatencyWindow(name="H-reflex", start_times=[16.0, 16.0], durations=[5.0, 5.0], color="red"),
        ],
    )

    ds_annot = DatasetAnnot.create_empty()
    dataset1 = Dataset(
        dataset_id="DS_E1",
        sessions=[sess_a],
        annot=ds_annot,
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
    ds_annot2 = DatasetAnnot.create_empty()
    dataset2 = Dataset(
        dataset_id="DS_E2",
        sessions=[sess_b],
        annot=ds_annot2,
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

    exp = Experiment(
        expt_id="EXP1",
        datasets=[dataset1, dataset2],
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

    # Validate experiment union includes Late
    names = exp.unique_latency_window_names()
    assert "Late" in names, f"Late window missing from experiment union: {names}"

    late_curve = exp.get_average_lw_reflex_curve(method="average_rectified", channel_index=0, window="Late")
    assert late_curve["voltages"].size > 0
    assert (late_curve["n_sessions"] <= 1).all(), "Late window contributions should be from at most one session per bin"

    notices = exp.collect_notices()
    codes = {n["code"] for n in notices}
    assert "heterogeneous_latency_windows" in codes, f"Expected experiment heterogeneity notice, got {codes}"
    # M-wave exists in at least one dataset; experiment-level missing M-wave should NOT appear
    assert "missing_m_wave_window" not in codes
