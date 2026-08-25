from types import SimpleNamespace

import numpy as np

from monstim_signals.core import LatencyWindow, SessionAnnot
from monstim_signals.domain.session import Session


def _window(name: str) -> LatencyWindow:
    return LatencyWindow(name=name, start_times=[1.0], durations=[2.0], color="white")


def test_latency_window_amplitudes_reuse_one_all_window_calculation():
    """One aggregate plot must not recalculate extrema once per window."""
    session = Session.__new__(Session)
    session.annot = SessionAnnot.create_empty()
    session.annot.latency_windows = [_window("M-wave"), _window("H-reflex")]
    session.annot.excluded_recordings = []
    calls = 0

    def all_results(method, channel_index):
        nonlocal calls
        calls += 1
        assert (method, channel_index) == ("exclusive_extrema_ptt", 0)
        return (
            SimpleNamespace(results=(SimpleNamespace(amplitude=1.0), SimpleNamespace(amplitude=2.0))),
            SimpleNamespace(results=(SimpleNamespace(amplitude=3.0), SimpleNamespace(amplitude=4.0))),
        )

    session.get_all_lw_reflex_amplitude_results = all_results

    m_wave = session.get_lw_reflex_amplitudes("exclusive_extrema_ptt", 0, "M-wave")
    h_wave = session.get_lw_reflex_amplitudes("exclusive_extrema_ptt", 0, "H-reflex")

    np.testing.assert_array_equal(m_wave, [1.0, 2.0])
    np.testing.assert_array_equal(h_wave, [3.0, 4.0])
    assert calls == 1

    # Callers receive a copy, so their edits cannot poison a later plot.
    m_wave[0] = 99.0
    np.testing.assert_array_equal(session.get_lw_reflex_amplitudes("exclusive_extrema_ptt", 0, "M-wave"), [1.0, 2.0])
    assert calls == 1
