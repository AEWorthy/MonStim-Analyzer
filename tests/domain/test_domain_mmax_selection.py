"""
Domain M-max Selection Logic

Purpose: Validate that Session.get_m_max uses a configured M-response window to
calculate M-max from a recruitment curve with a stable plateau.
Markers: unit (synthetic data), integration optional for real data.
"""

from __future__ import annotations

import numpy as np
import pytest

from monstim_signals.core import RecordingAnnot, RecordingMeta, SessionAnnot, StimCluster
from monstim_signals.domain import Recording, Session
pytestmark = pytest.mark.unit


def make_session_with_synth_data(levels=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5), num_channels=1) -> Session:
    recs = []
    n = 1000
    t = np.linspace(0, 0.1, n)
    for i, v in enumerate(levels):
        # Build a fully-specified StimCluster to satisfy current data model
        stim = StimCluster(
            stim_delay=2.0,
            stim_duration=1.0,
            stim_type="Electrical",
            stim_v=float(v),
            stim_min_v=float(v),
            stim_max_v=float(v),
            pulse_shape="Square",
            num_pulses=1,
            pulse_period=1.0,
            peak_duration=0.1,
            ramp_duration=0.0,
        )
        meta = RecordingMeta(
            recording_id=f"r{i}",
            num_samples=n,
            num_channels=num_channels,
            scan_rate=10000,
            recording_interval=1.0,
            channel_types=["EMG"] * num_channels,
            emg_amp_gains=[1000] * num_channels,
            stim_clusters=[stim],
            primary_stim=stim,
            pre_stim_acquired=20,
            post_stim_acquired=20,
        )
        # M-wave recruitment rises through 2.0 V then remains at a stable
        # plateau. Seven levels allow the plateau detector to validate the
        # high-stimulus region.
        data = np.zeros((n, num_channels))
        burst = (t > 0.005) & (t < 0.015)
        response_scale = min(v, 2.0)
        data[burst, 0] = np.sin(2 * np.pi * 1000 * t[burst]) * (response_scale * 100.0)
        recs.append(Recording(meta=meta, annot=RecordingAnnot.create_empty(), raw=data))

    annot = SessionAnnot.create_empty(num_channels)
    sess = Session(session_id="S-synth", recordings=recs, annot=annot)

    # Add M-window so M-max can be computed
    sess.add_latency_window(
        name="M-response",
        start_times=[5.0] * num_channels,
        durations=[10.0] * num_channels,
        color="blue",
    )
    return sess


def test_get_m_max_with_latency_window():
    sess = make_session_with_synth_data()
    try:
        m_wave_amplitudes = sess.get_m_wave_amplitudes(method="rms", channel_index=0)
        mmax = sess.get_m_max(method="rms", channel_index=0)
        plateau_amplitude = max(m_wave_amplitudes)

        assert m_wave_amplitudes[0] < plateau_amplitude
        assert m_wave_amplitudes[-3:] == pytest.approx([plateau_amplitude] * 3)
        assert mmax == pytest.approx(plateau_amplitude)
    finally:
        sess.close()
