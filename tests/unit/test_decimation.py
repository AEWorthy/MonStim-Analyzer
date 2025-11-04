import numpy as np

from monstim_signals.plotting import decimate_series


def test_decimate_minmax_reduces_points_and_preserves_extrema():
    # Create a signal with spikes to test extrema preservation
    n = 50000
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * 50 * x).astype(np.float32)
    # Add a sharp spike
    y[12345] = 5.0
    y[23456] = -6.0

    xd, yd = decimate_series(x, y, max_points=2000, strategy="minmax")

    assert len(yd) <= 2 * 2000 + 2  # minmax returns ~2 points per bin
    # Check that spike extrema are still represented (approximate envelope preservation)
    assert yd.max() >= 5.0 - 1e-6
    assert yd.min() <= -6.0 + 1e-6


def test_decimate_mean_reduces_points():
    n = 10000
    x = np.arange(n)
    y = np.arange(n).astype(float)
    xd, yd = decimate_series(x, y, max_points=100, strategy="mean")
    assert len(yd) <= 100
    assert len(xd) == len(yd)


def test_decimate_subsample_stride():
    n = 10000
    x = np.arange(n)
    y = np.arange(n).astype(float)
    xd, yd = decimate_series(x, y, max_points=100, strategy="subsample")
    assert len(yd) <= 100
    assert len(xd) == len(yd)
