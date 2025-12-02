from __future__ import annotations

from typing import Tuple

import numpy as np


def _as_numpy(x):
    try:
        return np.asarray(x)
    except Exception:
        return x


def decimate_series(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int = 10000,
    strategy: str = "minmax",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce the number of plotted points while preserving visual features.

    Parameters
    ----------
    x : np.ndarray
        Time axis (1D)
    y : np.ndarray
        Signal values (1D)
    max_points : int
        Maximum number of points to return (approximate upper bound)
    strategy : str
        One of {'minmax','mean','subsample'}

    Returns
    -------
    (x_d, y_d) : Tuple[np.ndarray, np.ndarray]
        Decimated series suitable for plotting
    """
    if max_points is None or max_points <= 0:
        return _as_numpy(x), _as_numpy(y)

    x = _as_numpy(x)
    y = _as_numpy(y)

    n = len(y)
    if n <= max_points:
        return x, y

    # Strategy-specific bin sizing
    if strategy == "minmax":
        # 2 points per bin → ensure total <= max_points
        bins = max(1, max_points // 2)
        step = int(np.ceil(n / bins))
    else:
        step = int(np.ceil(n / max_points))

    if step <= 1:
        return x, y

    if strategy == "subsample":
        return x[::step], y[::step]

    if strategy == "mean":
        # Pad to multiple of step
        pad = (-n) % step
        if pad:
            y_pad = np.pad(y, (0, pad), mode="edge")
            x_pad = np.pad(x, (0, pad), mode="edge")
        else:
            y_pad, x_pad = y, x
        y_b = y_pad.reshape(-1, step).mean(axis=1)
        x_b = x_pad.reshape(-1, step).mean(axis=1)
        return x_b, y_b

    if strategy == "minmax":
        # Vectorized min/max per bin with interleaved order
        pad = (-n) % step
        if pad:
            y_pad = np.pad(y, (0, pad), mode="edge")
            x_pad = np.pad(x, (0, pad), mode="edge")
        else:
            y_pad, x_pad = y, x
        y2 = y_pad.reshape(-1, step)
        x2 = x_pad.reshape(-1, step)
        rows = np.arange(y2.shape[0])
        idx_min = np.argmin(y2, axis=1)
        idx_max = np.argmax(y2, axis=1)
        # Ensure order within bin by index to avoid backtracking
        left = np.minimum(idx_min, idx_max)
        right = np.maximum(idx_min, idx_max)
        x_left = x2[rows, left]
        y_left = y2[rows, left]
        x_right = x2[rows, right]
        y_right = y2[rows, right]
        xs = np.empty(x_left.size * 2, dtype=x_left.dtype)
        ys = np.empty(y_left.size * 2, dtype=y_left.dtype)
        xs[0::2] = x_left
        xs[1::2] = x_right
        ys[0::2] = y_left
        ys[1::2] = y_right
        return xs, ys

    # Fallback
    return x[::step], y[::step]
