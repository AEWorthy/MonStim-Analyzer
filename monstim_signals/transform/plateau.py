"""Utilities for plateau detection and M-max calculations."""
import logging
import numpy as np
from scipy import signal


def savgol_filter_y(y, polyorder=3):
    """Smooth data using a Savitzky-Golay filter."""
    window_length = int((len(y) / 100) * 25)
    window_length = max(window_length, polyorder + 2)  # Ensure minimum size
    if window_length % 2 == 0:  # Ensure oddness
        window_length += 1
    return signal.savgol_filter(y, window_length, min(polyorder, window_length - 1))


def detect_plateau(x, y, max_window_size, min_window_size, threshold):
    """Detect a plateau region in a reflex curve."""
    plateau_start_idx = None
    plateau_end_idx = None
    y_filtered = savgol_filter_y(y)
    for i in range(len(y_filtered) - max_window_size):
        window = y_filtered[i : i + max_window_size]
        if np.std(window) < threshold:
            if plateau_start_idx is None:
                plateau_start_idx = i
            plateau_end_idx = i + max_window_size
        else:
            plateau_start_idx = None
            plateau_end_idx = None
    if plateau_start_idx and plateau_end_idx is not None:
        logging.info(
            f"Plateau region detected with window size {max_window_size}. Threshold: {threshold} times SD."
        )
        return plateau_start_idx, plateau_end_idx
    elif max_window_size > min_window_size:
        return detect_plateau(x, y, max_window_size - 1, min_window_size, threshold)
    else:
        logging.warning("No plateau region detected.")
        return None, None


def get_avg_mmax(
    stimulus_voltages,
    m_wave_amplitudes,
    max_window_size=20,
    min_window_size=3,
    threshold=0.3,
    return_mmax_stim_range=False,
):
    """Return the M-max amplitude and optionally its stimulus range."""
    m_wave_amplitudes = np.array(m_wave_amplitudes)
    plateau_start_idx, plateau_end_idx = detect_plateau(
        stimulus_voltages, m_wave_amplitudes, max_window_size, min_window_size, threshold
    )
    if plateau_start_idx is not None and plateau_end_idx is not None:
        plateau_data = np.array(m_wave_amplitudes[plateau_start_idx:plateau_end_idx])
        m_max = np.mean(plateau_data)
        if m_max < max(m_wave_amplitudes):
            m_max = m_max + np.mean(m_wave_amplitudes[m_wave_amplitudes > m_max]) - np.mean(
                plateau_data[plateau_data < np.max(plateau_data)]
            )
            logging.info(
                f"\tM-max corrected by: {np.mean(m_wave_amplitudes[m_wave_amplitudes > m_max]) - np.mean(plateau_data)}"
            )
        logging.info(f"\tM-max amplitude: {m_max}")
        if return_mmax_stim_range:
            return m_max, stimulus_voltages[plateau_start_idx], stimulus_voltages[plateau_end_idx]
        return m_max
    raise NoCalculableMmaxError()


class NoCalculableMmaxError(Exception):
    def __init__(self, message="No calculable M-max. Try adjusting the threshold values."):
        super().__init__(message)
