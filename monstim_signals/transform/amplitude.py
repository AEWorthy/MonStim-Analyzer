"""Amplitude calculation utilities."""

import numpy as np


def rectify_emg(emg_array):
    """Rectify EMG data by taking the absolute value."""
    return np.abs(emg_array)


def _calculate_average_amplitude_rectified(emg_data, start_ms, end_ms, scan_rate):
    start_index = int(start_ms * scan_rate / 1000)
    end_index = int(end_ms * scan_rate / 1000)
    emg_window = emg_data[start_index:end_index]
    if emg_window.size == 0:
        return np.nan
    rectified_emg_window = rectify_emg(emg_window)
    return np.mean(rectified_emg_window)


def _calculate_peak_to_trough_amplitude(emg_data, start_ms, end_ms, scan_rate):
    start_index = int(start_ms * scan_rate / 1000)
    end_index = int(end_ms * scan_rate / 1000)
    emg_window = emg_data[start_index:end_index]
    if emg_window.size == 0:
        return np.nan
    return np.max(emg_window) - np.min(emg_window)


def _calculate_rms_amplitude(emg_data, start_ms, end_ms, scan_rate):
    start_index = int(start_ms * scan_rate / 1000)
    end_index = int(end_ms * scan_rate / 1000)
    emg_window = emg_data[start_index:end_index]
    if emg_window.size == 0:
        return np.nan
    squared_emg_window = np.square(emg_window)
    mean_squared_value = np.mean(squared_emg_window)
    return np.sqrt(mean_squared_value)


def _calculate_average_amplitude_unrectified(emg_data, start_ms, end_ms, scan_rate):
    start_index = int(start_ms * scan_rate / 1000)
    end_index = int(end_ms * scan_rate / 1000)
    emg_window = emg_data[start_index:end_index]
    if emg_window.size == 0:
        return np.nan
    return np.mean(emg_window)


def calculate_emg_amplitude(emg_data, start_ms, end_ms, scan_rate, method):
    """Calculate the EMG amplitude using the specified *method*."""
    methods = {
        "average_rectified": _calculate_average_amplitude_rectified,
        "peak_to_trough": _calculate_peak_to_trough_amplitude,
        "rms": _calculate_rms_amplitude,
        "average_unrectified": _calculate_average_amplitude_unrectified,
    }
    if method not in methods:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of {', '.join(methods.keys())}"
        )
    calculation_function = methods[method]
    return calculation_function(emg_data, start_ms, end_ms, scan_rate)
