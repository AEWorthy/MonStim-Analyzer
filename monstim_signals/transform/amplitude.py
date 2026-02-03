"""Amplitude calculation utilities."""

import numpy as np


def rectify_emg(emg_array):
    """Rectify EMG data by taking the absolute value."""
    return np.abs(emg_array)


def _calculate_average_amplitude_rectified(emg_data, start_ms, end_ms, scan_rate):
    """Calculate the average amplitude of rectified EMG signal.

    Rectifies the EMG signal (absolute value) and returns the mean amplitude.

    Args:
        emg_data: EMG signal array
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        scan_rate: Sampling rate in Hz

    Returns:
        float: Mean of rectified signal, or np.nan if window is empty
    """
    start_index = int(start_ms * scan_rate / 1000)
    end_index = int(end_ms * scan_rate / 1000)
    emg_window = emg_data[start_index:end_index]
    if emg_window.size == 0:
        return np.nan
    rectified_emg_window = rectify_emg(emg_window)
    return np.mean(rectified_emg_window)


def _calculate_peak_to_trough_amplitude(emg_data, start_ms, end_ms, scan_rate):
    """Calculate peak-to-trough amplitude of EMG signal.

    Returns the difference between maximum and minimum values in the window.

    Args:
        emg_data: EMG signal array
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        scan_rate: Sampling rate in Hz

    Returns:
        float: Maximum minus minimum value, or np.nan if window is empty
    """
    start_index = int(start_ms * scan_rate / 1000)
    end_index = int(end_ms * scan_rate / 1000)
    emg_window = emg_data[start_index:end_index]
    if emg_window.size == 0:
        return np.nan
    return np.max(emg_window) - np.min(emg_window)


def _calculate_rms_amplitude(emg_data, start_ms, end_ms, scan_rate):
    """Calculate root mean square (RMS) amplitude of EMG signal.

    Computes the square root of the mean of squared signal values.

    Args:
        emg_data: EMG signal array
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        scan_rate: Sampling rate in Hz

    Returns:
        float: RMS amplitude, or np.nan if window is empty
    """
    start_index = int(start_ms * scan_rate / 1000)
    end_index = int(end_ms * scan_rate / 1000)
    emg_window = emg_data[start_index:end_index]
    if emg_window.size == 0:
        return np.nan
    squared_emg_window = np.square(emg_window)
    mean_squared_value = np.mean(squared_emg_window)
    return np.sqrt(mean_squared_value)


def _calculate_average_amplitude_unrectified(emg_data, start_ms, end_ms, scan_rate):
    """Calculate the average amplitude of unrectified EMG signal.

    Returns the mean of the raw signal without rectification.

    Args:
        emg_data: EMG signal array
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        scan_rate: Sampling rate in Hz

    Returns:
        float: Mean of raw signal, or np.nan if window is empty
    """
    start_index = int(start_ms * scan_rate / 1000)
    end_index = int(end_ms * scan_rate / 1000)
    emg_window = emg_data[start_index:end_index]
    if emg_window.size == 0:
        return np.nan
    return np.mean(emg_window)


def _calculate_auc_rectified(emg_data, start_ms, end_ms, scan_rate):
    """Calculate area under the curve (AUC) of rectified EMG signal.

    Rectifies the signal and computes the area under the curve by summing
    absolute values and multiplying by the time interval between samples.

    Args:
        emg_data: EMG signal array
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        scan_rate: Sampling rate in Hz

    Returns:
        float: Area under rectified curve in V·s, or np.nan if window is empty
    """
    start_index = int(start_ms * scan_rate / 1000)
    end_index = int(end_ms * scan_rate / 1000)
    emg_window = emg_data[start_index:end_index]
    if emg_window.size == 0:
        return np.nan
    rectified_emg_window = rectify_emg(emg_window)
    # Time interval between samples in seconds
    dt = 1.0 / scan_rate
    # Area under curve = sum of values * time interval
    return np.sum(rectified_emg_window) * dt


def calculate_emg_amplitude(emg_data, start_ms, end_ms, scan_rate, method):
    """Calculate the EMG amplitude using the specified method.

    Args:
        emg_data: EMG signal array
        start_ms: Start time of analysis window in milliseconds
        end_ms: End time of analysis window in milliseconds
        scan_rate: Sampling rate in Hz
        method: Calculation method - one of:
            - 'average_rectified': Mean of absolute values
            - 'peak_to_trough': Maximum minus minimum
            - 'rms': Root mean square
            - 'average_unrectified': Mean of raw signal
            - 'auc': Area under rectified curve

    Returns:
        float: Calculated amplitude value

    Raises:
        ValueError: If method is not recognized
    """
    methods = {
        "average_rectified": _calculate_average_amplitude_rectified,
        "peak_to_trough": _calculate_peak_to_trough_amplitude,
        "rms": _calculate_rms_amplitude,
        "average_unrectified": _calculate_average_amplitude_unrectified,
        "auc": _calculate_auc_rectified,
    }
    if method not in methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of {', '.join(methods.keys())}")
    calculation_function = methods[method]
    return calculation_function(emg_data, start_ms, end_ms, scan_rate)
