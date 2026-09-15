"""Filtering utilities for MonStim signals."""

import numpy as np
from scipy import signal


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int):
    """Design a Butterworth bandpass filter.

    Args:
        lowcut (float): The lower cutoff frequency.
        highcut (float): The upper cutoff frequency.
        fs (float): The sampling frequency.
        order (int): The order of the filter.
    Returns:
        tuple: The filter coefficients (b, a).
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data: np.ndarray, fs: float, lowcut=100, highcut=3500, order=4):
    """Apply a Butterworth bandpass filter to a 1D array of 'data'.

    Args:
        data (array): The input data to be filtered.
        fs (float): The sampling frequency.
        lowcut (float): The lower cutoff frequency.
        highcut (float): The upper cutoff frequency.
        order (int): The order of the filter.
    Returns:
        array: The filtered data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return signal.filtfilt(b, a, data)


def correct_emg_to_baseline(channel_recording: np.ndarray, scan_rate: float, baseline_end_ms: float):
    """Correct a signal relative to its recorded pre-stimulus baseline.

    Args:
        channel_recording (array): The EMG signal to be corrected.
        scan_rate (float): The scanning rate of the signal.
        baseline_end_ms (float): The acquisition-relative time of stimulus
            onset. Samples before this point form the baseline.
    Returns:
        array: The corrected EMG signal.
    """
    # Baseline correction is not a latency-window measurement. It covers the
    # samples strictly before stimulus onset, so the onset sample itself does
    # not influence the baseline estimate. A recording with no pre-stimulus
    # samples cannot be baseline-corrected; preserve it rather than converting
    # the complete channel to NaN through ``mean([])``.
    baseline_end_sample = min(int(baseline_end_ms * scan_rate / 1000), len(channel_recording))
    if baseline_end_sample <= 0:
        return np.array(channel_recording, copy=True)
    baseline_emg = np.mean(channel_recording[:baseline_end_sample])
    return channel_recording - baseline_emg
