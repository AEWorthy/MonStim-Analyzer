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


def correct_emg_to_baseline(channel_recording: np.ndarray, scan_rate: float, stim_delay: float):
    """Correct EMG absolute amplitude relative to pre-stim baseline amplitude by
    subtracting the average pre-stimulus amplitude from the entire signal.

    Args:
        channel_recording (array): The EMG signal to be corrected.
        scan_rate (float): The scanning rate of the signal.
        stim_delay (float): The delay between stimulus and signal acquisition.
    Returns:
        array: The corrected EMG signal.
    """
    # Baseline correction is not a latency-window measurement.  It covers the
    # samples strictly before stimulus onset, so the sample at stim_delay does
    # not influence the baseline estimate.
    baseline_end_sample = int(stim_delay * scan_rate / 1000)
    baseline_emg = np.mean(channel_recording[:baseline_end_sample])
    return channel_recording - baseline_emg
