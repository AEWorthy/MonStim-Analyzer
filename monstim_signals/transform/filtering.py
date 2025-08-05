"""Filtering utilities for MonStim signals."""

from scipy import signal

from .amplitude import _calculate_average_amplitude_unrectified


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int):
    """Design a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, fs, lowcut=100, highcut=3500, order=4):
    """Apply a Butterworth bandpass filter to *data*."""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return signal.filtfilt(b, a, data)


def correct_emg_to_baseline(channel_recording, scan_rate, stim_delay):
    """Correct EMG absolute amplitude relative to pre-stim baseline amplitude."""
    baseline_emg = _calculate_average_amplitude_unrectified(channel_recording, 0, stim_delay, scan_rate)
    return channel_recording - baseline_emg
