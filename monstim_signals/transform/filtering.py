"""Filtering utilities for MonStim signals."""
import numpy as np
from scipy import signal


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
