"""Deprecated compatibility module. Use :mod:`monstim_signals.transform` instead."""
from .transform import (
    butter_bandpass,
    butter_bandpass_filter,
    correct_emg_to_baseline,
    rectify_emg,
    calculate_emg_amplitude,
    savgol_filter_y,
    detect_plateau,
    get_avg_mmax,
    NoCalculableMmaxError,
)
