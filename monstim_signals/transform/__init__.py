"""Signal transformation utilities."""

from .amplitude import calculate_emg_amplitude, rectify_emg
from .filtering import butter_bandpass, butter_bandpass_filter, correct_emg_to_baseline
from .plateau import (
    NoCalculableMmaxError,
    detect_plateau,
    get_avg_mmax,
    savgol_filter_y,
)

__all__ = [
    "butter_bandpass",
    "butter_bandpass_filter",
    "correct_emg_to_baseline",
    "rectify_emg",
    "calculate_emg_amplitude",
    "savgol_filter_y",
    "detect_plateau",
    "get_avg_mmax",
    "NoCalculableMmaxError",
]
