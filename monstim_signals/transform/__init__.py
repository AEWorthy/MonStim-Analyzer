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
    "NoCalculableMmaxError",
    "butter_bandpass",
    "butter_bandpass_filter",
    "calculate_emg_amplitude",
    "correct_emg_to_baseline",
    "detect_plateau",
    "get_avg_mmax",
    "rectify_emg",
    "savgol_filter_y",
]
