"""Signal transformation utilities."""

from .filtering import butter_bandpass, butter_bandpass_filter, correct_emg_to_baseline
from .amplitude import (
    rectify_emg,
    calculate_emg_amplitude,
)
from .plateau import (
    savgol_filter_y,
    detect_plateau,
    get_avg_mmax,
    NoCalculableMmaxError,
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
