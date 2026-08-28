"""Utilities for plateau detection and M-max calculations."""

import logging

logger = logging.getLogger(__name__)

import numpy as np
from scipy.signal import savgol_filter


def savgol_filter_y(
    y: np.ndarray,
    polyorder: int = 3,
    window_length: int | None = None,
    window_ratio: float = 0.25,
):
    """Smooth data using a Savitzky-Golay filter.

    Args:
        y (np.ndarray): The input signal array.
        polyorder (int): The order of the polynomial used to fit the samples. Must be less than window_length.
        window_length (int or None): Explicit Savitzky-Golay window length. If omitted,
            it is calculated from ``window_ratio`` and the signal length.
        window_ratio (float): Fraction of the signal length used to calculate the
            window when ``window_length`` is omitted.
    Returns:
        np.ndarray: The smoothed signal.
    """
    if window_length is None:
        if window_ratio <= 0:
            raise ValueError("window_ratio must be greater than zero")
        window_length = int(len(y) * window_ratio)
    else:
        window_length = int(window_length)
    window_length = max(window_length, polyorder + 2)  # Ensure minimum size
    if window_length % 2 == 0:  # Ensure oddness
        window_length += 1
    return savgol_filter(y, window_length, min(polyorder, window_length - 1))


def detect_plateau(
    y: np.ndarray,
    max_window_size: int,
    min_window_size: int,
    threshold: float,
    savgol_window_length: int | None = None,
    savgol_window_ratio: float = 0.25,
):
    """Detect a plateau region in a reflex curve.

    A plateau is defined as a region where the standard deviation of the signal
    is below a certain threshold for a specified window size. The function
    recursively reduces the window size if no plateau is found, down to a minimum
    window size. If no plateau is detected, it returns None.

    Args:
        y (np.ndarray): The input signal array.
        max_window_size (int): The maximum size of the sliding window to check for a plateau.
        min_window_size (int): The minimum size of the sliding window to check for a plateau.
        threshold (float): The threshold for standard deviation to consider a region as a plateau.

    Returns:
        tuple: (start_index, end_index) of the detected plateau region, or (None, None) if no plateau is found.
    """
    plateau_start_idx = None
    plateau_end_idx = None
    y_filtered = savgol_filter_y(y, window_length=savgol_window_length, window_ratio=savgol_window_ratio)
    for i in range(len(y_filtered) - max_window_size):
        window = y_filtered[i : i + max_window_size]
        if np.std(window) < threshold:
            if plateau_start_idx is None:
                plateau_start_idx = i
            plateau_end_idx = i + max_window_size
        else:
            plateau_start_idx = None
            plateau_end_idx = None
    if plateau_start_idx is not None and plateau_end_idx is not None:
        logger.debug(f"Plateau region detected with window size {max_window_size}. Threshold: {threshold} times SD.")
        return plateau_start_idx, plateau_end_idx
    elif max_window_size > min_window_size:
        return detect_plateau(
            y,
            max_window_size - 1,
            min_window_size,
            threshold,
            savgol_window_length=savgol_window_length,
            savgol_window_ratio=savgol_window_ratio,
        )
    else:
        logger.warning("No plateau region detected.")
        return None, None


def get_avg_mmax(
    stimulus_voltages: list | np.ndarray,
    m_wave_amplitudes: list | np.ndarray,
    max_window_size=20,
    min_window_size=3,
    threshold=0.3,
    validation_tolerance=1.05,
    savgol_window_length=None,
    savgol_window_ratio=0.25,
    return_mmax_stim_range=False,
):
    """
    Return the average M-max amplitude of the given M-wave amplitudes and optionally return the stimulus range of the detected M-max plateau.

    Uses an algorithm that tries multiple calculation approaches:
    1. Maximum value in the plateau (stable, high-stimulus) region
    2. High-percentile approach in the plateau region
    3. Mean of top 20% values in the plateau region
    4. Traditional plateau detection with averaging

    If no plateau is detected, it falls back to analyzing the high-stimulus region.

    Args:
        stimulus_voltages (list or np.ndarray): Stimulus voltages corresponding to M-wave amplitudes.
        m_wave_amplitudes (list or np.ndarray): M-wave amplitudes corresponding to stimulus voltages.
        max_window_size (int): Maximum window size for plateau detection.
        min_window_size (int): Minimum window size for plateau detection.
        threshold (float): Threshold for standard deviation to consider a region as a plateau.
        validation_tolerance (float): Tolerance factor for validating M-max against plateau mean.
        savgol_window_length (int or None): Explicit smoothing window length.
        savgol_window_ratio (float): Fraction of the signal length used for the smoothing window
            when ``savgol_window_length`` is omitted.
        return_mmax_stim_range (bool): If True, return the stimulus range corresponding to the detected M-max.

    Returns:
        float or tuple: M-max amplitude, and optionally the stimulus range (start, end)

    Raises:
        NoCalculableMmaxError: If no calculable M-max can be determined.
    """
    m_wave_amplitudes = np.array(m_wave_amplitudes)
    stimulus_voltages = np.array(stimulus_voltages)

    try:
        plateau_start_idx, plateau_end_idx = detect_plateau(
            m_wave_amplitudes,
            max_window_size,
            min_window_size,
            threshold,
            savgol_window_length=savgol_window_length,
            savgol_window_ratio=savgol_window_ratio,
        )

    except Exception as e:
        logger.exception(f"Exception during plateau detection: {e}")
        logger.exception("Exception during plateau detection. Proceeding to fallback methods.")
        plateau_start_idx, plateau_end_idx = None, None

    if plateau_start_idx is not None and plateau_end_idx is not None:
        plateau_data = np.array(m_wave_amplitudes[plateau_start_idx:plateau_end_idx])

        # Use multiple approaches and take the most appropriate one
        approaches = []

        # Approach 1: Traditional mean with correction (most conservative)
        m_max_mean = np.mean(plateau_data)
        if m_max_mean < max(m_wave_amplitudes):
            outliers = m_wave_amplitudes[m_wave_amplitudes > m_max_mean]
            plateau_below_max = plateau_data[plateau_data < np.max(plateau_data)]
            if outliers.size > 0 and plateau_below_max.size > 0:
                correction = np.mean(outliers) - np.mean(plateau_below_max)
                m_max_mean = m_max_mean + correction
        approaches.append(("mean_corrected", m_max_mean))

        # Approach 2: 95th percentile of plateau region
        m_max_p95 = np.percentile(plateau_data, 95)
        approaches.append(("95th_percentile", m_max_p95))

        # Approach 3: Maximum value in plateau region (most aggressive)
        m_max_max = np.max(plateau_data)
        approaches.append(("maximum", m_max_max))

        # Approach 4: Mean of top 20% of plateau values (balanced)
        top_20_percent_threshold = np.percentile(plateau_data, 80)
        top_values = plateau_data[plateau_data >= top_20_percent_threshold]
        if len(top_values) > 0:
            m_max_top20 = np.mean(top_values)
            approaches.append(("top_20_percent_mean", m_max_top20))

        # Selection logic: prefer maximum approach if it's not too extreme relative to plateau mean
        plateau_mean = np.mean(plateau_data)

        # Improved validation: compare against plateau mean, not global maximum
        # This prevents artifacts from dominating and ensures plateau consistency
        if m_max_max <= plateau_mean * validation_tolerance:
            m_max = m_max_max
            selected_approach = "maximum"
            validation_note = f"within {validation_tolerance:.1%} of plateau mean"
        # Otherwise, try 95th percentile
        elif m_max_p95 <= plateau_mean * validation_tolerance:
            m_max = m_max_p95
            selected_approach = "95th_percentile"
            validation_note = f"within {validation_tolerance:.1%} of plateau mean"
        # Otherwise, try top 20% mean
        elif len(top_values) > 0 and m_max_top20 <= plateau_mean * validation_tolerance:
            m_max = m_max_top20
            selected_approach = "top_20_percent_mean"
            validation_note = f"within {validation_tolerance:.1%} of plateau mean"
        else:
            # Fallback to traditional approach
            m_max = m_max_mean
            selected_approach = "mean_corrected"
            validation_note = "fallback - other approaches exceeded tolerance"

        logger.debug(f"\tM-max calculation: selected '{selected_approach}' approach, value: {m_max}")
        logger.debug(f"\t  Validation: {validation_note}")

        # Log all approaches for debugging
        for name, val in approaches:
            logger.debug(f"\t  {name}: {val:.6f}")
        logger.debug(f"\t  plateau_mean: {plateau_mean:.6f}")
        logger.debug(f"\t  validation_tolerance: {validation_tolerance:.3f}")

        # Final validation: ensure M-max is reasonable compared to global maximum
        max_overall = np.max(m_wave_amplitudes)
        if m_max > max_overall:
            logger.warning(f"\tM-max ({m_max}) > max amplitude ({max_overall}), capping at max")
            m_max = max_overall

        logger.debug(f"\tFinal M-max amplitude: {m_max}")
        if return_mmax_stim_range:
            return (
                m_max,
                stimulus_voltages[plateau_start_idx],
                stimulus_voltages[plateau_end_idx],
            )
        return m_max

    # Fallback: if no plateau detected, apply multi-approach method to high-stimulus region
    # Look for the best estimate in the top 25% of stimulus intensities
    logger.warning("No plateau detected, using fallback multi-approach detection in high-stimulus region")

    # Sort by stimulus voltage and take top 25%
    sorted_indices = np.argsort(stimulus_voltages)
    top_25_percent = int(len(sorted_indices) * 0.75)
    high_stim_indices = sorted_indices[top_25_percent:]

    if len(high_stim_indices) > 0:
        high_stim_amplitudes = m_wave_amplitudes[high_stim_indices]

        # Apply the same multi-approach methodology to the high-stimulus region
        approaches = []

        # Approach 1: Mean (no correction needed since we're in high-stimulus region)
        m_max_mean = np.mean(high_stim_amplitudes)
        approaches.append(("mean", m_max_mean))

        # Approach 2: 95th percentile
        m_max_p95 = np.percentile(high_stim_amplitudes, 95)
        approaches.append(("95th_percentile", m_max_p95))

        # Approach 3: Maximum value
        m_max_max = np.max(high_stim_amplitudes)
        approaches.append(("maximum", m_max_max))

        # Skip top 20% mean for fallback (would be same as mean for small regions)

        # Selection logic: prefer maximum if reasonable, same validation as main algorithm
        region_mean = np.mean(high_stim_amplitudes)

        if m_max_max <= region_mean * validation_tolerance:
            m_max = m_max_max
            selected_approach = "maximum"
            validation_note = f"within {validation_tolerance:.1%} of high-stim region mean"
        elif m_max_p95 <= region_mean * validation_tolerance:
            m_max = m_max_p95
            selected_approach = "95th_percentile"
            validation_note = f"within {validation_tolerance:.1%} of high-stim region mean"
        else:
            m_max = m_max_mean
            selected_approach = "mean"
            validation_note = "fallback to mean - other approaches exceeded tolerance"

        logger.debug(f"\tFallback M-max calculation: selected '{selected_approach}' approach, value: {m_max}")
        logger.debug(f"\t  Validation: {validation_note}")

        # Log all approaches for debugging
        for name, val in approaches:
            logger.debug(f"\t  {name}: {val:.6f}")
        logger.debug(f"\t  high_stim_region_mean: {region_mean:.6f}")
        logger.debug(f"\t  validation_tolerance: {validation_tolerance:.3f}")

        logger.debug(f"\tFallback M-max amplitude: {m_max}")
        if return_mmax_stim_range:
            return (
                m_max,
                stimulus_voltages[high_stim_indices[0]],
                stimulus_voltages[high_stim_indices[-1]],
            )
        return m_max

    raise NoCalculableMmaxError()


class NoCalculableMmaxError(Exception):
    """Custom exception raised when no calculable M-max can be determined."""

    def __init__(self, message="No calculable M-max. Try adjusting the threshold values."):
        super().__init__(message)
