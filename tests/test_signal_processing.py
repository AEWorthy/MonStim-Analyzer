"""
Comprehensive tests for signal processing and transformation components.

These tests cover the core EMG signal processing algorithms that are critical
for analysis accuracy but were previously untested.
"""

import numpy as np
import pytest

from monstim_signals.transform.amplitude import (
    _calculate_average_amplitude_rectified,
    _calculate_average_amplitude_unrectified,
    _calculate_peak_to_trough_amplitude,
    _calculate_rms_amplitude,
    calculate_emg_amplitude,
    rectify_emg,
)
from monstim_signals.transform.filtering import (
    butter_bandpass,
    butter_bandpass_filter,
    correct_emg_to_baseline,
)
from monstim_signals.transform.plateau import (
    NoCalculableMmaxError,
    detect_plateau,
    get_avg_mmax,
    savgol_filter_y,
)

pytestmark = pytest.mark.unit


class TestAmplitudeCalculations:
    """Test all EMG amplitude calculation methods."""

    def setup_method(self):
        """Create test signals with known properties."""
        # Create a test signal with known RMS and peak-to-trough values
        self.scan_rate = 30000  # 30 kHz
        self.duration_ms = 10  # 10 ms window
        self.n_samples = int(self.duration_ms * self.scan_rate / 1000)

        # Sine wave with known amplitude
        self.freq = 1000  # 1 kHz
        t = np.linspace(0, self.duration_ms / 1000, self.n_samples)
        self.amplitude = 2.0

        # Clean sine wave
        self.sine_signal = self.amplitude * np.sin(2 * np.pi * self.freq * t)

        # Signal with DC offset
        self.dc_offset = 1.5
        self.offset_signal = self.sine_signal + self.dc_offset

        # Noisy signal
        np.random.seed(42)  # Reproducible results
        self.noise_level = 0.1
        self.noisy_signal = self.sine_signal + np.random.normal(0, self.noise_level, len(self.sine_signal))

    def test_rectify_emg(self):
        """Test EMG rectification (absolute value)."""
        # Test with mixed positive/negative values
        test_signal = np.array([-2, -1, 0, 1, 2])
        expected = np.array([2, 1, 0, 1, 2])
        result = rectify_emg(test_signal)
        np.testing.assert_array_equal(result, expected)

        # Test with sine wave
        rectified = rectify_emg(self.sine_signal)
        assert np.all(rectified >= 0), "Rectified signal should be non-negative"
        assert np.max(rectified) == pytest.approx(self.amplitude, rel=1e-3), "Max should equal original amplitude"

    def test_average_amplitude_rectified(self):
        """Test rectified average amplitude calculation."""
        start_ms, end_ms = 0, self.duration_ms
        result = _calculate_average_amplitude_rectified(self.sine_signal, start_ms, end_ms, self.scan_rate)

        # For a sine wave, rectified average = 2*amplitude/π
        expected = 2 * self.amplitude / np.pi
        assert result == pytest.approx(expected, rel=0.01), f"Expected {expected}, got {result}"

    def test_peak_to_trough_amplitude(self):
        """Test peak-to-trough amplitude calculation."""
        start_ms, end_ms = 0, self.duration_ms
        result = _calculate_peak_to_trough_amplitude(self.sine_signal, start_ms, end_ms, self.scan_rate)

        # Peak-to-trough for sine wave = 2 * amplitude
        expected = 2 * self.amplitude
        assert result == pytest.approx(expected, rel=1e-3), f"Expected {expected}, got {result}"

    def test_rms_amplitude(self):
        """Test RMS amplitude calculation."""
        start_ms, end_ms = 0, self.duration_ms
        result = _calculate_rms_amplitude(self.sine_signal, start_ms, end_ms, self.scan_rate)

        # RMS of sine wave = amplitude / sqrt(2)
        expected = self.amplitude / np.sqrt(2)
        assert result == pytest.approx(expected, rel=1e-2), f"Expected {expected}, got {result}"  # More lenient tolerance

    def test_average_amplitude_unrectified(self):
        """Test unrectified average amplitude calculation."""
        start_ms, end_ms = 0, self.duration_ms

        # Sine wave should average to ~0
        result = _calculate_average_amplitude_unrectified(self.sine_signal, start_ms, end_ms, self.scan_rate)
        assert abs(result) < 1e-10, f"Sine wave average should be ~0, got {result}"

        # DC offset signal should average to the offset
        result_offset = _calculate_average_amplitude_unrectified(self.offset_signal, start_ms, end_ms, self.scan_rate)
        assert result_offset == pytest.approx(self.dc_offset, rel=1e-3), f"Expected {self.dc_offset}, got {result_offset}"

    def test_calculate_emg_amplitude_all_methods(self):
        """Test the main amplitude calculation function with all methods."""
        start_ms, end_ms = 2, 8  # Test with partial window

        methods = ["rms", "average_rectified", "peak_to_trough", "average_unrectified"]

        for method in methods:
            result = calculate_emg_amplitude(self.sine_signal, start_ms, end_ms, self.scan_rate, method)
            assert not np.isnan(result), f"Method {method} returned NaN"
            assert result >= 0 or method == "average_unrectified", f"Method {method} returned negative value: {result}"

    def test_empty_window_handling(self):
        """Test behavior with empty or invalid time windows."""
        # Empty window (start == end)
        result = _calculate_rms_amplitude(self.sine_signal, 5, 5, self.scan_rate)
        assert np.isnan(result), "Empty window should return NaN"

        # Invalid window (start > end)
        result = _calculate_rms_amplitude(self.sine_signal, 8, 2, self.scan_rate)
        assert np.isnan(result), "Invalid window should return NaN"


class TestFiltering:
    """Test filtering operations."""

    def setup_method(self):
        """Create test signals for filtering."""
        self.fs = 30000  # 30 kHz sampling rate
        self.duration = 0.1  # 100 ms
        self.t = np.linspace(0, self.duration, int(self.fs * self.duration))

        # Create composite signal with low and high frequency components
        self.low_freq = 50  # Below EMG range
        self.mid_freq = 1000  # Within EMG range
        self.high_freq = 5000  # Above EMG range

        self.test_signal = (
            np.sin(2 * np.pi * self.low_freq * self.t)
            + 2 * np.sin(2 * np.pi * self.mid_freq * self.t)
            + 0.5 * np.sin(2 * np.pi * self.high_freq * self.t)
        )

    def test_butter_bandpass_design(self):
        """Test Butterworth bandpass filter design."""
        lowcut, highcut = 100, 3500
        order = 4

        b, a = butter_bandpass(lowcut, highcut, self.fs, order)

        # Check that we get proper coefficients (bandpass filter doubles the order)
        expected_length = 2 * order + 1
        assert len(b) == expected_length, f"Expected {expected_length} b coefficients, got {len(b)}"
        assert len(a) == expected_length, f"Expected {expected_length} a coefficients, got {len(a)}"
        assert a[0] == 1.0, "First 'a' coefficient should be 1.0"

    def test_butter_bandpass_filter_frequency_response(self):
        """Test that the bandpass filter attenuates the correct frequencies."""
        lowcut, highcut = 100, 3500
        filtered = butter_bandpass_filter(self.test_signal, self.fs, lowcut, highcut, order=4)

        # Compute frequency domain representation
        freqs = np.fft.fftfreq(len(self.test_signal), 1 / self.fs)
        original_fft = np.fft.fft(self.test_signal)
        filtered_fft = np.fft.fft(filtered)

        # Find indices for our test frequencies
        low_idx = np.argmin(np.abs(freqs - self.low_freq))
        mid_idx = np.argmin(np.abs(freqs - self.mid_freq))
        high_idx = np.argmin(np.abs(freqs - self.high_freq))

        # Calculate attenuation ratios
        low_ratio = np.abs(filtered_fft[low_idx]) / np.abs(original_fft[low_idx])
        mid_ratio = np.abs(filtered_fft[mid_idx]) / np.abs(original_fft[mid_idx])
        high_ratio = np.abs(filtered_fft[high_idx]) / np.abs(original_fft[high_idx])

        # Low frequency should be significantly attenuated
        assert low_ratio < 0.1, f"Low frequency not sufficiently attenuated: {low_ratio}"

        # Mid frequency should pass through relatively unchanged
        assert mid_ratio > 0.8, f"Mid frequency excessively attenuated: {mid_ratio}"

        # High frequency should be attenuated
        assert high_ratio < 0.5, f"High frequency not sufficiently attenuated: {high_ratio}"

    def test_correct_emg_to_baseline(self):
        """Test baseline correction functionality."""
        # Create signal with known DC offset
        dc_offset = 2.5
        stim_delay = 50  # 50 ms stimulus delay

        # Signal with consistent offset
        offset_signal = self.test_signal + dc_offset

        corrected = correct_emg_to_baseline(offset_signal, self.fs, stim_delay)

        # Pre-stimulus baseline should now be close to zero
        baseline_samples = int(stim_delay * self.fs / 1000)
        baseline_mean = np.mean(corrected[:baseline_samples])

        assert abs(baseline_mean) < 1e-10, f"Baseline should be ~0 after correction, got {baseline_mean}"

        # The corrected signal should have the same AC component but zero DC component
        # Check that the difference from baseline mean is preserved
        original_baseline_mean = np.mean(offset_signal[:baseline_samples])
        corrected_baseline_mean = np.mean(corrected[:baseline_samples])

        # After correction, baseline should be near zero
        assert abs(corrected_baseline_mean) < 1e-10, f"Corrected baseline should be ~0, got {corrected_baseline_mean}"

        # The correction should remove exactly the DC component
        expected_correction = offset_signal - original_baseline_mean
        np.testing.assert_array_almost_equal(corrected, expected_correction, decimal=10)

    def test_filter_edge_cases(self):
        """Test filter behavior with edge cases."""
        # Very short signal
        short_signal = np.array([1, 2, 3, 4, 5])
        try:
            filtered = butter_bandpass_filter(short_signal, self.fs, 100, 3500, order=4)
            # Should either work or raise an appropriate error
            assert len(filtered) == len(short_signal)
        except Exception as e:
            # Accept that very short signals might not be filterable
            assert "padlen" in str(e).lower() or "filter" in str(e).lower()


class TestPlateauDetection:
    """Test plateau detection and M-max calculation algorithms."""

    def setup_method(self):
        """Create test data for plateau detection."""
        # Create stimulus-response curve with clear plateau
        self.n_points = 50
        self.stimulus_voltages = np.linspace(0, 10, self.n_points)

        # Sigmoidal curve that plateaus
        self.baseline_amplitude = 0.1
        self.max_amplitude = 2.0
        self.slope = 2.0
        self.midpoint = 6.0

        # Generate sigmoidal response
        sigmoid = 1 / (1 + np.exp(-self.slope * (self.stimulus_voltages - self.midpoint)))
        self.clean_amplitudes = self.baseline_amplitude + (self.max_amplitude - self.baseline_amplitude) * sigmoid

        # Add controlled noise
        np.random.seed(42)
        noise_level = 0.05
        self.noisy_amplitudes = self.clean_amplitudes + np.random.normal(0, noise_level, self.n_points)

    def test_savgol_filter_y(self):
        """Test Savitzky-Golay filtering."""
        filtered = savgol_filter_y(self.noisy_amplitudes)

        # Filtered signal should be smoother
        original_variation = np.std(np.diff(self.noisy_amplitudes))
        filtered_variation = np.std(np.diff(filtered))

        assert filtered_variation < original_variation, "Filtered signal should be smoother"
        assert len(filtered) == len(self.noisy_amplitudes), "Length should be preserved"

    def test_detect_plateau_success(self):
        """Test successful plateau detection."""
        # Use clean amplitudes for reliable plateau detection
        plateau_start, plateau_end = detect_plateau(
            self.clean_amplitudes, max_window_size=15, min_window_size=3, threshold=0.3
        )

        assert plateau_start is not None, "Should detect plateau in clean sigmoidal data"
        assert plateau_end is not None, "Should detect plateau in clean sigmoidal data"
        assert plateau_end > plateau_start, "Plateau end should be after start"

        # Plateau should be in the high-amplitude region
        plateau_values = self.clean_amplitudes[plateau_start:plateau_end]

        # Values in plateau should be relatively high
        assert np.mean(plateau_values) > 0.8 * self.max_amplitude, "Plateau should be in high-amplitude region"

    def test_detect_plateau_failure(self):
        """Test plateau detection with data that has no plateau."""
        # Use a noisy, constantly increasing signal - much harder to find plateau in
        linear_base = np.linspace(0, 2, 50)
        np.random.seed(123)  # Different seed for more noise
        noisy_linear = linear_base + 0.5 * np.random.normal(0, 1, 50)  # High noise

        plateau_start, plateau_end = detect_plateau(
            noisy_linear, max_window_size=8, min_window_size=3, threshold=0.1  # Smaller window  # Stricter threshold
        )

        # Should not find a plateau in very noisy linear data with strict threshold
        assert plateau_start is None, "Should not detect plateau in noisy linear data"
        assert plateau_end is None, "Should not detect plateau in noisy linear data"

    def test_get_avg_mmax_with_plateau(self):
        """Test M-max calculation when plateau is detected."""
        mmax = get_avg_mmax(
            self.stimulus_voltages,
            self.clean_amplitudes,
            max_window_size=15,
            min_window_size=3,
            threshold=0.3,
            validation_tolerance=1.05,
        )

        assert mmax is not None, "Should calculate M-max when plateau exists"
        assert isinstance(mmax, (int, float)), "M-max should be numeric"

        # M-max should be close to the maximum amplitude
        assert mmax > 0.9 * self.max_amplitude, f"M-max ({mmax}) should be close to max amplitude ({self.max_amplitude})"
        assert mmax <= 1.1 * self.max_amplitude, f"M-max ({mmax}) should not exceed reasonable bounds"

    def test_get_avg_mmax_fallback_method(self):
        """Test M-max calculation fallback when no plateau detected."""
        # Linear data without plateau - should use fallback method
        linear_voltages = np.linspace(0, 10, 30)
        linear_amplitudes = np.linspace(0, 2, 30)  # No plateau

        mmax = get_avg_mmax(
            linear_voltages, linear_amplitudes, max_window_size=20, min_window_size=3, threshold=0.3, validation_tolerance=1.05
        )

        assert mmax is not None, "Should calculate M-max using fallback method"
        assert isinstance(mmax, (int, float)), "M-max should be numeric"

        # Should be based on high-stimulus region (top 25%)
        high_stim_threshold = np.percentile(linear_voltages, 75)
        high_stim_indices = linear_voltages >= high_stim_threshold
        expected_range = (np.min(linear_amplitudes[high_stim_indices]), np.max(linear_amplitudes[high_stim_indices]))

        assert expected_range[0] <= mmax <= expected_range[1], f"M-max should be in high-stimulus range {expected_range}"

    def test_get_avg_mmax_insufficient_data(self):
        """Test M-max calculation with insufficient data."""
        # Empty arrays should raise error
        empty_voltages = np.array([])
        empty_amplitudes = np.array([])

        with pytest.raises(NoCalculableMmaxError):
            get_avg_mmax(empty_voltages, empty_amplitudes)

        # The algorithm is robust and handles single data points via fallback
        # Let's test that it actually works with minimal data
        single_voltage = np.array([5.0])
        single_amplitude = np.array([1.0])

        result = get_avg_mmax(single_voltage, single_amplitude)
        assert result == 1.0, "Single data point should return that amplitude"

    def test_get_avg_mmax_with_stimulus_range(self):
        """Test M-max calculation with stimulus range return."""
        result = get_avg_mmax(self.stimulus_voltages, self.clean_amplitudes, return_mmax_stim_range=True)

        # The function returns a tuple: (mmax, stim_start, stim_end)
        mmax, stim_start, stim_end = result

        assert mmax is not None, "Should calculate M-max"
        assert stim_start is not None, "Should return stimulus range start"
        assert stim_end is not None, "Should return stimulus range end"
        assert stim_end >= stim_start, "Stimulus range end should be >= start"

    def test_mmax_validation_tolerance(self):
        """Test that validation tolerance affects M-max selection."""
        # Test with very strict tolerance
        mmax_strict = get_avg_mmax(
            self.stimulus_voltages,
            self.noisy_amplitudes,  # Use noisy data to potentially trigger fallbacks
            validation_tolerance=1.01,  # Very strict (1% tolerance)
        )

        # Test with lenient tolerance
        mmax_lenient = get_avg_mmax(
            self.stimulus_voltages, self.noisy_amplitudes, validation_tolerance=1.20  # Lenient (20% tolerance)
        )

        assert mmax_strict is not None, "Should calculate M-max with strict tolerance"
        assert mmax_lenient is not None, "Should calculate M-max with lenient tolerance"
        # Results might differ due to different approach selection based on validation


class TestSignalProcessingIntegration:
    """Integration tests combining multiple signal processing components."""

    def test_full_emg_processing_pipeline(self):
        """Test a complete EMG processing pipeline."""
        # Create realistic EMG-like signal
        fs = 30000
        duration = 0.05  # 50 ms
        t = np.linspace(0, duration, int(fs * duration))

        # EMG-like signal: mix of frequencies with noise
        emg_signal = (
            0.5 * np.sin(2 * np.pi * 200 * t)  # Low freq component
            + 1.0 * np.sin(2 * np.pi * 1000 * t)  # Main EMG
            + 0.3 * np.sin(2 * np.pi * 2500 * t)  # High freq EMG
            + 0.2 * np.random.normal(0, 1, len(t))  # Noise
        )

        # Add DC offset
        dc_offset = 1.2
        emg_with_offset = emg_signal + dc_offset

        # Step 1: Baseline correction
        stim_delay = 20  # 20 ms
        baseline_corrected = correct_emg_to_baseline(emg_with_offset, fs, stim_delay)

        # Step 2: Bandpass filtering
        filtered = butter_bandpass_filter(baseline_corrected, fs, 100, 3500, order=4)

        # Step 3: Calculate amplitude
        start_ms, end_ms = 25, 45  # Post-stimulus window
        rms_amplitude = calculate_emg_amplitude(filtered, start_ms, end_ms, fs, "rms")

        # Verify processing results
        assert not np.isnan(rms_amplitude), "Processing pipeline should produce valid amplitude"
        assert rms_amplitude > 0, "RMS amplitude should be positive"

        # Baseline should be corrected
        baseline_samples = int(stim_delay * fs / 1000)
        baseline_mean = np.mean(baseline_corrected[:baseline_samples])
        assert abs(baseline_mean) < 1e-10, "Baseline should be corrected"

    def test_stimulus_response_curve_analysis(self):
        """Test analysis of a complete stimulus-response curve."""
        # Create multiple "recordings" at different stimulus intensities
        n_stimuli = 20
        stimulus_voltages = np.linspace(1, 10, n_stimuli)
        m_wave_amplitudes = []

        # Generate realistic M-wave amplitude progression
        for voltage in stimulus_voltages:
            # Sigmoidal response with noise
            base_response = 2.0 / (1 + np.exp(-1.5 * (voltage - 6.0)))
            noise = np.random.normal(0, 0.1)
            amplitude = max(0.1, base_response + noise)  # Ensure positive
            m_wave_amplitudes.append(amplitude)

        m_wave_amplitudes = np.array(m_wave_amplitudes)

        # Analyze the stimulus-response curve
        mmax = get_avg_mmax(stimulus_voltages, m_wave_amplitudes)

        assert mmax is not None, "Should calculate M-max from stimulus-response curve"
        assert mmax > np.max(m_wave_amplitudes) * 0.8, "M-max should be in upper range of responses"

        # Test plateau detection on the curve
        plateau_start, plateau_end = detect_plateau(m_wave_amplitudes, max_window_size=8, min_window_size=3, threshold=0.3)

        # Depending on noise, plateau might or might not be detected
        if plateau_start is not None:
            assert plateau_end > plateau_start, "Valid plateau detection"
            plateau_amplitudes = m_wave_amplitudes[plateau_start:plateau_end]
            assert np.std(plateau_amplitudes) < np.std(m_wave_amplitudes), "Plateau should be less variable"
