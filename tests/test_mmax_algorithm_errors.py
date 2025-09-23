"""Tests for M-max algorithm and plateau detection error handling."""

import numpy as np
import pytest

from monstim_signals.transform.plateau import (
    NoCalculableMmaxError,
    get_avg_mmax,
)

pytestmark = pytest.mark.unit


class TestMmaxAlgorithmErrorHandling:
    """Test error handling in M-max plateau detection algorithm."""

    def test_no_calculable_mmax_error_creation(self):
        """Test that NoCalculableMmaxError can be created with custom message."""
        error = NoCalculableMmaxError()
        assert "No calculable M-max" in str(error)

        custom_msg = "Custom error message"
        error_custom = NoCalculableMmaxError(custom_msg)
        assert str(error_custom) == custom_msg

    def test_get_avg_mmax_with_invalid_data(self):
        """Test that get_avg_mmax handles invalid data gracefully."""
        # Test with empty data
        empty_stim = np.array([])
        empty_resp = np.array([])
        with pytest.raises((NoCalculableMmaxError, ValueError, IndexError)):
            get_avg_mmax(empty_stim, empty_resp)

        # Test with all zeros (should not produce valid plateau)
        zero_stim = np.linspace(0, 10, 100)
        zero_resp = np.zeros(100)
        try:
            result = get_avg_mmax(zero_stim, zero_resp)
            # If it doesn't raise an error, it should be a reasonable value
            assert isinstance(result, (int, float))
            assert result >= 0
        except (NoCalculableMmaxError, ValueError):
            # This is acceptable - all zeros may not produce a valid M-max
            pass

        # Test with insufficient data points
        small_stim = np.array([1.0, 2.0])
        small_resp = np.array([10.0, 20.0])
        try:
            result = get_avg_mmax(small_stim, small_resp)
            # The algorithm may succeed using fallback methods
            assert isinstance(result, (int, float))
            assert result > 0
        except (NoCalculableMmaxError, ValueError, IndexError):
            # It's also acceptable to fail with insufficient data
            pass

    def test_get_avg_mmax_with_noisy_data(self):
        """Test M-max detection with highly noisy data that might not have clear plateaus."""
        # Generate highly random/noisy data
        np.random.seed(42)  # For reproducible tests
        noisy_stim = np.linspace(0, 10, 100)
        noisy_resp = np.random.random(100) * 1000  # Very noisy data

        try:
            result = get_avg_mmax(noisy_stim, noisy_resp)
            # If it succeeds, result should be a reasonable number
            assert isinstance(result, (int, float))
            assert result > 0
        except NoCalculableMmaxError:
            # This is acceptable - the algorithm correctly identified
            # that no reliable M-max could be calculated
            pass

    def test_get_avg_mmax_with_ascending_data(self):
        """Test M-max detection with strictly ascending data (no plateau)."""
        # Create strictly ascending data (no plateau should exist)
        ascending_stim = np.linspace(0, 10, 100)
        ascending_resp = np.linspace(0, 1000, 100)

        try:
            result = get_avg_mmax(ascending_stim, ascending_resp)
            # If it finds something, it should be reasonable
            assert isinstance(result, (int, float))
            assert result > 0
        except NoCalculableMmaxError:
            # This is acceptable - no clear plateau in ascending data
            pass

    def test_get_avg_mmax_with_valid_plateau_data(self):
        """Test M-max detection with data that should have a clear plateau."""
        # Create stimulus and response data with a clear plateau
        stim_vals = np.linspace(0, 10, 100)
        responses = np.concatenate(
            [
                np.linspace(0, 100, 20),  # Rising phase
                np.full(30, 100),  # Plateau at 100
                np.linspace(100, 120, 20),  # Continue rising
                np.full(30, 120),  # Higher plateau at 120
            ]
        )

        # This should succeed and find a reasonable M-max
        result = get_avg_mmax(stim_vals, responses)
        assert isinstance(result, (int, float))
        assert result > 0
        # Should be close to one of our plateau values
        assert 90 <= result <= 130  # Allow some tolerance

    def test_get_avg_mmax_with_mismatched_arrays(self):
        """Test that get_avg_mmax handles mismatched array lengths."""
        # Test with mismatched arrays
        stim_vals = np.array([1, 2, 3])
        responses = np.array([10, 20])  # Different length

        with pytest.raises((ValueError, IndexError)):
            get_avg_mmax(stim_vals, responses)

    def test_plateau_detection_with_edge_cases(self):
        """Test plateau detection with various edge cases."""
        # Single point
        single_val_stim = np.array([5.0])
        single_val_resp = np.array([100.0])

        try:
            result = get_avg_mmax(single_val_stim, single_val_resp)
            assert isinstance(result, (int, float))
            assert result > 0
        except (NoCalculableMmaxError, ValueError, IndexError):
            # Single point may not be sufficient for plateau detection
            pass

        # Two identical points
        two_point_stim = np.array([5.0, 5.0])
        two_point_resp = np.array([100.0, 100.0])

        try:
            result = get_avg_mmax(two_point_stim, two_point_resp)
            assert isinstance(result, (int, float))
            assert result > 0
        except (NoCalculableMmaxError, ValueError, IndexError):
            # Two points may not be sufficient for robust plateau detection
            pass

    def test_mmax_algorithm_robustness(self):
        """Test that M-max algorithm is robust to different data patterns."""
        test_cases = [
            # Monotonically increasing
            (np.linspace(0, 10, 50), np.linspace(0, 100, 50)),
            # Step function
            (np.linspace(0, 10, 50), np.concatenate([np.zeros(25), np.full(25, 100)])),
            # Noisy plateau
            (np.linspace(0, 10, 50), np.concatenate([np.linspace(0, 80, 25), 80 + np.random.normal(0, 5, 25)])),
        ]

        for stim_vals, responses in test_cases:
            try:
                result = get_avg_mmax(stim_vals, responses)
                # If it succeeds, should return a number
                assert isinstance(result, (int, float))
                assert result > 0

            except (NoCalculableMmaxError, ValueError, IndexError):
                # It's acceptable for the algorithm to fail on difficult cases
                # The important thing is that it fails gracefully
                pass
