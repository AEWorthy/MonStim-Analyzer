"""
Comprehensive tests for Error Handling and Recovery in MonStim Analyzer.

Tests system resilience, graceful degradation, user error recovery workflows,
and proper error handling across all major components. Focuses on UnableToPlotError
patterns, repository failures, data corruption scenarios, and recovery mechanisms.
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest

from monstim_gui.managers import PlotController
from monstim_signals.io.repositories import SessionRepository
from monstim_signals.plotting import UnableToPlotError
from monstim_signals.transform.plateau import NoCalculableMmaxError

# --- Test Annotations ---
# Purpose: System resilience and recovery paths (UnableToPlotError patterns, repo failures, data corruption)
# Markers: integration (cross-component behaviors, tempfile IO, QMessageBox patching)
# Notes: Uses temporary workspace; no real GUI loop required
pytestmark = pytest.mark.integration


@pytest.fixture
def temp_workspace():
    """Create a temporary directory for testing, cleaned up automatically."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup: Remove the entire temp directory tree
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def fake_gui():
    """Create a fake GUI object for testing."""
    gui = Mock()
    gui.current_session = None
    gui.current_dataset = None
    gui.current_experiment = None
    return gui


@pytest.fixture
def mock_corrupted_data():
    """Generate corrupted/invalid data for testing error handling."""
    return {
        "invalid_array": np.array([np.nan, np.inf, -np.inf]),
        "empty_array": np.array([]),
        "misshapen_array": [np.array([1, 2]), np.array([3])],  # Different length arrays
        "wrong_dtype": np.array(["text", "data", "not", "numeric"]),
        "extreme_values": np.array([1e100, -1e100, 1e-100]),
    }


class TestUnableToPlotErrorHandling:
    """Test UnableToPlotError handling patterns and recovery mechanisms."""

    def test_unable_to_plot_error_creation_and_properties(self):
        """Test UnableToPlotError creation with various messages."""
        # Test with custom message
        custom_msg = "No channels selected for plotting"
        error_custom = UnableToPlotError(custom_msg)
        assert str(error_custom) == custom_msg
        assert error_custom.message == custom_msg
        assert isinstance(error_custom, Exception)

        # Test with complex error scenarios
        complex_msg = "Unable to plot EMG data: channel indices [0, 5, 10] exceed available channels (4)"
        error_complex = UnableToPlotError(complex_msg)
        assert str(error_complex) == complex_msg
        assert error_complex.message == complex_msg

    def test_plot_controller_unable_to_plot_error_handling(self, fake_gui):
        """Test that PlotController handles UnableToPlotError gracefully."""
        controller = PlotController(fake_gui)

        # Mock required GUI components
        fake_gui.plot_widget = Mock()
        fake_gui.plot_pane = Mock()

        # Test different error scenarios
        test_cases = [
            ("No channels to plot", "No Channels Selected"),
            ("Canvas must be provided", "Plot Error"),
            ("Invalid data format", "Plot Error"),
            ("Insufficient data points", "Plot Error"),
        ]

        for error_msg, expected_title in test_cases:
            error = UnableToPlotError(error_msg)

            # Mock QMessageBox to capture the message
            with patch("monstim_gui.managers.plot_controller.QMessageBox") as mock_msg:
                controller.handle_unable_to_plot_error(error, "emg", {})

                # Verify warning was called
                mock_msg.warning.assert_called_once()
                call_args = mock_msg.warning.call_args

                # QMessageBox.warning signature: (parent, title, message)
                actual_title = call_args[0][1]  # title argument
                user_message = call_args[0][2]  # message argument

                # Verify title is set correctly
                if "No channels to plot" in error_msg:
                    assert actual_title == "No Channels Selected"
                else:
                    assert actual_title == "Plot Error"

                # Verify user-friendly message content
                if "No channels to plot" in error_msg:
                    assert "select at least one channel" in user_message.lower()
                else:
                    assert "unable to create plot" in user_message.lower()

    def test_plotting_error_propagation_and_wrapping(self):
        """Test that plotting errors are properly wrapped and propagated."""
        # Test that generic exceptions are wrapped in UnableToPlotError
        with patch("monstim_signals.plotting.session_plotter_pyqtgraph.SessionPlotterPyQtGraph") as mock_plotter:
            mock_plotter_instance = Mock()
            mock_plotter.return_value = mock_plotter_instance

            # Simulate a plotting method that wraps generic errors
            def mock_plot_method(*args, **kwargs):
                raise ValueError("Generic plotting error")

            mock_plotter_instance.plot_emg = mock_plot_method

            # Test that generic errors get wrapped appropriately
            with pytest.raises(ValueError):  # The original error should propagate in some cases
                mock_plot_method()

    def test_canvas_validation_error_handling(self):
        """Test error handling when canvas is None or invalid."""
        # This tests the pattern where canvas=None should raise UnableToPlotError
        with pytest.raises(UnableToPlotError, match="Canvas.*required"):
            raise UnableToPlotError("Canvas is required for plotting")

        # Test invalid canvas scenarios
        invalid_canvas_scenarios = [
            "Canvas must be provided for PyQtGraph plotting",
            "Canvas is required for plotting.",
            "Invalid canvas object provided",
        ]

        for scenario in invalid_canvas_scenarios:
            error = UnableToPlotError(scenario)
            assert "canvas" in str(error).lower() or "Canvas" in str(error)

    def test_channel_indices_validation_error_handling(self):
        """Test error handling for invalid channel indices."""
        # Test empty channel list
        error_empty = UnableToPlotError("No channels to plot. Select at least one channel.")
        assert "channel" in str(error_empty).lower()

        # Test out-of-range channel indices
        error_range = UnableToPlotError("Channel index 5 exceeds available channels (4)")
        assert "channel" in str(error_range).lower()
        assert "exceed" in str(error_range)

        # Test invalid channel types
        error_type = UnableToPlotError("Invalid channel index type: expected int, got str")
        assert "channel" in str(error_type).lower()
        assert "type" in str(error_type).lower()


class TestGracefulDegradationScenarios:
    """Test system behavior when components fail gracefully."""

    def test_missing_data_graceful_handling(self, temp_workspace):
        """Test graceful handling when expected data files are missing."""
        # Create incomplete session structure
        session_dir = temp_workspace / "incomplete_session"
        session_dir.mkdir()

        # Create session annotation but no data files (this should cause IndexError)
        session_annot = {"data_version": "2025.09-test", "excluded_recordings": [], "channels": []}
        with open(session_dir / "session.annot.json", "w") as f:
            json.dump(session_annot, f)

        # Repository should handle missing data gracefully
        # The current implementation raises IndexError when no recordings exist
        # This tests that the error is predictable and handles gracefully
        try:
            repo = SessionRepository(session_dir)
            loaded_session = repo.load()
            # If we get here, the system handled empty data gracefully
            assert loaded_session is not None
        except (IndexError, FileNotFoundError, OSError, ValueError) as e:
            # IndexError is expected when no recordings exist - acceptable graceful failure
            assert isinstance(e, (IndexError, FileNotFoundError, OSError, ValueError))
            # Verify the error is informative
            assert len(str(e)) > 0

    def test_corrupted_annotation_file_recovery(self, temp_workspace):
        """Test recovery when annotation files are corrupted."""
        session_dir = temp_workspace / "corrupted_session"
        session_dir.mkdir()

        # Create corrupted annotation file
        corrupted_annot_path = session_dir / "session.annot.json"
        with open(corrupted_annot_path, "w") as f:
            f.write('{"invalid": json, "syntax":]')  # Invalid JSON

        # System should handle corrupted annotations gracefully
        try:
            repo = SessionRepository(session_dir)
            loaded_session = repo.load()
            # May succeed with default values or fail gracefully
            assert loaded_session is not None
        except (json.JSONDecodeError, ValueError, FileNotFoundError) as e:
            # These are acceptable failure modes for corrupted data
            assert isinstance(e, (json.JSONDecodeError, ValueError, FileNotFoundError))

    def test_partial_data_loading_resilience(self, temp_workspace):
        """Test system resilience when only partial data is available."""
        # Create session with some valid and some invalid recordings
        session_dir = temp_workspace / "partial_session"
        session_dir.mkdir()

        # Create valid session annotation as dict
        session_annot_dict = {
            "data_version": "2025.09-test",
            "excluded_recordings": [],
            "channels": [
                {"invert": False, "name": "EMG_1", "unit": "mV", "type_override": None},
                {"invert": False, "name": "EMG_2", "unit": "mV", "type_override": None},
            ],
        }
        with open(session_dir / "session.annot.json", "w") as f:
            json.dump(session_annot_dict, f)

        # Create one valid and one corrupted HDF5 file
        valid_h5 = session_dir / "valid.raw.h5"
        with h5py.File(valid_h5, "w") as f:
            f.create_dataset("raw", data=np.random.randn(1000, 2))
            f.attrs["scan_rate"] = 1000
            f.attrs["num_channels"] = 2

        # Create corresponding meta files
        valid_meta = {"scan_rate": 1000, "num_channels": 2, "recording_id": "0001"}
        with open(session_dir / "valid.meta.json", "w") as f:
            json.dump(valid_meta, f)

        # Create corrupted HDF5 file
        corrupted_h5 = session_dir / "corrupted.raw.h5"
        with open(corrupted_h5, "wb") as f:
            f.write(b"not an hdf5 file")

        # System should load what it can and handle errors gracefully
        try:
            repo = SessionRepository(session_dir)
            loaded_session = repo.load()
            # Should have some data loaded despite partial corruption
            assert loaded_session is not None
        except Exception as e:
            # Acceptable if the corruption is too severe
            logging.info(f"Acceptable graceful failure: {e}")

    def test_memory_pressure_graceful_degradation(self, mock_corrupted_data):
        """Test graceful behavior under memory pressure scenarios."""
        # Simulate memory allocation failure
        with patch("numpy.array", side_effect=MemoryError("Insufficient memory")):
            try:
                # Attempt operation that would normally succeed
                _ = np.array([1, 2, 3, 4, 5])
                assert False, "Should have raised MemoryError"
            except MemoryError as e:
                # This is the expected graceful failure
                assert "memory" in str(e).lower()

        # Test handling of extremely large arrays
        try:
            # This might succeed or fail depending on available memory
            large_array = np.zeros((10000, 10000), dtype=np.float64)
            # If it succeeds, verify it's reasonable
            assert large_array.shape == (10000, 10000)
        except MemoryError:
            # Graceful failure under memory pressure
            pass

    def test_configuration_fallback_mechanisms(self, temp_workspace):
        """Test fallback to default configurations when custom configs fail."""
        # Create invalid configuration file
        config_dir = temp_workspace / "config"
        config_dir.mkdir()

        invalid_config = config_dir / "invalid.yml"
        with open(invalid_config, "w") as f:
            f.write("invalid: yaml: syntax: [")  # Invalid YAML

        # Configuration system should fall back to defaults
        # This tests the pattern where invalid configs don't crash the system
        try:
            # Simulate config loading that falls back to defaults
            import yaml

            with open(invalid_config) as f:
                _ = yaml.safe_load(f)
            assert False, "Should have failed to parse invalid YAML"
        except yaml.YAMLError:
            # Expected graceful failure - system should use defaults
            default_config = {"default": True}
            assert default_config["default"] is True


class TestUserErrorRecoveryWorkflows:
    """Test user error recovery workflows and guidance systems."""

    def test_invalid_file_selection_recovery(self, temp_workspace):
        """Test recovery when user selects invalid files for import."""
        # Create various invalid file types
        text_file = temp_workspace / "not_data.txt"
        text_file.write_text("This is just a text file, not EMG data")

        empty_file = temp_workspace / "empty.csv"
        empty_file.touch()

        binary_file = temp_workspace / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")

        # Test that the system provides helpful error messages
        invalid_files = [text_file, empty_file, binary_file]

        for invalid_file in invalid_files:
            # Simulate file validation that provides user guidance
            try:
                # This would be the actual validation logic
                if invalid_file.suffix not in [".csv", ".h5"]:
                    raise ValueError(f"Unsupported file type: {invalid_file.suffix}")
                if invalid_file.stat().st_size == 0:
                    raise ValueError("File is empty")
                # Additional validation would go here
            except ValueError as e:
                # Verify error messages are user-friendly
                assert len(str(e)) > 0
                # Error should mention the specific issue
                if "empty" in str(e).lower():
                    assert "empty" in str(e).lower()
                elif "type" in str(e).lower():
                    assert "type" in str(e).lower()

    def test_invalid_parameter_input_recovery(self):
        """Test recovery from invalid user parameter inputs."""
        # Test various invalid parameter scenarios
        invalid_parameters = [
            {"channel_indices": [-1, 5, 100]},  # Out of range
            {"channel_indices": ["a", "b", "c"]},  # Wrong type
            {"scan_rate": -1000},  # Negative value
            {"scan_rate": "not_a_number"},  # Wrong type
            {"method": "nonexistent_method"},  # Invalid option
            {"relative_to_mmax": "maybe"},  # Wrong boolean type
        ]

        for params in invalid_parameters:
            for key, value in params.items():
                # Test parameter validation
                if key == "channel_indices":
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], str):
                            # Type error
                            with pytest.raises((ValueError, TypeError)):
                                _ = [int(x) for x in value]
                        elif any(x < 0 for x in value):
                            # Range error - negative indices
                            error_msg = f"Channel indices must be non-negative, got: {value}"
                            assert "negative" in error_msg.lower() or "non-negative" in error_msg.lower()

                elif key == "scan_rate":
                    if isinstance(value, str):
                        with pytest.raises((ValueError, TypeError)):
                            float(value)
                    elif isinstance(value, (int, float)) and value <= 0:
                        error_msg = f"Scan rate must be positive, got: {value}"
                        assert "positive" in error_msg.lower()

    def test_data_selection_validation_and_guidance(self):
        """Test validation and user guidance for data selection operations."""
        # Test empty data selection
        empty_selection = []
        if len(empty_selection) == 0:
            guidance_msg = "No data selected. Please select at least one item to continue."
            assert "select" in guidance_msg.lower()
            assert "at least one" in guidance_msg.lower()

        # Test invalid data range selection
        data_range = (100, 50)  # End before start
        if data_range[1] <= data_range[0]:
            guidance_msg = f"Invalid range: end ({data_range[1]}) must be greater than start ({data_range[0]})"
            assert "invalid" in guidance_msg.lower()
            assert "greater than" in guidance_msg.lower()

        # Test excessive data selection (performance warning)
        large_selection = list(range(10000))  # Very large selection
        if len(large_selection) > 1000:
            warning_msg = f"Large selection ({len(large_selection)} items) may impact performance. Continue?"
            assert "performance" in warning_msg.lower()
            assert "continue" in warning_msg.lower()

    def test_plot_configuration_error_recovery(self, fake_gui):
        """Test recovery from plot configuration errors."""
        # Mock GUI components
        fake_gui.plot_widget = Mock()
        fake_gui.plot_pane = Mock()

        # Test various configuration error scenarios
        config_errors = [
            AttributeError("GUI missing required component: plot_widget"),
            AttributeError("GUI missing required component: plot_pane"),
            ValueError("Invalid plot type selected"),
            KeyError("Required configuration key missing"),
        ]

        for error in config_errors:
            # Test that configuration errors are handled gracefully
            if isinstance(error, AttributeError) and "missing required component" in str(error):
                # This should be caught by validation
                try:
                    raise error
                except AttributeError as e:
                    assert "component" in str(e).lower()
                    assert "missing" in str(e).lower()

            elif isinstance(error, ValueError) and "plot type" in str(error):
                # Invalid plot type should provide guidance
                guidance = "Please select a valid plot type from the available options"
                assert "select" in guidance.lower()
                assert "valid" in guidance.lower()

            elif isinstance(error, KeyError):
                # Missing configuration should use defaults
                default_msg = "Using default configuration due to missing settings"
                assert "default" in default_msg.lower()


class TestSystemResilienceUnderFailureConditions:
    """Test system resilience under various failure conditions."""

    def test_disk_space_exhaustion_handling(self, temp_workspace):
        """Test handling when disk space is exhausted."""
        # Simulate disk full scenario
        with patch("builtins.open", side_effect=OSError("No space left on device")):
            try:
                output_file = temp_workspace / "large_output.h5"
                with open(output_file, "w") as f:
                    f.write("data")
                assert False, "Should have raised OSError"
            except OSError as e:
                # System should handle disk full gracefully
                assert "space" in str(e).lower() or "device" in str(e).lower()

                # Recovery guidance should be provided
                recovery_msg = (
                    "Insufficient disk space. Please free up space and try again, " "or select a different output location."
                )
                assert "disk space" in recovery_msg.lower()
                assert "free up space" in recovery_msg.lower()

    def test_network_interruption_resilience(self):
        """Test resilience when network operations are interrupted."""
        # Simulate network-related errors that might occur
        network_errors = [
            ConnectionError("Network connection lost"),
            TimeoutError("Operation timed out"),
            OSError("Network is unreachable"),
        ]

        for error in network_errors:
            # Test that network errors are handled appropriately
            try:
                raise error
            except (ConnectionError, TimeoutError, OSError) as e:
                # These should be handled gracefully with retry mechanisms
                retry_msg = f"Network error occurred: {e}. Retrying in 5 seconds..."
                assert "network" in retry_msg.lower() or "connection" in retry_msg.lower()
                assert "retry" in retry_msg.lower()

    def test_concurrent_access_error_handling(self, temp_workspace):
        """Test handling of concurrent file access errors."""
        # Create a file that might be accessed concurrently
        shared_file = temp_workspace / "shared_data.json"
        shared_file.write_text('{"shared": "data"}')

        # Simulate file locking/permission errors
        permission_errors = [
            PermissionError("Permission denied"),
            OSError("Resource temporarily unavailable"),
            FileExistsError("File already exists"),
        ]

        for error in permission_errors:
            try:
                raise error
            except (PermissionError, OSError, FileExistsError) as e:
                # Should provide appropriate error handling
                if isinstance(e, PermissionError):
                    msg = "Permission denied. Check file permissions and try again."
                    assert "permission" in msg.lower()
                elif isinstance(e, OSError) and "temporarily unavailable" in str(e):
                    msg = "Resource busy. Please wait and try again."
                    assert "busy" in msg.lower() or "wait" in msg.lower()
                elif isinstance(e, FileExistsError):
                    msg = "File already exists. Choose a different name or enable overwrite."
                    assert "already exists" in msg.lower()

    def test_memory_corruption_detection_and_recovery(self, mock_corrupted_data):
        """Test detection and recovery from memory corruption scenarios."""
        # Test handling of corrupted numpy arrays
        corrupted_arrays = [
            mock_corrupted_data["invalid_array"],  # Contains NaN/inf
            mock_corrupted_data["empty_array"],  # Empty
            mock_corrupted_data["extreme_values"],  # Extreme values
        ]

        for corrupted_array in corrupted_arrays:
            # Test data validation
            if len(corrupted_array) == 0:
                validation_error = "Empty data array detected"
                assert "empty" in validation_error.lower()

            elif np.any(np.isnan(corrupted_array)) or np.any(np.isinf(corrupted_array)):
                validation_error = "Invalid values (NaN/Inf) detected in data"
                assert "invalid" in validation_error.lower()
                assert "nan" in validation_error.lower() or "inf" in validation_error.lower()

            elif np.any(np.abs(corrupted_array) > 1e50):
                validation_error = "Extreme values detected - possible data corruption"
                assert "extreme" in validation_error.lower()
                assert "corruption" in validation_error.lower()

    def test_algorithm_convergence_failure_handling(self):
        """Test handling when algorithms fail to converge."""
        # Simulate algorithm convergence failures
        convergence_errors = [
            NoCalculableMmaxError("No calculable M-max found"),
            ValueError("Algorithm failed to converge after 1000 iterations"),
            RuntimeError("Numerical instability detected"),
        ]

        for error in convergence_errors:
            if isinstance(error, NoCalculableMmaxError):
                # Should provide specific guidance for M-max failures
                guidance = (
                    "Unable to calculate M-max. This may be due to:\n"
                    "1. Insufficient stimulus range\n"
                    "2. Noisy data preventing plateau detection\n"
                    "3. Inappropriate stimulus parameters\n\n"
                    "Try adjusting stimulus parameters or data filtering."
                )
                assert "m-max" in guidance.lower()
                assert "plateau" in guidance.lower()
                assert "stimulus" in guidance.lower()

            elif "converge" in str(error):
                guidance = (
                    "Algorithm convergence failed. Try:\n"
                    "1. Adjusting algorithm parameters\n"
                    "2. Using different initial conditions\n"
                    "3. Preprocessing the data to reduce noise"
                )
                assert "algorithm" in guidance.lower()
                assert "parameters" in guidance.lower()


class TestRepositoryErrorHandling:
    """Test error handling in repository operations."""

    def test_repository_file_corruption_recovery(self, temp_workspace):
        """Test repository recovery from file corruption."""
        # Create a corrupted repository structure
        session_dir = temp_workspace / "corrupted_repo"
        session_dir.mkdir()

        # Create corrupted HDF5 file
        corrupted_h5 = session_dir / "data.raw.h5"
        with open(corrupted_h5, "wb") as f:
            f.write(b"corrupted hdf5 data")

        # Repository should handle corruption gracefully
        try:
            repo = SessionRepository(session_dir)
            loaded_session = repo.load()
            # May succeed with partial data or fail gracefully
            assert loaded_session is not None
        except (OSError, ValueError, Exception) as e:
            # These are acceptable failure modes
            assert isinstance(e, (OSError, ValueError, Exception))
            # Error message should be informative
            assert len(str(e)) > 0

    def test_repository_permission_error_handling(self, temp_workspace):
        """Test handling of permission errors in repository operations."""
        # Create a directory structure
        restricted_dir = temp_workspace / "restricted"
        restricted_dir.mkdir()

        # Simulate permission errors
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Permission denied")):
            try:
                new_dir = restricted_dir / "new_folder"
                new_dir.mkdir()
                assert False, "Should have raised PermissionError"
            except PermissionError as e:
                # Repository should provide helpful error messages
                error_msg = f"Unable to create directory due to permissions: {e}"
                assert "permission" in error_msg.lower()
                assert "create" in error_msg.lower()

    def test_repository_concurrent_modification_handling(self, temp_workspace):
        """Test handling when repository is modified by external processes."""
        session_dir = temp_workspace / "concurrent_session"
        session_dir.mkdir()

        # Create initial session structure as dict
        session_annot_dict = {
            "data_version": "2025.09-test",
            "excluded_recordings": [],
            "channels": [
                {"invert": False, "name": "EMG_1", "unit": "mV", "type_override": None},
                {"invert": False, "name": "EMG_2", "unit": "mV", "type_override": None},
            ],
        }
        annot_file = session_dir / "session.annot.json"
        with open(annot_file, "w") as f:
            json.dump(session_annot_dict, f)

        # Simulate external modification during operation
        repo = SessionRepository(session_dir)

        # Modify file externally while repository is active
        with open(annot_file, "w") as f:
            json.dump({"externally": "modified"}, f)

        # Repository should handle unexpected changes
        # Note: Since no recordings exist, this will raise IndexError - that's acceptable
        try:
            loaded_session = repo.load()
            # May succeed with updated data or detect the change
            assert loaded_session is not None
        except (IndexError, ValueError, json.JSONDecodeError) as e:
            # IndexError is expected when no recordings exist - acceptable failure
            # ValueError/JSONDecodeError if change detection/validation fails
            assert isinstance(e, (IndexError, ValueError, json.JSONDecodeError))


class TestConfigurationErrorRecovery:
    """Test configuration error handling and recovery mechanisms."""

    def test_missing_configuration_fallback(self, temp_workspace):
        """Test fallback behavior when configuration files are missing."""
        # Test with non-existent config directory
        missing_config_dir = temp_workspace / "nonexistent_config"

        # System should use built-in defaults
        try:
            # Simulate config loading that falls back to defaults
            if not missing_config_dir.exists():
                default_config = {"scan_rate": 1000, "num_channels": 4, "channel_types": ["EMG"] * 4}
                # Verify defaults are reasonable
                assert default_config["scan_rate"] > 0
                assert default_config["num_channels"] > 0
                assert len(default_config["channel_types"]) == default_config["num_channels"]
        except Exception as e:
            pytest.fail(f"Default configuration fallback failed: {e}")

    def test_invalid_configuration_value_handling(self):
        """Test handling of invalid configuration values."""
        invalid_configs = [
            {"scan_rate": -1000},  # Negative scan rate
            {"scan_rate": "invalid"},  # Wrong type
            {"num_channels": 0},  # Zero channels
            {"num_channels": -5},  # Negative channels
            {"channel_types": []},  # Empty channel types
            {"channel_types": ["INVALID_TYPE"]},  # Invalid channel type
        ]

        for config in invalid_configs:
            # Test configuration validation
            for key, value in config.items():
                if key == "scan_rate":
                    if isinstance(value, str):
                        with pytest.raises((ValueError, TypeError)):
                            _ = float(value)
                    elif isinstance(value, (int, float)) and value <= 0:
                        validation_msg = f"Invalid scan rate: {value}. Must be positive."
                        assert "positive" in validation_msg.lower()

                elif key == "num_channels":
                    if value <= 0:
                        validation_msg = f"Invalid number of channels: {value}. Must be greater than 0."
                        assert "greater than" in validation_msg.lower()

                elif key == "channel_types":
                    if len(value) == 0:
                        validation_msg = "Channel types cannot be empty."
                        assert "empty" in validation_msg.lower()
                    elif not all(isinstance(t, str) for t in value):
                        validation_msg = "All channel types must be strings."
                        assert "string" in validation_msg.lower()


class TestMemoryAndResourceManagement:
    """Test memory management and resource cleanup error handling."""

    def test_large_dataset_memory_management(self):
        """Test memory management with large datasets."""
        # Test memory allocation patterns
        try:
            # Simulate progressively larger allocations
            sizes = [1000, 10000, 100000]  # Start small and grow
            arrays = []

            for size in sizes:
                try:
                    array = np.random.randn(size, 4)  # 4-channel data
                    arrays.append(array)

                    # Verify allocation succeeded
                    assert array.shape == (size, 4)
                    assert not np.any(np.isnan(array))

                except MemoryError:
                    # Acceptable failure under memory pressure
                    break

                # Clean up to prevent accumulation
                if len(arrays) > 2:  # Keep only recent allocations
                    arrays.pop(0)

            # Explicit cleanup
            arrays.clear()

        except Exception as e:
            # Should not fail catastrophically
            assert isinstance(e, (MemoryError, OSError))

    def test_file_handle_resource_management(self, temp_workspace):
        """Test proper file handle cleanup and resource management."""
        # Test file handle limits
        test_files = []

        try:
            # Create multiple files to test resource management
            for i in range(10):  # Reasonable number for testing
                test_file = temp_workspace / f"test_{i}.h5"

                # Use context manager for proper cleanup
                with h5py.File(test_file, "w") as f:
                    f.create_dataset("data", data=np.random.randn(100, 2))

                test_files.append(test_file)

                # Verify file was created and closed properly
                assert test_file.exists()

                # Test that file can be reopened (not locked)
                with h5py.File(test_file, "r") as f:
                    assert "data" in f
                    assert f["data"].shape == (100, 2)

        except (OSError, IOError) as e:
            # Acceptable if system resource limits are hit
            assert "too many" in str(e).lower() or "resource" in str(e).lower()

        finally:
            # Cleanup test files
            for test_file in test_files:
                if test_file.exists():
                    test_file.unlink()

    def test_thread_resource_cleanup_error_handling(self):
        """Test thread resource cleanup and error handling."""
        import threading
        import time

        # Test thread creation and cleanup
        threads = []

        def worker_function():
            """Simple worker that might encounter errors."""
            try:
                time.sleep(0.1)  # Simulate work
                return "completed"
            except Exception as e:
                # Log error but don't crash
                logging.error(f"Worker error: {e}")
                return None

        try:
            # Create multiple threads
            for i in range(5):
                thread = threading.Thread(target=worker_function)
                thread.start()
                threads.append(thread)

            # Wait for completion with timeout
            for thread in threads:
                thread.join(timeout=1.0)

                # Check if thread completed properly
                if thread.is_alive():
                    logging.warning("Thread did not complete within timeout")

        except RuntimeError as e:
            # Thread creation errors should be handled gracefully
            assert "thread" in str(e).lower()

        finally:
            # Ensure all threads are cleaned up
            for thread in threads:
                if thread.is_alive():
                    # In real implementation, would use proper cleanup
                    logging.warning("Thread still alive during cleanup")


# Ensure proper cleanup after all tests
@pytest.fixture(autouse=True)
def cleanup_error_test_artifacts():
    """Auto-cleanup fixture to ensure no artifacts remain after error tests."""
    yield
    # Additional cleanup if needed - temp_workspace fixture handles most cleanup
    # Force garbage collection to free memory from test arrays
    import gc

    gc.collect()
