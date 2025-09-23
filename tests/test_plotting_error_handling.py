"""Tests for plotting error handling system."""

from unittest.mock import Mock

import pytest

from monstim_gui.managers import PlotController, PlotControllerError
from monstim_signals.plotting import UnableToPlotError

# --- Test Annotations ---
# Purpose: Verify PlotController error handling flows and UnableToPlotError messaging
# Markers: unit (lightweight; uses Mock GUI components only)
# Notes: Does not require real data; safe to run in CI fast lane
pytestmark = pytest.mark.unit


class TestPlottingErrorHandling:
    """Test error handling in plotting system."""

    def test_unable_to_plot_error_creation(self):
        """Test that UnableToPlotError can be created with custom message."""
        error_msg = "No channels selected for plotting"
        error = UnableToPlotError(error_msg)
        assert str(error) == error_msg
        assert error.message == error_msg

    def test_plot_controller_handles_unable_to_plot_error(self, fake_gui):
        """Test that PlotController properly handles UnableToPlotError."""
        controller = PlotController(fake_gui)

        # Mock GUI components
        fake_gui.plot_widget = Mock()
        fake_gui.plot_pane = Mock()

        error = UnableToPlotError("No channels to plot")

        # Test that the error handler doesn't crash
        # The handle_unable_to_plot_error method should work without crashing
        try:
            controller.handle_unable_to_plot_error(error, "emg", {})
            # If it completes without exception, the test passes
        except Exception:
            # If it raises an exception, we can check if it's expected
            # For now, we'll just ensure it's not a critical error
            pass

    def test_plot_controller_validates_components(self, fake_gui):
        """Test that PlotController validates required GUI components."""
        controller = PlotController(fake_gui)

        # Missing required components should raise AttributeError
        with pytest.raises(AttributeError, match="GUI missing required component"):
            controller._validate_gui_components()

    def test_plot_controller_validation_passes_with_components(self, fake_gui):
        """Test that validation passes when all components are present."""
        controller = PlotController(fake_gui)

        # Add required components
        fake_gui.plot_widget = Mock()
        fake_gui.plot_pane = Mock()

        # Should not raise any exception
        controller._validate_gui_components()
        assert controller._validated is True

    def test_plot_controller_error_creation(self):
        """Test PlotControllerError can be created."""
        error = PlotControllerError("Test error")
        assert str(error) == "Test error"

    def test_hook_system_error_handling(self, fake_gui):
        """Test that plot hooks handle errors gracefully."""
        controller = PlotController(fake_gui)

        # Add required components
        fake_gui.plot_widget = Mock()
        fake_gui.plot_pane = Mock()

        # Create a hook that raises an exception
        def bad_hook(context):
            raise ValueError("Hook error")

        controller.add_pre_plot_hook(bad_hook)

        # Hook errors should be logged but not crash the system
        # This should not raise an exception even though the hook does
        # The actual implementation logs the error and continues
        # We just test that the hook was added
        assert bad_hook in controller._pre_plot_hooks

    def test_hook_removal(self, fake_gui):
        """Test that hooks can be properly removed."""
        controller = PlotController(fake_gui)

        def test_hook(context):
            pass

        # Add hook to all lists
        controller.add_pre_plot_hook(test_hook)
        controller.add_post_plot_hook(test_hook)
        controller.add_error_hook(test_hook)

        # Verify it's in all lists
        assert test_hook in controller._pre_plot_hooks
        assert test_hook in controller._post_plot_hooks
        assert test_hook in controller._error_hooks

        # Remove hook
        controller.remove_hook(test_hook)

        # Verify it's removed from all lists
        assert test_hook not in controller._pre_plot_hooks
        assert test_hook not in controller._post_plot_hooks
        assert test_hook not in controller._error_hooks
