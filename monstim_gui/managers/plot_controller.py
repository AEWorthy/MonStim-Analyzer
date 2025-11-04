import copy
import logging
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from monstim_gui import MonstimGUI

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from monstim_signals.plotting import UnableToPlotError

from ..core.utils.dataframe_exporter import DataFrameDialog
from ..plotting import PLOT_NAME_DICT


class PlotControllerError(Exception):
    """Custom exception for PlotController errors."""

    pass


class PlotController:
    """Handle plotting and returning raw data."""

    def __init__(self, gui: "MonstimGUI"):
        self.gui: "MonstimGUI" = gui
        self._validated = False

        # Hook system for extensibility
        self._pre_plot_hooks = []
        self._post_plot_hooks = []
        self._error_hooks = []

    def _validate_gui_components(self):
        """Validate that the GUI has all required components for plotting."""
        if self._validated:
            return

        required_components = [
            ("plot_widget", "PlotWidget for plot controls"),
            ("plot_pane", "PlotPane for displaying plots"),
        ]

        for component, description in required_components:
            if not hasattr(self.gui, component):
                raise AttributeError(f"GUI missing required component: {component} ({description})")

        self._validated = True

    def initialize(self):
        """Initialize the plot controller after GUI components are ready."""
        self._validate_gui_components()
        logging.debug("PlotController initialized successfully")

    def get_plot_configuration(self):
        """Get current plot configuration from GUI in a robust way."""
        self._validate_gui_components()  # Ensure components exist before accessing them

        if not hasattr(self.gui.plot_widget, "plot_type_combo"):
            raise AttributeError("PlotWidget missing plot_type_combo")

        plot_type_raw = self.gui.plot_widget.plot_type_combo.currentText()
        plot_type = PLOT_NAME_DICT.get(plot_type_raw)
        plot_options = self.gui.plot_widget.get_plot_options()

        return {
            "plot_type_raw": plot_type_raw,
            "plot_type": plot_type,
            "plot_options": plot_options,
        }

    def get_plot_level_and_object(self):
        """Get the current plot level (session/dataset/experiment) and corresponding data object."""
        self._validate_gui_components()  # Ensure components exist before accessing them

        plot_widget = self.gui.plot_widget

        if plot_widget.session_radio.isChecked():
            return "session", self.gui.current_session
        elif plot_widget.dataset_radio.isChecked():
            return "dataset", self.gui.current_dataset
        elif plot_widget.experiment_radio.isChecked():
            return "experiment", self.gui.current_experiment
        else:
            return None, None

    def plot_data(self, return_raw_data: bool = False):
        try:
            self._validate_gui_components()
            self.gui.plot_pane.show()
            config_data = self.get_plot_configuration()
        except AttributeError as e:
            QMessageBox.critical(self.gui, "Configuration Error", f"Plot configuration error: {e}")
            logging.error(f"Plot configuration error: {e}")
            return None

        plot_type_raw = config_data["plot_type_raw"]
        plot_type = config_data["plot_type"]
        # Use a separate copy of plot options for hooks/plotting so hooks
        # cannot silently mutate the UI's live options object. Hooks will
        # receive and can modify this copy; the modified copy is what gets
        # passed to the plotting layer. This ensures the UI always reflects
        # the user's selected options.
        plot_options = config_data["plot_options"]
        raw_data = None

        # Work on a deepcopy that will be passed to hooks and used for plotting
        plot_options_for_hooks = copy.deepcopy(plot_options) if plot_options is not None else {}

        # Only inject stimuli_to_plot for session-level EMG plots
        config = None
        if self.gui.current_session and hasattr(self.gui.current_session, "_config"):
            config = self.gui.current_session._config

        is_session_emg_plot = self.gui.plot_widget.session_radio.isChecked() and plot_type_raw == "EMG"

        # Determine if the default profile is selected
        default_profile_selected = False
        if hasattr(self.gui, "profile_selector_combo"):
            default_profile_selected = self.gui.profile_selector_combo.currentIndex() == 0

        if is_session_emg_plot:
            if config and "stimuli_to_plot" in config:
                plot_options_for_hooks["stimuli_to_plot"] = config["stimuli_to_plot"]
            elif default_profile_selected:
                plot_options_for_hooks["stimuli_to_plot"] = ["Electrical"]

        level, level_object = self.get_plot_level_and_object()

        if level is None:
            QMessageBox.warning(
                self.gui,
                "Warning",
                "Please select a level to plot data from (session, dataset, or experiment).",
            )
            logging.warning("No level selected for plotting data.")
            return None

        # Validate the plot request
        is_valid, error_msg = self.validate_plot_request(level, level_object, plot_type)
        if not is_valid:
            QMessageBox.warning(self.gui, "Warning", error_msg)
            logging.warning(error_msg)
            return None

        # Call pre-plot hooks
        plot_context = {
            "level": level,
            "level_object": level_object,
            "plot_type": plot_type,
            "plot_type_raw": plot_type_raw,
            "plot_options": plot_options_for_hooks,
        }

        for hook in self._pre_plot_hooks:
            try:
                hook(plot_context)
            except Exception as hook_error:
                logging.warning(f"Error in pre-plot hook: {hook_error}")

        try:
            # Use the possibly modified copy from hooks for plotting
            final_plot_options = plot_context.get("plot_options", plot_options_for_hooks)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            raw_data = level_object.plot(
                plot_type=plot_type,
                **final_plot_options,
                canvas=self.gui.plot_pane,
            )
        except UnableToPlotError as e:
            self.handle_unable_to_plot_error(e, plot_type, plot_options)
            return None
        except Exception as e:
            self.handle_plot_error(e, plot_type, plot_options)
            return None
        finally:
            QApplication.restoreOverrideCursor()

        logging.info(f"Plot Created. level: {level} type: {plot_type}, return_raw_data: {return_raw_data}.")

        # Update plot pane with error handling
        try:
            if hasattr(self.gui.plot_pane, "layout") and self.gui.plot_pane.layout:
                self.gui.plot_pane.layout.update()
        except (AttributeError, RuntimeError) as e:
            logging.warning(f"Could not update plot pane layout: {e}")

        # Call post-plot hooks
        plot_result = {
            "level": level,
            "plot_type": plot_type,
            "raw_data": raw_data,
            "return_raw_data": return_raw_data,
        }

        for hook in self._post_plot_hooks:
            try:
                hook(plot_result)
            except Exception as hook_error:
                logging.warning(f"Error in post-plot hook: {hook_error}")

        if return_raw_data:
            return raw_data
        return None

    def get_raw_data(self):
        """Get raw data from plotting and display it in a dialog."""
        try:
            raw_data = self.plot_data(return_raw_data=True)
        except AttributeError as e:
            QMessageBox.critical(self.gui, "Configuration Error", f"Cannot get raw data: {e}")
            logging.error(f"Configuration error in get_raw_data: {e}")
            return

        if raw_data is not None:
            dialog = DataFrameDialog(raw_data, self.gui)
            dialog.exec()
        else:
            QMessageBox.warning(self.gui, "Warning", "No data to display.")

    def validate_plot_request(self, level, level_object, plot_type):
        """Validate that the plot request is feasible."""
        if level_object is None:
            return (
                False,
                f"No {level} data exists to plot. Please try importing experiment data first.",
            )

        if plot_type is None:
            return (
                False,
                f"Invalid plot type selected: level={level}, plot_type={plot_type}.",
            )

        # Add more specific validations as needed
        return True, None

    def handle_unable_to_plot_error(self, e, plot_type, plot_options):
        """Handle UnableToPlotError with user-friendly messages."""
        error_msg = str(e)

        # Provide user-friendly guidance based on the error message
        if "No channels to plot" in error_msg:
            user_msg = (
                "No channels are currently selected for plotting.\n\n"
                "Please select at least one channel in the channel selection area "
                "and try plotting again."
            )
            title = "No Channels Selected"
        else:
            user_msg = f"Unable to create plot: {error_msg}"
            title = "Plot Error"

        # Show user-friendly message (no stack trace) if not running in a headless/testing context.
        suppress_dialog = hasattr(self.gui, "headless") and getattr(self.gui, "headless") is True
        if not suppress_dialog:
            QMessageBox.warning(self.gui, title, user_msg)

        # Log the error for debugging purposes
        logging.warning(f"Unable to plot: {error_msg}")
        logging.info(f"Plot type: {plot_type}, options: {plot_options}")
        logging.info(f"Current session: {self.gui.current_session}, current dataset: {self.gui.current_dataset}")
    # TODO: Structured plotting errors
    # - Replace ad-hoc string matching (e.g., "No channels to plot") with
    #   structured exception types or error codes (e.g., UnableToPlotError(reason='no_channels')).
    # - This makes it trivial to present contextual help dialogs and to
    #   programmatically handle recoverable conditions in hooks.

    def handle_plot_error(self, e, plot_type, plot_options):
        """Centralized error handling for plot operations."""
        error_msg = f"An error occurred while plotting: {e}"
        QMessageBox.critical(self.gui, "Error", error_msg)

        logging.error(error_msg)
        logging.error(f"Plot type: {plot_type}, options: {plot_options}")
        logging.error(f"Current session: {self.gui.current_session}, current dataset: {self.gui.current_dataset}")
        logging.error(traceback.format_exc())

        # Call error hooks
        for hook in self._error_hooks:
            try:
                hook(e, plot_type, plot_options)
            except Exception as hook_error:
                logging.warning(f"Error in error hook: {hook_error}")

# TODO: Hook examples and recipes
# - Document example pre/post plot hooks in the codebase (or a small docs file)
#   demonstrating how to automatically overlay latency windows, highlight
#   plateau regions, or compute summary metrics after each successful plot.

    def add_pre_plot_hook(self, hook_func):
        """Add a function to be called before plotting."""
        self._pre_plot_hooks.append(hook_func)

    def add_post_plot_hook(self, hook_func):
        """Add a function to be called after successful plotting."""
        self._post_plot_hooks.append(hook_func)

    def add_error_hook(self, hook_func):
        """Add a function to be called when plotting errors occur."""
        self._error_hooks.append(hook_func)

    def remove_hook(self, hook_func):
        """Remove a hook function from all hook lists."""
        for hook_list in [
            self._pre_plot_hooks,
            self._post_plot_hooks,
            self._error_hooks,
        ]:
            if hook_func in hook_list:
                hook_list.remove(hook_func)
