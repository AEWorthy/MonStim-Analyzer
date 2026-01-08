import copy
import gc
import logging
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from monstim_gui import MonstimGUI

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

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

        # Diagnostic tracking for extended sessions
        self._plot_count = 0
        self._total_memory_freed = 0.0

        # Safety flag: disable automatic GC if it causes crashes
        # Can be set to False if deferred GC still causes issues on certain systems
        self._enable_auto_gc = True
        self._gc_failure_count = 0  # Track GC-related issues
        self._max_gc_failures = 3  # Disable GC after this many issues

        # Input blocking during critical graphics operations
        # Prevents user interactions from interfering with screen refresh/layout updates
        self._block_input_during_graphics_update = True  # Safety feature to prevent crashes
        self._is_in_graphics_critical_section = False

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

    def _disable_plot_controls(self):
        """Temporarily disable plot controls to prevent user input during critical graphics operations."""
        try:
            if hasattr(self.gui, "plot_widget") and self.gui.plot_widget:
                self.gui.plot_widget.setEnabled(False)
            if hasattr(self.gui, "data_selection_widget") and self.gui.data_selection_widget:
                self.gui.data_selection_widget.setEnabled(False)
        except Exception as e:
            logging.warning(f"Could not disable plot controls: {e}")

    def _enable_plot_controls(self):
        """Re-enable plot controls after critical graphics operations complete."""
        try:
            if hasattr(self.gui, "plot_widget") and self.gui.plot_widget:
                self.gui.plot_widget.setEnabled(True)
            if hasattr(self.gui, "data_selection_widget") and self.gui.data_selection_widget:
                self.gui.data_selection_widget.setEnabled(True)
        except Exception as e:
            logging.warning(f"Could not re-enable plot controls: {e}")

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
        # Increment plot counter for extended session diagnostics
        self._plot_count += 1
        logging.debug(f"=== Starting plot operation #{self._plot_count} ===")

        # Check if GC should be temporarily disabled due to previous issues
        if self._plot_count > 100 and self._total_memory_freed < 1.0 and self._enable_auto_gc:
            logging.warning(
                f"Auto-GC not freeing memory after {self._plot_count} plots "
                f"({self._total_memory_freed:.2f} MB freed total). Consider disabling if stability issues occur."
            )

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

        # Thread safety check: Ensure we're on the main GUI thread
        from PySide6.QtCore import QThread

        if QThread.currentThread() != QApplication.instance().thread():
            error_msg = "CRITICAL: Plotting attempted from non-main thread! This can cause crashes."
            logging.error(error_msg)
            logging.error(f"Current thread: {QThread.currentThread()}")
            logging.error(f"Main thread: {QApplication.instance().thread()}")
            raise RuntimeError(error_msg)

        try:
            # Use the possibly modified copy from hooks for plotting
            final_plot_options = plot_context.get("plot_options", plot_options_for_hooks)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            logging.debug(f"Starting plot rendering for {plot_type} at {level} level...")
            raw_data = level_object.plot(
                plot_type=plot_type,
                **final_plot_options,
                canvas=self.gui.plot_pane,
            )
            logging.debug("Plot rendering completed successfully.")

            # Ensure PyQtGraph graphics scene is fully updated before continuing
            # This helps prevent crashes when Qt processes pending paint events
            try:
                if hasattr(self.gui.plot_pane, "graphics_layout"):
                    self.gui.plot_pane.graphics_layout.update()
                    logging.debug("Graphics layout update() called to stabilize scene")
            except Exception as e:
                logging.warning(f"Could not call update() on graphics layout: {e}")

        except UnableToPlotError as e:
            self.handle_unable_to_plot_error(e, plot_type, plot_options)
            # Ensure controls are re-enabled on error
            if self._is_in_graphics_critical_section:
                self._enable_plot_controls()
                self._is_in_graphics_critical_section = False
            return None
        except Exception as e:
            self.handle_plot_error(e, plot_type, plot_options)
            # Ensure controls are re-enabled on error
            if self._is_in_graphics_critical_section:
                self._enable_plot_controls()
                self._is_in_graphics_critical_section = False
            return None
        finally:
            QApplication.restoreOverrideCursor()

        logging.info(f"Plot Created. level: {level} type: {plot_type}, return_raw_data: {return_raw_data}.")
        logging.debug("Plot creation completed successfully. Beginning post-plot operations.")

        # Block user input during critical graphics update phase to prevent crashes
        # This prevents mouse/keyboard events from interfering with screen refresh
        if self._block_input_during_graphics_update:
            self._is_in_graphics_critical_section = True
            self._disable_plot_controls()
            logging.debug("User input blocked during graphics update phase")

        # Defer layout update to avoid immediate Qt/graphics driver issues
        # Uses QTimer.singleShot to push the update to the next event loop iteration
        def safe_layout_update():
            """Safely update plot pane layout with comprehensive error handling."""
            try:
                logging.debug("Attempting deferred layout update...")
                if hasattr(self.gui.plot_pane, "layout") and self.gui.plot_pane.layout:
                    self.gui.plot_pane.layout.update()
                    logging.debug("Layout update completed successfully.")
                else:
                    logging.warning("Plot pane layout not available for update")
            except (AttributeError, RuntimeError) as e:
                logging.warning(f"Could not update plot pane layout: {e}")
            except Exception as e:
                logging.error(f"CRITICAL: Unexpected error during layout update: {e}", exc_info=True)
            finally:
                # Layout update is complete - this is a good time to re-enable input
                # We'll still wait for the heartbeat before fully re-enabling
                pass

        # Schedule layout update for next event loop iteration (safer timing)
        try:
            QTimer.singleShot(0, safe_layout_update)
            logging.debug("Layout update scheduled successfully")
        except Exception as e:
            logging.error(f"CRITICAL: Failed to schedule layout update: {e}", exc_info=True)

        # Call post-plot hooks with enhanced error tracking
        plot_result = {
            "level": level,
            "plot_type": plot_type,
            "raw_data": raw_data,
            "return_raw_data": return_raw_data,
        }

        logging.debug(f"Executing {len(self._post_plot_hooks)} post-plot hooks...")
        for i, hook in enumerate(self._post_plot_hooks):
            try:
                logging.debug(f"Executing post-plot hook {i+1}/{len(self._post_plot_hooks)}...")
                hook(plot_result)
                logging.debug(f"Post-plot hook {i+1} completed successfully.")
            except Exception as hook_error:
                logging.warning(f"Error in post-plot hook {i+1}: {hook_error}", exc_info=True)

        # Memory management: Defer garbage collection to avoid Qt/graphics driver issues
        # This helps with memory accumulation during extended sessions
        # CRITICAL: GC must happen AFTER Qt has fully processed all graphics events
        # Immediate GC can crash if Qt objects with native handles are collected
        # Can be disabled by setting self._enable_auto_gc = False if it causes issues
        if self._enable_auto_gc:
            logging.debug("Scheduling deferred garbage collection (500ms delay)...")

            def deferred_gc():
                """Safely perform garbage collection after Qt events are fully processed."""
                try:
                    logging.debug("Starting deferred garbage collection...")

                    # Get memory usage before collection (if psutil available)
                    mem_before = None
                    try:
                        import psutil

                        process = psutil.Process()
                        mem_before = process.memory_info().rss / 1024 / 1024  # MB
                        logging.debug(f"Memory usage before GC: {mem_before:.2f} MB")
                    except ImportError:
                        pass  # psutil not available, skip memory monitoring
                    except Exception as e:
                        logging.debug(f"Could not get memory info: {e}")

                    logging.debug("Calling gc.collect()...")
                    collected = gc.collect()
                    logging.debug(f"Garbage collection completed. Collected {collected} objects.")

                    # Get memory usage after collection
                    if mem_before is not None:
                        try:
                            mem_after = process.memory_info().rss / 1024 / 1024  # MB
                            mem_freed = mem_before - mem_after
                            self._total_memory_freed += max(0, mem_freed)  # Track cumulative freed memory
                            logging.debug(f"Memory usage after GC: {mem_after:.2f} MB (freed: {mem_freed:.2f} MB)")
                            logging.debug(
                                f"Session totals: {self._plot_count} plots, {self._total_memory_freed:.2f} MB freed cumulatively"
                            )
                        except Exception as e:
                            logging.debug(f"Could not calculate memory freed: {e}")

                except Exception as gc_error:
                    logging.error(f"CRITICAL: Deferred garbage collection failed: {gc_error}", exc_info=True)

            # Schedule GC for 500ms later - gives Qt plenty of time to finish all operations
            try:
                QTimer.singleShot(500, deferred_gc)
                logging.debug("Deferred GC scheduled successfully")
            except Exception as e:
                logging.error(f"CRITICAL: Failed to schedule deferred GC: {e}", exc_info=True)
        else:
            logging.debug("Automatic garbage collection disabled (potential stability issue).")

        logging.info("Plot operation completed successfully. Returning control to GUI.")

        # Diagnostic heartbeat: log when we successfully return to event loop
        # This helps identify if crashes happen during the return or after
        # Also re-enable user input after verifying graphics operations are stable
        def heartbeat_check():
            try:
                logging.debug(f"Heartbeat: Plot #{self._plot_count} - GUI event loop responsive after plot completion")

                # Re-enable user input after graphics operations are complete
                if self._is_in_graphics_critical_section:
                    self._enable_plot_controls()
                    self._is_in_graphics_critical_section = False
                    logging.debug("User input re-enabled after graphics update completion")
            except Exception as e:
                logging.error(f"Error in heartbeat check: {e}", exc_info=True)
                # Ensure controls are re-enabled even if there's an error
                try:
                    self._enable_plot_controls()
                    self._is_in_graphics_critical_section = False
                except Exception:
                    logging.error("Failed to re-enable plot controls in heartbeat error handling", exc_info=True)
                    pass

        try:
            # Re-enable input after ~50ms (after layout update and GC are complete)
            QTimer.singleShot(50, heartbeat_check)
        except Exception as e:
            logging.error(f"CRITICAL: Failed to schedule heartbeat check: {e}", exc_info=True)
            # Emergency: re-enable controls immediately if scheduling fails
            self._enable_plot_controls()
            self._is_in_graphics_critical_section = False

        if return_raw_data:
            logging.debug(f"Returning raw data with {len(raw_data) if raw_data else 0} entries.")
            return raw_data

        logging.debug("Returning None from plot_data()")
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
            # Get current plot configuration for metadata
            config = self.get_plot_configuration()
            level, _ = self.get_plot_level_and_object()

            dialog = DataFrameDialog(raw_data, self.gui, plot_type=config["plot_type_raw"], data_level=level)
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

    def force_memory_cleanup(self):
        """Manually trigger aggressive memory cleanup for extended sessions.

        This can be called periodically (e.g., every N plots) or when the user
        notices performance degradation during extended sessions.
        """
        logging.info("Forcing aggressive memory cleanup...")
        try:
            # Clear current plots
            if hasattr(self.gui, "plot_pane") and self.gui.plot_pane:
                self.gui.plot_pane.clear_plots()
                logging.debug("Plot pane cleared.")

            # Process pending Qt events
            QApplication.processEvents()

            # Run garbage collection multiple times for thorough cleanup
            for generation in range(3):
                collected = gc.collect(generation)
                logging.debug(f"GC generation {generation}: collected {collected} objects")

            # Log session statistics
            logging.info(
                f"Session statistics: {self._plot_count} plots created, "
                f"{self._total_memory_freed:.2f} MB freed cumulatively"
            )

        except Exception as e:
            logging.error(f"Error during forced memory cleanup: {e}", exc_info=True)
