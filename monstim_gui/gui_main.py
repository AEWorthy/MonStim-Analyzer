from __future__ import annotations

import logging
import os
import sys
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from monstim_gui.widgets.gui_layout import (
        MenuBar,
        DataSelectionWidget,
        ReportsWidget,
        PlotPane,
        PlotWidget,
    )
    from PySide6.QtWidgets import QStatusBar

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
)

from monstim_gui.commands import (
    ChangeChannelNamesCommand,
    CommandInvoker,
    ExcludeDatasetCommand,
    ExcludeRecordingCommand,
    ExcludeSessionCommand,
    InvertChannelPolarityCommand,
    RestoreDatasetCommand,
    RestoreRecordingCommand,
    RestoreSessionCommand,
)
from monstim_gui.core.splash import SPLASH_INFO
from monstim_gui.dialogs import (
    AboutDialog,
    ChangeChannelNamesDialog,
    InvertChannelPolarityDialog,
    LatencyWindowsDialog,
)
from monstim_gui.io.config_repository import ConfigRepository
from monstim_gui.io.help_repository import HelpFileRepository
from monstim_gui.managers import BulkExportManager, DataManager, PlotController, ReportManager
from monstim_gui.widgets.gui_layout import setup_main_layout
from monstim_signals.core import (
    get_config_path,
    get_docs_path,
    get_output_path,
    get_source_path,
)
from monstim_signals.domain.dataset import Dataset
from monstim_signals.domain.experiment import Experiment
from monstim_signals.domain.session import Session


class MonstimGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"MonStim Analyzer {SPLASH_INFO['version']}")
        self.setWindowIcon(QIcon(os.path.join(get_source_path(), "icon.png")))
        self.handle_qt_error_logs()

        # Use responsive window sizing with state restoration
        # from monstim_gui.core.ui_scaling import ui_scaling
        from monstim_gui.core.ui_config import ui_config

        # Try to restore previous window state, otherwise use responsive sizing
        if not ui_config.restore_window_state(self):
            x, y, width, height = ui_config.get_window_geometry()
            self.setGeometry(x, y, width, height)

        # Initialize variables
        self.expts_dict = {}
        self.expts_dict_keys = []  # type: list[str]
        self.current_experiment: Experiment | None = None
        self.current_dataset: Dataset | None = None
        self.current_session: Session | None = None
        self.channel_names = []

        # Set default paths
        self.output_path = get_output_path()
        self.config_file = get_config_path()

        # Profile manager for analysis profiles
        from monstim_gui.managers import ProfileManager

        self.profile_manager = ProfileManager()
        self.active_profile_path = None
        self.active_profile_data = None

        # Helper managers
        self.data_manager = DataManager(self)
        self.report_manager = ReportManager(self)
        self.plot_controller = PlotController(self)
        self.bulk_export_manager = BulkExportManager(self)

        self.config_repo = ConfigRepository(get_config_path())
        self.help_repo = HelpFileRepository(get_docs_path())

        self.init_ui()

        # Initialize managers after UI is set up
        self.plot_controller.initialize()

        # Initialize data selection combos (preserve selection if any existing state)
        self.data_selection_widget.refresh()

        self.plot_widget.initialize_plot_widget()

        # Re-center window after UI is fully initialized to account for final size
        self._recenter_window()

        # Restore last session state (profile, experiment, dataset, session)
        self._restore_last_session()

        self.command_invoker = CommandInvoker(self)
        # Initialize undo/redo menu state
        self.menu_bar.update_undo_redo_labels()

    def init_ui(self):
        widgets = setup_main_layout(self)
        self.menu_bar: "MenuBar" = widgets["menu_bar"]
        self.data_selection_widget: "DataSelectionWidget" = widgets["data_selection_widget"]
        self.reports_widget: "ReportsWidget" = widgets["reports_widget"]
        self.plot_pane: "PlotPane" = widgets["plot_pane"]
        self.plot_widget: "PlotWidget" = widgets["plot_widget"]
        self.status_bar: "QStatusBar" = widgets["status_bar"]

        # --- Add Profile Selector to Main Window ---
        from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

        self.profile_selector_row = QWidget()
        self.profile_selector_layout = QHBoxLayout(self.profile_selector_row)
        self.profile_selector_layout.setContentsMargins(8, 2, 8, 2)
        self.profile_selector_label = QLabel("Analysis Profile:")
        self.profile_selector_combo = QComboBox()
        self.profile_selector_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.profile_selector_combo.setMinimumContentsLength(16)
        self.profile_selector_combo.setEditable(False)
        self.profile_selector_layout.addWidget(self.profile_selector_label)
        self.profile_selector_layout.addWidget(self.profile_selector_combo, 1)
        self.profile_selector_row.setMaximumHeight(36)
        # Insert at the top of the left panel (above data selection)
        left_panel = self.data_selection_widget.parentWidget()
        while left_panel is not None and not hasattr(left_panel, "layout"):
            left_panel = left_panel.parentWidget()
        if left_panel is not None:
            left_layout = left_panel.layout()
            if hasattr(left_layout, "insertWidget"):
                left_layout.insertWidget(0, self.profile_selector_row)  # type: ignore
            else:
                raise RuntimeError("Left panel layout does not support insertWidget.")
        else:
            logging.error("Left panel not found or does not have a layout.")

        self._populate_profile_selector()
        self.profile_selector_combo.currentIndexChanged.connect(self._on_profile_selector_changed)

        self.status_bar.showMessage(f"Welcome to MonStim Analyzer, {SPLASH_INFO['version']}", 10000)

        # Migration banner (hidden by default)
        try:
            from monstim_gui.widgets.migration_banner import MigrationBanner

            self.migration_banner = MigrationBanner(self)
            # Place the banner just above the status bar in main layout if available
            main = self.centralWidget()
            if main and hasattr(main, "layout") and callable(getattr(main, "layout")):
                lay = main.layout()
                if hasattr(lay, "addWidget"):
                    lay.addWidget(self.migration_banner)
            # Wire button to trigger background migrations
            self.migration_banner.run_clicked.connect(self._run_background_migrations)
        except Exception:
            logging.debug("Failed to create migration banner (non-fatal).", exc_info=True)

    @staticmethod
    def handle_qt_error_logs():
        # Install Qt message handler to suppress QPainter warnings during resize operations
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler

        def qt_message_handler(mode, context, message):
            # Suppress specific QPainter warnings that occur during window resize/fullscreen operations
            if any(
                warning in message
                for warning in [
                    "QPainter::begin: Paint device returned engine == 0",
                    "QPainter::setCompositionMode: Painter not active",
                    "QPainter::fillRect: Painter not active",
                    "QPainter::setBrush: Painter not active",
                    "QPainter::setPen: Painter not active",
                    "QPainter::drawPath: Painter not active",
                    "QPainter::setFont: Painter not active",
                    "QPainter::end: Painter not active",
                ]
            ):
                return  # Suppress these messages

            # Let other messages through to normal logging
            if mode == QtMsgType.QtWarningMsg:
                logging.warning(f"Qt: {message}")
            elif mode == QtMsgType.QtCriticalMsg:
                logging.error(f"Qt: {message}")
            elif mode == QtMsgType.QtFatalMsg:
                logging.critical(f"Qt: {message}")
            elif mode == QtMsgType.QtDebugMsg:
                logging.debug(f"Qt: {message}")

        qInstallMessageHandler(qt_message_handler)

    def _populate_profile_selector(self):
        self.profile_selector_combo.blockSignals(True)
        self.profile_selector_combo.clear()
        self.profile_selector_combo.addItem("(default)", userData=None)
        self.profile_selector_combo.setItemData(
            0,
            "Use the global/default analysis settings.",
            role=Qt.ItemDataRole.ToolTipRole,
        )
        self._profile_list = self.profile_manager.list_profiles()
        for idx, (name, path, data) in enumerate(self._profile_list, start=1):
            self.profile_selector_combo.addItem(name, userData=path)
            desc = data.get("description", "")
            if desc:
                self.profile_selector_combo.setItemData(idx, desc, role=Qt.ItemDataRole.ToolTipRole)
        self.profile_selector_combo.blockSignals(False)
        self.profile_selector_combo.setCurrentIndex(0)
        self._set_profile_selector_tooltip(0)

    def _set_profile_selector_tooltip(self, idx):
        # Set the tooltip for the whole combobox to the selected profile's description
        tooltip = self.profile_selector_combo.itemData(idx, role=Qt.ItemDataRole.ToolTipRole)
        if tooltip:
            self.profile_selector_combo.setToolTip(tooltip)
        else:
            self.profile_selector_combo.setToolTip("")

    def _on_profile_selector_changed(self, idx):
        from monstim_gui.core.application_state import app_state

        self._set_profile_selector_tooltip(idx)
        profile_name = self.profile_selector_combo.currentText()

        if idx == 0:
            # Global config
            self.active_profile_path = None
            self.active_profile_data = None
            config = self.config_repo.read_config()
        else:
            name, path, data = self._profile_list[idx - 1]
            self.active_profile_path = path
            self.active_profile_data = data
            # Merge profile data with global config for fallback
            config = self.config_repo.read_config()
            # Overlay profile analysis_parameters and latency_window_preset, etc.
            if "analysis_parameters" in data:
                config.update(data["analysis_parameters"])
            if "latency_window_preset" in data:
                config["latency_window_preset"] = data["latency_window_preset"]
            if "stimuli_to_plot" in data:
                config["stimuli_to_plot"] = data["stimuli_to_plot"]

        # Save profile selection and update session state
        app_state.save_last_profile(profile_name)
        app_state.save_recent_profile(profile_name)  # Also add to recent profiles list
        if self.current_experiment:
            app_state.save_current_session_state(
                experiment_id=self.current_experiment.id,
                dataset_id=self.current_dataset.id if self.current_dataset else None,
                session_id=self.current_session.id if self.current_session else None,
                profile_name=profile_name,
            )
        else:
            # Even when no experiment is loaded, ensure we can track profile changes
            logging.info(f"No experiment loaded, but profile selection saved: {profile_name}")

        self.update_domain_configs(config)
        self.status_bar.showMessage(f"Profile applied: {self.profile_selector_combo.currentText()}", 4000)

    def refresh_profile_selector(self):
        # Store the current selected profile name
        current_name = self.profile_selector_combo.currentText()
        self._populate_profile_selector()
        # Try to restore the same profile selection if it still exists
        idx = self.profile_selector_combo.findText(current_name)
        if idx >= 0:
            self.profile_selector_combo.setCurrentIndex(idx)
        else:
            self.profile_selector_combo.setCurrentIndex(0)

    def refresh_preferences_dependent_ui(self):
        """Refresh any UI elements that depend on program preferences."""
        # This method can be called when preferences change to update the UI
        # Currently no specific UI updates are needed, but this is a hook
        # for future functionality that might depend on preference settings
        pass

    def _recenter_window(self):
        """Re-center the window after UI initialization to account for actual final size."""
        from monstim_gui.core.ui_config import ui_config

        # Only recenter if centering is enabled and we're not restoring saved window state
        if not ui_config.get("center_windows", True):
            return

        # Let the window adjust to its final size based on layout constraints
        self.adjustSize()

        # Get the actual final size
        current_size = self.size()
        width = current_size.width()
        height = current_size.height()

        # Calculate centered position for the actual final size
        from monstim_gui.core.ui_scaling import ui_scaling

        centered_geometry = ui_scaling.get_centered_geometry(width, height)

        # Move to centered position (preserving the current size)
        self.move(centered_geometry.x(), centered_geometry.y())

    def restore_last_profile_selection(self):
        """Restore the last selected analysis profile."""
        from monstim_gui.core.application_state import app_state

        # Check if profile tracking is enabled
        if not app_state.should_track_analysis_profiles():
            logging.debug("Profile tracking is disabled - not restoring profile selection")
            return False

        last_profile = app_state.get_last_profile()
        logging.debug(f"Attempting to restore last profile selection: '{last_profile}'")

        if last_profile:
            idx = self.profile_selector_combo.findText(last_profile)
            if idx >= 0:
                # Block signals to prevent triggering the change handler during restoration
                self.profile_selector_combo.blockSignals(True)
                self.profile_selector_combo.setCurrentIndex(idx)
                self.profile_selector_combo.blockSignals(False)

                # Manually apply the profile configuration
                self._set_profile_selector_tooltip(idx)
                self._apply_profile_configuration(idx, last_profile)

                logging.debug(f"Successfully restored last profile selection: {last_profile}")
                return True
            else:
                logging.warning(f"Profile '{last_profile}' not found in available profiles")

        logging.info("No valid analysis profile to restore, using default")
        return False

    def _apply_profile_configuration(self, idx: int, profile_name: str):
        """Apply the configuration for the selected profile without saving it again."""
        if idx == 0:
            # Global config
            self.active_profile_path = None
            self.active_profile_data = None
            config = self.config_repo.read_config()
        else:
            name, path, data = self._profile_list[idx - 1]
            self.active_profile_path = path
            self.active_profile_data = data
            # Merge profile data with global config for fallback
            config = self.config_repo.read_config()
            # Overlay profile analysis_parameters and latency_window_preset, etc.
            if "analysis_parameters" in data:
                config.update(data["analysis_parameters"])
            if "latency_window_preset" in data:
                config["latency_window_preset"] = data["latency_window_preset"]
            if "stimuli_to_plot" in data:
                config["stimuli_to_plot"] = data["stimuli_to_plot"]

        self.update_domain_configs(config)
        logging.info(f"Applied configuration for profile: {profile_name}")

    def _restore_last_session(self):
        """Restore the last session state and analysis profile based on preferences."""
        from monstim_gui.core.application_state import app_state

        try:
            # Always attempt to restore profile selection (controlled by its own preference)
            self.restore_last_profile_selection()

            # Only restore data selections if session restoration is enabled
            if app_state.should_restore_session():
                success = app_state.restore_last_session(self)
                if success:
                    self.status_bar.showMessage("Previous session state restored", 5000)
                    logging.debug("Session restoration completed successfully")
                else:
                    logging.warning("Session restoration was attempted, but not possible")
            else:
                logging.debug("Session restoration is disabled - not restoring experiment/dataset/session")

        except Exception as e:
            logging.error(f"Error during session restoration: {e}")
            logging.error(traceback.format_exc())
            # Clear problematic state
            app_state.clear_session_state()

    def _run_background_migrations(self):
        try:
            if self.current_experiment and self.current_experiment.repo:
                from monstim_gui.io.migration_runner import MigrationRunner

                runner = MigrationRunner(str(self.current_experiment.repo.folder))
                from PySide6.QtWidgets import QProgressDialog

                dlg = QProgressDialog("Migrating annotations...", "Cancel", 0, 100, self)
                dlg.setWindowTitle("Background Migration")
                dlg.setWindowModality(Qt.WindowModality.WindowModal)
                dlg.setAutoClose(True)
                dlg.setAutoReset(True)
                runner.progress.connect(dlg.setValue)
                runner.status_update.connect(dlg.setLabelText)
                runner.finished.connect(dlg.close)
                runner.finished.connect(lambda n: self.status_bar.showMessage(f"Migrations complete: {n} files.", 5000))
                runner.error.connect(lambda e: QMessageBox.critical(self, "Migration Error", e))
                runner.error.connect(dlg.close)

                # Hide banner on completion and re-scan to decide future visibility
                def _on_migrations_complete(_n: int):
                    try:
                        if hasattr(self, "migration_banner"):
                            self.migration_banner.hide()
                        # Trigger a quick background re-scan; banner will only reappear if new work exists
                        from monstim_gui.io.migration_runner import MigrationScanThread

                        scan = MigrationScanThread(str(self.current_experiment.repo.folder))

                        def _on_scan(has_work: bool, count: int):
                            if has_work and hasattr(self, "migration_banner"):
                                msg = f"Annotation migrations detected ({count}). Run now to update files."
                                self.migration_banner.show_message(msg)
                            else:
                                if hasattr(self, "migration_banner"):
                                    self.migration_banner.hide()

                        scan.has_work.connect(_on_scan)
                        scan.error.connect(lambda e: logging.debug(f"Migration re-scan error: {e}"))
                        # Keep a reference to avoid GC during run
                        self._migration_rescan = scan
                        scan.start()
                    except Exception:
                        logging.debug("Failed to hide banner / rescan after migrations.", exc_info=True)

                runner.finished.connect(_on_migrations_complete)
                self._migration_runner = runner
                dlg.show()
                runner.start()
        except Exception:
            logging.debug("Failed to start background migrations.", exc_info=True)

    # Command functions
    def undo(self):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
            self.command_invoker.undo()
        finally:
            QApplication.restoreOverrideCursor()

    def redo(self):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
            self.command_invoker.redo()
        finally:
            QApplication.restoreOverrideCursor()

    def exclude_recording(self, recording_id: str):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
            command = ExcludeRecordingCommand(self, recording_id)
            self.command_invoker.execute(command)
        finally:
            QApplication.restoreOverrideCursor()

    def restore_recording(self, recording_id: str):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
            command = RestoreRecordingCommand(self, recording_id)
            self.command_invoker.execute(command)
        finally:
            QApplication.restoreOverrideCursor()

    def exclude_session(self):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
            command = ExcludeSessionCommand(self)
            self.command_invoker.execute(command)
        finally:
            QApplication.restoreOverrideCursor()

    def exclude_dataset(self):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
            command = ExcludeDatasetCommand(self)
            self.command_invoker.execute(command)
        finally:
            QApplication.restoreOverrideCursor()

    def restore_session(self, session_id: str):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            command = RestoreSessionCommand(self, session_id)
            self.command_invoker.execute(command)
        finally:
            QApplication.restoreOverrideCursor()

    def restore_dataset(self, dataset_id: str):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            command = RestoreDatasetCommand(self, dataset_id)
            self.command_invoker.execute(command)
        finally:
            QApplication.restoreOverrideCursor()

    def prompt_restore_session(self):
        if not self.current_dataset:
            QMessageBox.warning(self, "Warning", "Please select a dataset first.")
            return
        excluded = list(self.current_dataset.excluded_sessions)
        if not excluded:
            QMessageBox.information(self, "Info", "No excluded sessions to restore.")
            return
        session_id, ok = QInputDialog.getItem(
            self,
            "Restore Session",
            "Select session to restore:",
            excluded,
            0,
            False,
        )
        if ok and session_id:
            self.restore_session(session_id)

    def prompt_restore_dataset(self):
        if not self.current_experiment:
            QMessageBox.warning(self, "Warning", "Please select an experiment first.")
            return
        excluded = list(self.current_experiment.excluded_datasets)
        if not excluded:
            QMessageBox.information(self, "Info", "No excluded datasets to restore.")
            return
        dataset_id, ok = QInputDialog.getItem(
            self,
            "Restore Dataset",
            "Select dataset to restore:",
            excluded,
            0,
            False,
        )
        if ok and dataset_id:
            self.restore_dataset(dataset_id)

    # Menu bar functions
    def manage_latency_windows(self, level: str):
        logging.debug("Managing latency windows.")
        match level:
            case "experiment":
                if not self.current_experiment:
                    QMessageBox.warning(self, "Warning", "Please select an experiment first.")
                    return
                emg_data = self.current_experiment
            case "dataset":
                if not self.current_dataset:
                    QMessageBox.warning(self, "Warning", "Please load a dataset first.")
                    return
                emg_data = self.current_dataset
            case "session":
                if not self.current_session:
                    QMessageBox.warning(self, "Warning", "Please select a session first.")
                    return
                emg_data = self.current_session
            case _:
                QMessageBox.warning(self, "Warning", "Invalid level for managing latency windows.")
                return

        # Check if a latency windows dialog is already open and close it
        if hasattr(self, "_latency_dialog") and self._latency_dialog:
            self._latency_dialog.close()

        self._latency_dialog = LatencyWindowsDialog(emg_data, self)
        self._latency_dialog.show()  # Use show() instead of exec() to allow interaction with main window
        self._latency_dialog.raise_()  # Bring to front
        self._latency_dialog.activateWindow()  # Give focus

    def append_replace_latency_windows(self, level: str):
        """Open specialized dialog for appending/replacing a single latency window."""
        logging.debug(f"Opening append/replace latency window dialog for {level}.")
        match level:
            case "experiment":
                if not self.current_experiment:
                    QMessageBox.warning(self, "Warning", "Please select an experiment first.")
                    return
                emg_data = self.current_experiment
            case "dataset":
                if not self.current_dataset:
                    QMessageBox.warning(self, "Warning", "Please load a dataset first.")
                    return
                emg_data = self.current_dataset
            case "session":
                if not self.current_session:
                    QMessageBox.warning(self, "Warning", "Please select a session first.")
                    return
                emg_data = self.current_session
            case _:
                QMessageBox.warning(self, "Warning", "Invalid level for append/replace operation.")
                return

        from monstim_gui.dialogs.latency import AppendReplaceLatencyWindowDialog

        dialog = AppendReplaceLatencyWindowDialog(emg_data, self)
        dialog.exec()

    def invert_channel_polarity(self, level: str):
        logging.debug("Inverting channel polarity.")

        match level:  # Check the level of the channel polarity inversion.
            case "experiment":
                if not self.current_experiment:
                    QMessageBox.warning(self, "Warning", "Please select an experiment first.")
                    return
                else:
                    dialog = InvertChannelPolarityDialog(self.current_experiment, self)
            case "dataset":
                if not self.current_dataset:
                    QMessageBox.warning(self, "Warning", "Please load a dataset first.")
                    return
                else:
                    dialog = InvertChannelPolarityDialog(self.current_dataset, self)
            case "session":
                if not self.current_session:
                    QMessageBox.warning(self, "Warning", "Please select a session first.")
                    return
                else:
                    dialog = InvertChannelPolarityDialog(self.current_session, self)
            case _:
                QMessageBox.warning(self, "Warning", "Invalid level for inverting channel polarity.")
                return

        try:
            if dialog.exec():  # Show the dialog and wait for the user's response
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
                channel_indexes_to_invert = dialog.get_selected_channel_indexes()
                if not channel_indexes_to_invert:
                    QMessageBox.warning(self, "Warning", "Please select at least one channel to invert.")
                    return
                else:
                    command = InvertChannelPolarityCommand(self, level, channel_indexes_to_invert)
                    self.command_invoker.execute(command)
                    self.status_bar.showMessage("Channel polarity inverted successfully.", 5000)
            else:
                QMessageBox.warning(self, "Warning", "Please load a dataset first.")
        finally:
            QApplication.restoreOverrideCursor()

    def change_channel_names(self, level: str):
        logging.debug("Changing channel names.")

        match level:  # Check the level of the channel name change and set the channel names accordingly.
            case "experiment":
                if not self.current_experiment:
                    QMessageBox.warning(self, "Warning", "Please select an experiment first.")
                    return
                else:
                    self.channel_names = self.current_experiment.channel_names
            case "dataset":
                if not self.current_dataset:
                    QMessageBox.warning(self, "Warning", "Please load a dataset first.")
                    return
                else:
                    self.channel_names = self.current_dataset.channel_names
            case "session":
                if not self.current_session:
                    QMessageBox.warning(self, "Warning", "Please select a session first.")
                    return
                else:
                    self.channel_names = self.current_session.channel_names
            case _:
                QMessageBox.warning(self, "Warning", "Invalid level for changing channel names.")
                return

        # Open dialog to change channel names
        dialog = ChangeChannelNamesDialog(self.channel_names, self)
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
                new_names = dialog.get_new_names()
                if new_names and any(old != new for old, new in new_names.items()):
                    # Only execute command if there are actual changes
                    command = ChangeChannelNamesCommand(self, level, new_names)
                    self.command_invoker.execute(command)
                    self.status_bar.showMessage("Channel names updated successfully.", 5000)  # Show message for 5 seconds
                    logging.debug("Channel names updated successfully.")
                else:
                    self.status_bar.showMessage("No changes made to channel names.", 5000)  # Show message for 5 seconds
                    logging.debug("No changes made to channel names.")
        finally:
            QApplication.restoreOverrideCursor()

    def show_about_screen(self):
        dialog = AboutDialog(self)
        dialog.show()

    def show_help_dialog(self, topic=None):
        """Show help dialog using HelpFileRepository."""
        file = "readme.md" if topic is None else topic
        markdown_content = self.help_repo.read_help_file(file)
        from monstim_gui.dialogs.help_about import create_help_window

        self.help_window = create_help_window(markdown_content, title=topic, parent=self)
        self.help_window.show()

    def _get_effective_config(self):
        """Get the effective config including active profile data."""
        config = self.config_repo.read_config()
        if self.active_profile_data:
            data = self.active_profile_data
            if "analysis_parameters" in data:
                config.update(data["analysis_parameters"])
            if "latency_window_preset" in data:
                config["latency_window_preset"] = data["latency_window_preset"]
            if "stimuli_to_plot" in data:
                config["stimuli_to_plot"] = data["stimuli_to_plot"]
        return config

    def update_domain_configs(self, config=None):
        """Propagate the current config to all loaded domain objects."""
        if config is None:
            config = self._get_effective_config()
        if self.current_experiment:
            self.current_experiment.set_config(config)
        if self.current_dataset:
            self.current_dataset.set_config(config)
        if self.current_session:
            self.current_session.set_config(config)

    def set_current_experiment(self, experiment: Experiment | None):
        """Set the current experiment and ensure config is injected."""
        if experiment is None:
            self.current_experiment = None
            return
        config = self._get_effective_config()
        experiment.set_config(config)
        self.current_experiment = experiment

    def set_current_dataset(self, dataset: Dataset | None):
        """Set the current dataset and ensure config is injected."""
        if dataset is None:
            self.current_dataset = None
            return
        config = self._get_effective_config()
        dataset.set_config(config)
        self.current_dataset = dataset

    def set_current_session(self, session: Session | None):
        """Set the current session and ensure config is injected."""
        if session is None:
            self.current_session = None
            return
        config = self._get_effective_config()
        session.set_config(config)
        self.current_session = session

    def _cleanup_on_close(self):
        """Cleanup resources before closing.

        Saves the window state and attempts to clear the math cache.
        Any errors while clearing the cache are logged at info level.
        """
        from monstim_gui.core.ui_config import ui_config
        from monstim_gui.dialogs import clear_math_cache

        # Stop any running background threads gracefully
        logging.debug("Stopping background threads...")

        # Stop loading thread if running
        if hasattr(self.data_manager, "loading_thread") and self.data_manager.loading_thread.isRunning():
            logging.info("Stopping experiment loading thread...")
            self.data_manager.loading_thread.request_cancel()
            if not self.data_manager.loading_thread.wait(3000):
                logging.warning("Loading thread did not stop in time, forcing termination")
                self.data_manager.loading_thread.terminate()
                self.data_manager.loading_thread.wait()

        # Stop import threads if running
        if (
            hasattr(self.data_manager, "thread")
            and hasattr(self.data_manager.thread, "isRunning")
            and self.data_manager.thread.isRunning()
        ):
            logging.info("Stopping import thread...")
            self.data_manager.thread.cancel()
            if not self.data_manager.thread.wait(3000):
                logging.warning("Import thread did not stop in time")

        if (
            hasattr(self.data_manager, "multi_thread")
            and hasattr(self.data_manager.multi_thread, "isRunning")
            and self.data_manager.multi_thread.isRunning()
        ):
            logging.info("Stopping multi-import thread...")
            self.data_manager.multi_thread.cancel()
            if not self.data_manager.multi_thread.wait(3000):
                logging.warning("Multi-import thread did not stop in time")

        # Stop migration threads if running
        if hasattr(self, "_migration_runner") and self._migration_runner.isRunning():
            logging.info("Stopping migration runner...")
            self._migration_runner.request_cancel()
            if not self._migration_runner.wait(2000):
                logging.warning("Migration runner did not stop in time")

        if hasattr(self, "_migration_rescan") and self._migration_rescan.isRunning():
            logging.debug("Stopping migration rescan...")
            self._migration_rescan.request_cancel()
            if not self._migration_rescan.wait(2000):
                logging.warning("Migration rescan did not stop in time")

        logging.debug("All background threads stopped.")

        ui_config.save_window_state(self)
        try:
            clear_math_cache()
        except Exception as e:
            logging.info(f"Cache clear failed: {e}")

    def closeEvent(self, event):
        """Handle application close event - save current session state."""
        from monstim_gui.core.application_state import app_state

        try:
            # Save current window state before closing
            app_state.save_last_profile(self.profile_selector_combo.currentText())

            if self.current_experiment:
                profile_name = self.profile_selector_combo.currentText() if hasattr(self, "profile_selector_combo") else None
                app_state.save_current_session_state(
                    experiment_id=self.current_experiment.id,
                    dataset_id=(self.current_dataset.id if self.current_dataset else None),
                    session_id=(self.current_session.id if self.current_session else None),
                    profile_name=profile_name,
                )
                logging.info("Session state saved on application close")

            # All data changes auto-save via command pattern, so just cleanup and close
            self._cleanup_on_close()
            event.accept()

        except Exception as e:
            logging.error(f"Error during application close: {e}")
            # Still save window state and allow closing even if other operations fail
            try:
                self._cleanup_on_close()
            except Exception as e2:
                logging.error(f"Error saving window state: {e2}")
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MonstimGUI()
    gui.show()
    sys.exit(app.exec())
