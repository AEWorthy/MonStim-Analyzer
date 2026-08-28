from typing import TYPE_CHECKING

from PySide6.QtGui import QFont, QKeySequence
from PySide6.QtWidgets import QMenu, QMenuBar, QMessageBox

if TYPE_CHECKING:
    from gui_main import MonstimGUI


class MenuBar(QMenuBar):
    def __init__(self, parent: MonstimGUI):
        super().__init__(parent)
        self.parent = parent  # type: MonstimGUI
        self.create_file_menu()
        self.create_edit_menu()
        self.create_tools_menu()
        self.create_help_menu()

    def create_file_menu(self):
        # File menu
        # Keep explicit references to the top-level menus.  Besides making
        # ownership unambiguous, this prevents their Python wrappers from
        # being collected while another top-level window is active.
        file_menu = QMenu("File", self)
        self.addMenu(file_menu)
        self.file_menu = file_menu

        import_action = file_menu.addAction("Import an Experiment")
        import_action.triggered.connect(self.parent.data_manager.import_expt_data)
        import_action.setStatusTip("Import a single experiment from disk")

        import_multiple_action = file_menu.addAction("Import Multiple Experiments")
        import_multiple_action.triggered.connect(self.parent.data_manager.import_multiple_expt_data)
        import_multiple_action.setStatusTip("Import multiple experiments from disk")

        rename_experiment_action = file_menu.addAction("Rename Current Experiment")
        rename_experiment_action.triggered.connect(self.parent.data_manager.rename_experiment)
        rename_experiment_action.setStatusTip("Change the name of the currently selected experiment")

        delete_experiment_action = file_menu.addAction("Delete Current Experiment")
        delete_experiment_action.triggered.connect(self.parent.data_manager.delete_current_experiment)
        delete_experiment_action.setStatusTip("Delete the currently selected experiment")

        file_menu.addSeparator()

        # refresh existing datasets button
        refresh_datasets_action = file_menu.addAction("Refresh Experiments list")
        refresh_datasets_action.triggered.connect(self._refresh_experiments_list)
        refresh_datasets_action.setShortcut(QKeySequence.StandardKey.Refresh)
        refresh_datasets_action.setStatusTip("Refresh the list of available experiments")

        rebuild_catalogs_action = file_menu.addAction("Force Rebuild Data Catalog")
        rebuild_catalogs_action.triggered.connect(self._force_rebuild_catalogs)
        rebuild_catalogs_action.setStatusTip("Re-scan all files in the currently selected experiment and rebuild the cached catalogs")

        file_menu.addSeparator()

        # Bulk Export Data
        bulk_export_action = file_menu.addAction("Bulk Export Data…")
        bulk_export_action.triggered.connect(self.parent.bulk_export_manager.show_bulk_export_dialog)
        bulk_export_action.setStatusTip("Export aggregated data for multiple experiments/datasets to Excel")

        file_menu.addSeparator()

        # Preferences button
        preferences_action = file_menu.addAction("Settings Center")
        preferences_action.triggered.connect(self.parent.data_manager.show_preferences_window)
        preferences_action.setStatusTip("Configure program settings, global analysis defaults, and profile overrides")

        file_menu.addSeparator()

        # Save button
        save_action = file_menu.addAction("Save Current Experiment")
        save_action.triggered.connect(self.parent.data_manager.save_experiment)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.setStatusTip("Save changes to the currently selected experiment")

        # Exit button
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.parent.close)

    def create_edit_menu(self):
        # Edit menu
        edit_menu = QMenu("Edit", self)
        self.addMenu(edit_menu)
        self.edit_menu = edit_menu

        # # Add undo and redo buttons to the menu bar
        self.undo_action = edit_menu.addAction("Undo")
        self.redo_action = edit_menu.addAction("Redo")
        self.undo_action.triggered.connect(self.parent.undo)
        self.redo_action.triggered.connect(self.parent.redo)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.undo_action.setStatusTip("Undo the most recent change")
        self.redo_action.setStatusTip("Redo the most recently undone change")

        edit_menu.addSeparator()

        # Submenus for each data level
        experiment_menu = edit_menu.addMenu("Experiment")
        dataset_menu = edit_menu.addMenu("Dataset")
        session_menu = edit_menu.addMenu("Session")

        # Data curation submenu
        curation_menu = edit_menu.addMenu("Data Curation")

        # Experiment level actions
        update_window_action = experiment_menu.addAction("Manage Latency Windows")
        update_window_action.triggered.connect(lambda: self.parent.manage_latency_windows("experiment"))
        update_window_action.setStatusTip("Create and edit latency windows for the current experiment")
        append_replace_window_action = experiment_menu.addAction("Append/Replace Latency Window(s)")
        append_replace_window_action.triggered.connect(lambda: self.parent.append_replace_latency_windows("experiment"))
        append_replace_window_action.setStatusTip("Append or replace latency windows throughout the current experiment")
        invert_polarity_action = experiment_menu.addAction("Invert Channel Polarity")
        invert_polarity_action.triggered.connect(lambda: self.parent.invert_channel_polarity("experiment"))
        invert_polarity_action.setStatusTip("Invert the signal polarity of channels in the current experiment")
        change_names_action = experiment_menu.addAction("Change Channel Names")
        change_names_action.triggered.connect(lambda: self.parent.change_channel_names("experiment"))
        change_names_action.setStatusTip("Rename channels throughout the current experiment")
        experiment_menu.addSeparator()
        reload_experiment_action = experiment_menu.addAction("Reload Current Experiment")
        reload_experiment_action.triggered.connect(self.confirm_reload_experiment)
        reload_experiment_action.setStatusTip("Restore the current experiment to its original state")
        remove_experiment_action = experiment_menu.addAction("Remove Current Experiment")
        remove_experiment_action.triggered.connect(self.parent.data_manager.delete_current_experiment)
        remove_experiment_action.setStatusTip("Remove the current experiment from the application")

        # Dataset level actions
        update_window_action = dataset_menu.addAction("Manage Latency Windows")
        update_window_action.triggered.connect(lambda: self.parent.manage_latency_windows("dataset"))
        update_window_action.setStatusTip("Create and edit latency windows for the current dataset")
        append_replace_window_action = dataset_menu.addAction("Append/Replace Latency Window(s)")
        append_replace_window_action.triggered.connect(lambda: self.parent.append_replace_latency_windows("dataset"))
        append_replace_window_action.setStatusTip("Append or replace latency windows throughout the current dataset")
        invert_polarity_action = dataset_menu.addAction("Invert Channel Polarity")
        invert_polarity_action.triggered.connect(lambda: self.parent.invert_channel_polarity("dataset"))
        invert_polarity_action.setStatusTip("Invert the signal polarity of channels in the current dataset")
        change_names_action = dataset_menu.addAction("Change Channel Names")
        change_names_action.triggered.connect(lambda: self.parent.change_channel_names("dataset"))
        change_names_action.setStatusTip("Rename channels throughout the current dataset")
        edit_metadata_action = dataset_menu.addAction("Edit Metadata")
        edit_metadata_action.triggered.connect(self.parent.data_manager.edit_dataset_metadata)
        edit_metadata_action.setStatusTip("Edit metadata for the current dataset")
        dataset_menu.addSeparator()
        reload_dataset_action = dataset_menu.addAction("Reload Current Dataset")
        reload_dataset_action.triggered.connect(self.confirm_reload_dataset)
        reload_dataset_action.setStatusTip("Restore the current dataset to its original state")
        exclude_dataset_action = dataset_menu.addAction("Exclude Current Dataset")
        exclude_dataset_action.triggered.connect(self.parent.exclude_dataset)
        exclude_dataset_action.setStatusTip("Exclude the current dataset from analysis")
        restore_dataset_action = dataset_menu.addAction("Restore Excluded Dataset")
        restore_dataset_action.triggered.connect(self.parent.prompt_restore_dataset)
        restore_dataset_action.setStatusTip("Restore a previously excluded dataset")

        # Session level actions
        update_window_action = session_menu.addAction("Manage Latency Windows")
        update_window_action.triggered.connect(lambda: self.parent.manage_latency_windows("session"))
        update_window_action.setStatusTip("Create and edit latency windows for the current session")
        append_replace_window_action = session_menu.addAction("Append/Replace Latency Window(s)")
        append_replace_window_action.triggered.connect(lambda: self.parent.append_replace_latency_windows("session"))
        append_replace_window_action.setStatusTip("Append or replace latency windows throughout the current session")
        invert_polarity_action = session_menu.addAction("Invert Channel Polarity")
        invert_polarity_action.triggered.connect(lambda: self.parent.invert_channel_polarity("session"))
        invert_polarity_action.setStatusTip("Invert the signal polarity of channels in the current session")
        change_names_action = session_menu.addAction("Change Channel Names")
        change_names_action.triggered.connect(lambda: self.parent.change_channel_names("session"))
        change_names_action.setStatusTip("Rename channels throughout the current session")
        session_menu.addSeparator()
        reload_session_action = session_menu.addAction("Reload Current Session")
        reload_session_action.triggered.connect(self.confirm_reload_session)
        reload_session_action.setStatusTip("Restore the current session to its original state")
        exclude_session_action = session_menu.addAction("Exclude Current Session")
        exclude_session_action.triggered.connect(self.parent.exclude_session)
        exclude_session_action.setStatusTip("Exclude the current session from analysis")
        restore_session_action = session_menu.addAction("Restore Excluded Session")
        restore_session_action.triggered.connect(self.parent.prompt_restore_session)
        restore_session_action.setStatusTip("Restore a previously excluded session")

        # Data curation actions
        manage_data_action = curation_menu.addAction("Manage Data...")
        manage_data_action.triggered.connect(self.show_data_curation_manager)
        manage_data_action.setStatusTip("Create, import, delete, rename experiments and organize datasets")

        curation_menu.addSeparator()

        exclude_recordings_action = curation_menu.addAction("Recording Exclusion Editor...")
        exclude_recordings_action.triggered.connect(self.show_recording_exclusion_editor)
        exclude_recordings_action.setStatusTip("Exclude recordings based on stimulus amplitude and other criteria")

    def create_help_menu(self):
        # Help menu
        help_menu = QMenu("Help", self)
        self.addMenu(help_menu)
        self.help_menu = help_menu

        # About button
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.parent.show_about_screen)

        help_menu.addSeparator()

        # Show Help button
        help_action = help_menu.addAction("Show Help")
        help_action.triggered.connect(lambda: self.parent.show_help_dialog("user/index.md"))
        help_action.setStatusTip("Open the application help documentation")

        # Show EMG processing info button
        processing_info_action = help_menu.addAction("Show EMG Processing Info")
        processing_info_action.triggered.connect(lambda: self.parent.show_help_dialog("science/emg_processing.md"))
        processing_info_action.setStatusTip("Read about the EMG processing and transformation steps")

        # Show Experiment Import Info button
        data_import_action = help_menu.addAction("Show Experiment Import Info")
        data_import_action.triggered.connect(lambda: self.parent.show_help_dialog("user/importing_experiments.md"))
        data_import_action.setStatusTip("Read instructions for importing experiments")

        help_menu.addSeparator()

        open_logs_action = help_menu.addAction("Open Log Folder")
        open_logs_action.triggered.connect(self.parent.data_manager.open_log_directory)
        open_logs_action.setStatusTip("Open the folder containing application log files")

        save_report_action = help_menu.addAction("Save Error Report")
        save_report_action.triggered.connect(self.parent.data_manager.save_error_report)
        save_report_action.setStatusTip("Save diagnostic information for troubleshooting")

    def create_tools_menu(self):
        """Create clearly separated developer/recovery operations."""
        advanced_menu = QMenu("Tools", self)
        self.addMenu(advanced_menu)
        self.tools_menu = advanced_menu
        rebuild_all_action = advanced_menu.addAction("Force Rebuild All Data Catalogs…")
        rebuild_all_action.triggered.connect(self._force_rebuild_all_catalogs)
        rebuild_all_action.setStatusTip("Re-scan every experiment and rebuild all catalogs; this may take a very long time")

    # Edit menu functions
    def confirm_reload_session(self):
        reply = QMessageBox.warning(
            self,
            "Confirm Reload",
            "Are you sure you want to restore the current session to its original state?"
            "\n\nNote: This will add back any recordings that were removed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.parent.data_manager.reload_current_session()

    def confirm_reload_dataset(self):
        reply = QMessageBox.warning(
            self,
            "Confirm Reload",
            "Are you sure you want to restore the current dataset to its original state?"
            "\n\nNote: This will add back any sessions/recordings that were removed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.parent.data_manager.reload_current_dataset()

    def confirm_reload_experiment(self):
        reply = QMessageBox.warning(
            self,
            "Confirm Reload",
            "Are you sure you want to restore the current experiment to its original state?\n\nNote: THIS ACTION IS NOT REVERSIBLE. "
            "This will add back any datasets/sessions/recordings that were removed and will completely reset any changes you made to "
            "the data contained within this experiment.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.parent.data_manager.reload_current_experiment()

    # Update functions
    def update_undo_redo_labels(self):
        undo_command_name = self.parent.command_invoker.get_undo_command_name()
        redo_command_name = self.parent.command_invoker.get_redo_command_name()

        undo_text = "Undo"
        redo_text = "Redo"

        # Add the command name as a hint if available
        if undo_command_name:
            undo_text += f" ({undo_command_name})"
        if redo_command_name:
            redo_text += f" ({redo_command_name})"

        # Set shadowed or disabled effect for the hint part
        hint_font = QFont()
        hint_font.setItalic(True)

        self.undo_action.setText(undo_text)
        self.redo_action.setText(redo_text)

        # Enable/disable actions based on availability
        self.undo_action.setEnabled(undo_command_name is not None)
        self.redo_action.setEnabled(redo_command_name is not None)

        # Set the hint font for the actions
        self.undo_action.setFont(hint_font)
        self.redo_action.setFont(hint_font)

    def show_program_preferences(self):
        """Show the program preferences dialog."""
        try:
            from monstim_gui.dialogs import ProgramSettingsDialog

            dialog = ProgramSettingsDialog(self.parent)

            # Connect the preferences changed signal to refresh any UI that might depend on it
            dialog.settings_changed.connect(self.parent.refresh_preferences_dependent_ui)

            dialog.exec()
        except ImportError as e:
            QMessageBox.warning(
                self.parent,
                "Program Preferences",
                f"Program preferences dialog is not available: {e}",
            )

    def show_recording_exclusion_editor(self):
        """Show the recording exclusion editor dialog."""
        if not self.parent.current_session:
            QMessageBox.warning(
                self.parent,
                "No Session Selected",
                "Please select a session first to exclude recordings.",
            )
            return

        try:
            from monstim_gui.dialogs.recording_exclusion_editor import RecordingExclusionEditor

            dialog = RecordingExclusionEditor(self.parent)

            # Connect the exclusions applied signal to refresh the UI
            dialog.exclusions_applied.connect(self.parent.plot_controller.plot_data)

            dialog.exec()
        except ImportError as e:
            QMessageBox.warning(
                self.parent,
                "Recording Exclusion Editor",
                f"Recording exclusion editor is not available: {e}",
            )
        except Exception as e:
            QMessageBox.critical(
                self.parent,
                "Error",
                f"Failed to open recording exclusion editor:\n{e!s}",
            )

    def _refresh_experiments_list(self):
        """Refresh the experiments list and reset all selections."""
        # Clear current state and refresh the experiments list
        self.parent.set_current_experiment(None)
        self.parent.set_current_dataset(None)
        self.parent.set_current_session(None)

        # Reset all combo boxes to placeholders (refresh() will handle filesystem rescan)
        self.parent.data_selection_widget.refresh()

        # Notify plot widget of the state change
        if hasattr(self.parent, "plot_widget") and hasattr(self.parent.plot_widget, "on_data_selection_changed"):
            self.parent.plot_widget.on_data_selection_changed()

    def _force_rebuild_catalogs(self):
        """Rebuild the active experiment catalog from authoritative files."""
        self.parent.data_manager.rebuild_current_catalog()

    def _force_rebuild_all_catalogs(self):
        """Start the expensive all-experiment catalog rebuild."""
        self.parent.data_manager.rebuild_all_catalogs()

    def show_data_curation_manager(self):
        """Show the data curation manager dialog."""
        if not self.parent.expts_dict_keys:
            QMessageBox.warning(
                self.parent,
                "No Experiments Available",
                "No experiments are currently loaded. Please import an experiment first.",
            )
            return

        try:
            from monstim_gui.dialogs.data_curation_manager import DataCurationManager

            dialog = DataCurationManager(self.parent)

            # Execute the dialog and refresh UI only after it closes if changes were made
            dialog.exec()

            # The dialog is modal, so main UI is frozen during execution
            # Only refresh after dialog closes if changes were made
            if hasattr(dialog, "_changes_made") and dialog._changes_made:
                # Refresh experiments list and reset all selections after structural changes
                self.parent.data_manager.unpack_existing_experiments()

                # If an experiment is currently loaded, reload it to pick up annotation changes
                # (e.g., excluded datasets list)
                if self.parent.current_experiment:
                    self.parent.data_manager.reload_current_experiment()
                else:
                    # Otherwise just update the UI
                    self.parent.data_selection_widget.update()
        except ImportError as e:
            QMessageBox.warning(
                self.parent,
                "Data Curation Manager",
                f"Data curation manager is not available: {e}",
            )
        except Exception as e:
            QMessageBox.critical(
                self.parent,
                "Error",
                f"Failed to open data curation manager:\n{e!s}",
            )
