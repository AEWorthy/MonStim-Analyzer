from typing import TYPE_CHECKING

from PyQt6.QtGui import QFont, QKeySequence
from PyQt6.QtWidgets import QMenuBar, QMessageBox

if TYPE_CHECKING:
    from gui_main import MonstimGUI


class MenuBar(QMenuBar):
    def __init__(self, parent: "MonstimGUI"):
        super().__init__(parent)
        self.parent = parent  # type: MonstimGUI
        self.create_file_menu()
        self.create_edit_menu()
        self.create_help_menu()

    def create_file_menu(self):
        # File menu
        file_menu = self.addMenu("File")

        import_action = file_menu.addAction("Import an Experiment")
        import_action.triggered.connect(self.parent.data_manager.import_expt_data)

        import_multiple_action = file_menu.addAction("Import Multiple Experiments")
        import_multiple_action.triggered.connect(self.parent.data_manager.import_multiple_expt_data)

        rename_experiment_action = file_menu.addAction("Rename Current Experiment")
        rename_experiment_action.triggered.connect(self.parent.data_manager.rename_experiment)

        delete_experiment_action = file_menu.addAction("Delete Current Experiment")
        delete_experiment_action.triggered.connect(self.parent.data_manager.delete_experiment)

        file_menu.addSeparator()

        # refresh existing datasets button
        refresh_datasets_action = file_menu.addAction("Refresh Experiments List")
        refresh_datasets_action.triggered.connect(self.parent.data_manager.refresh_existing_experiments)
        refresh_datasets_action.setShortcut(QKeySequence.StandardKey.Refresh)

        file_menu.addSeparator()

        # Preferences button
        preferences_action = file_menu.addAction("Preferences")
        preferences_action.triggered.connect(self.parent.data_manager.show_preferences_window)

        # UI Scaling Preferences button
        ui_scaling_action = file_menu.addAction("Display Preferences")
        ui_scaling_action.triggered.connect(self.show_ui_scaling_preferences)

        # Program Preferences button
        program_prefs_action = file_menu.addAction("Program Preferences")
        program_prefs_action.triggered.connect(self.show_program_preferences)

        file_menu.addSeparator()

        # Save button
        save_action = file_menu.addAction("Save Current Experiment")
        save_action.triggered.connect(self.parent.data_manager.save_experiment)
        save_action.setShortcut(QKeySequence.StandardKey.Save)

        # Save As button
        # save_as_action = file_menu.addAction("Save Current Experiment As")
        # save_as_action.triggered.connect(self.parent.save_experiment_as)

        # Exit button
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.parent.close)

    def create_edit_menu(self):
        # Edit menu
        edit_menu = self.addMenu("Edit")

        # # Add undo and redo buttons to the menu bar
        self.undo_action = edit_menu.addAction("Undo")
        self.redo_action = edit_menu.addAction("Redo")
        self.undo_action.triggered.connect(self.parent.undo)
        self.redo_action.triggered.connect(self.parent.redo)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)

        edit_menu.addSeparator()

        # Submenus for each data level
        experiment_menu = edit_menu.addMenu("Experiment")
        dataset_menu = edit_menu.addMenu("Dataset")
        session_menu = edit_menu.addMenu("Session")

        # Experiment level actions
        update_window_action = experiment_menu.addAction("Manage Latency Windows")
        update_window_action.triggered.connect(lambda: self.parent.manage_latency_windows("experiment"))
        invert_polarity_action = experiment_menu.addAction("Invert Channel Polarity")
        invert_polarity_action.triggered.connect(lambda: self.parent.invert_channel_polarity("experiment"))
        change_names_action = experiment_menu.addAction("Change Channel Names")
        change_names_action.triggered.connect(lambda: self.parent.change_channel_names("experiment"))
        experiment_menu.addSeparator()
        reload_experiment_action = experiment_menu.addAction("Reload Current Experiment")
        reload_experiment_action.triggered.connect(self.confirm_reload_experiment)
        remove_experiment_action = experiment_menu.addAction("Remove Current Experiment")
        remove_experiment_action.triggered.connect(self.parent.data_manager.delete_experiment)

        # Dataset level actions
        update_window_action = dataset_menu.addAction("Manage Latency Windows")
        update_window_action.triggered.connect(lambda: self.parent.manage_latency_windows("dataset"))
        invert_polarity_action = dataset_menu.addAction("Invert Channel Polarity")
        invert_polarity_action.triggered.connect(lambda: self.parent.invert_channel_polarity("dataset"))
        change_names_action = dataset_menu.addAction("Change Channel Names")
        change_names_action.triggered.connect(lambda: self.parent.change_channel_names("dataset"))
        dataset_menu.addSeparator()
        reload_dataset_action = dataset_menu.addAction("Reload Current Dataset")
        reload_dataset_action.triggered.connect(self.confirm_reload_dataset)
        exclude_dataset_action = dataset_menu.addAction("Exclude Current Dataset")
        exclude_dataset_action.triggered.connect(self.parent.exclude_dataset)
        restore_dataset_action = dataset_menu.addAction("Restore Excluded Dataset")
        restore_dataset_action.triggered.connect(self.parent.prompt_restore_dataset)

        # Session level actions
        update_window_action = session_menu.addAction("Manage Latency Windows")
        update_window_action.triggered.connect(lambda: self.parent.manage_latency_windows("session"))
        invert_polarity_action = session_menu.addAction("Invert Channel Polarity")
        invert_polarity_action.triggered.connect(lambda: self.parent.invert_channel_polarity("session"))
        change_names_action = session_menu.addAction("Change Channel Names")
        change_names_action.triggered.connect(lambda: self.parent.change_channel_names("session"))
        session_menu.addSeparator()
        reload_session_action = session_menu.addAction("Reload Current Session")
        reload_session_action.triggered.connect(self.confirm_reload_session)
        exclude_session_action = session_menu.addAction("Exclude Current Session")
        exclude_session_action.triggered.connect(self.parent.exclude_session)
        restore_session_action = session_menu.addAction("Restore Excluded Session")
        restore_session_action.triggered.connect(self.parent.prompt_restore_session)

    def create_help_menu(self):
        # Help menu
        help_menu = self.addMenu("Help")

        # About button
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.parent.show_about_screen)

        help_menu.addSeparator()

        # Show Help button
        help_action = help_menu.addAction("Show Help")
        help_action.triggered.connect(lambda: self.parent.show_help_dialog("readme.md"))

        # Show EMG processing info button
        processing_info_action = help_menu.addAction("Show EMG Processing Info")
        processing_info_action.triggered.connect(lambda: self.parent.show_help_dialog("Transform_EMG.md", latex=True))

        # Show Experiment Import Info button
        data_import_action = help_menu.addAction("Show Experiment Import Info")
        data_import_action.triggered.connect(lambda: self.parent.show_help_dialog("multi_experiment_import.md"))

        help_menu.addSeparator()

        open_logs_action = help_menu.addAction("Open Log Folder")
        open_logs_action.triggered.connect(self.parent.data_manager.open_log_directory)

        save_report_action = help_menu.addAction("Save Error Report")
        save_report_action.triggered.connect(self.parent.data_manager.save_error_report)

    # Edit menu functions
    def confirm_reload_session(self):
        reply = QMessageBox.warning(
            self,
            "Confirm Reload",
            "Are you sure you want to restore the current session to its original state?\n\nNote: This will add back any recordings that were removed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.parent.data_manager.reload_current_session()

    def confirm_reload_dataset(self):
        reply = QMessageBox.warning(
            self,
            "Confirm Reload",
            "Are you sure you want to restore the current dataset to its original state?\n\nNote: This will add back any sessions/recordings that were removed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.parent.data_manager.reload_current_dataset()

    def confirm_reload_experiment(self):
        reply = QMessageBox.warning(
            self,
            "Confirm Reload",
            "Are you sure you want to restore the current experiment to its original state?\n\nNote: THIS ACTION IS NOT REVERSIBLE. This will add back any datasets/sessions/recordings that were removed and will completely reset any changes you made to the data contained within this experiment.",
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

    def show_ui_scaling_preferences(self):
        """Show the UI scaling preferences dialog."""
        try:
            from monstim_gui.dialogs.ui_scaling_preferences import (
                UIScalingPreferencesDialog,
            )

            dialog = UIScalingPreferencesDialog(self.parent)
            dialog.exec()
        except ImportError as e:
            QMessageBox.warning(
                self.parent,
                "UI Scaling Preferences",
                f"UI scaling preferences dialog is not available: {e}",
            )

    def show_program_preferences(self):
        """Show the program preferences dialog."""
        try:
            from monstim_gui.dialogs.program_preferences import ProgramPreferencesDialog

            dialog = ProgramPreferencesDialog(self.parent)

            # Connect the preferences changed signal to refresh any UI that might depend on it
            dialog.preferences_changed.connect(self.parent.refresh_preferences_dependent_ui)

            dialog.exec()
        except ImportError as e:
            QMessageBox.warning(
                self.parent,
                "Program Preferences",
                f"Program preferences dialog is not available: {e}",
            )
