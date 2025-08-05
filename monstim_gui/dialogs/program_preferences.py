"""
Program Preferences Dialog
Allows users to control application behavior and data tracking preferences.
"""

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from monstim_gui.core.application_state import ApplicationState


class ProgramPreferencesDialog(QDialog):
    preferences_changed = pyqtSignal()  # Signal emitted when preferences change

    def __init__(self, parent=None):
        super().__init__(parent)
        self.app_state = ApplicationState()
        self.setup_ui()
        self.load_current_preferences()

    def setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Program Preferences")
        self.setModal(True)
        self.resize(450, 350)

        # Main layout
        main_layout = QVBoxLayout(self)

        # Create preference groups
        main_layout.addWidget(self.create_tracking_group())
        main_layout.addWidget(self.create_data_management_group())

        # Button layout
        button_layout = QHBoxLayout()

        # Reset to defaults button
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.reset_button)

        button_layout.addStretch()

        # OK and Cancel buttons
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        main_layout.addLayout(button_layout)

    def create_tracking_group(self):
        """Create the data tracking preferences group."""
        group = QGroupBox("Data Tracking Preferences")
        layout = QFormLayout(group)

        # Session restoration tracking
        self.session_tracking_checkbox = QCheckBox()
        self.session_tracking_checkbox.setToolTip(
            "Remember and restore the last opened experiment, dataset, and session on startup"
        )
        layout.addRow("Track session restoration:", self.session_tracking_checkbox)

        # Analysis profiles tracking
        self.profile_tracking_checkbox = QCheckBox()
        self.profile_tracking_checkbox.setToolTip("Remember the last selected analysis profile")
        layout.addRow("Track analysis profiles:", self.profile_tracking_checkbox)

        # Import/Export paths tracking
        self.path_tracking_checkbox = QCheckBox()
        self.path_tracking_checkbox.setToolTip("Remember the last used folders for importing and exporting data")
        layout.addRow("Track import/export paths:", self.path_tracking_checkbox)

        # Recent files tracking
        self.recent_files_checkbox = QCheckBox()
        self.recent_files_checkbox.setToolTip("Maintain a list of recently accessed files")
        layout.addRow("Track recent files:", self.recent_files_checkbox)

        return group

    def create_data_management_group(self):
        """Create the data management group."""
        group = QGroupBox("Data Management")
        layout = QVBoxLayout(group)

        # Information label
        info_label = QLabel("Clear all saved user preferences and tracking data:")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Clear all data button
        self.clear_data_button = QPushButton("Clear All Saved Data")
        self.clear_data_button.setToolTip("Remove all saved session data, recent files, paths, and preferences")
        self.clear_data_button.clicked.connect(self.clear_all_data)
        layout.addWidget(self.clear_data_button)

        return group

    def load_current_preferences(self):
        """Load current preference values into the UI."""
        self.session_tracking_checkbox.setChecked(self.app_state.should_track_session_restoration())
        self.profile_tracking_checkbox.setChecked(self.app_state.should_track_analysis_profiles())
        self.path_tracking_checkbox.setChecked(self.app_state.should_track_import_export_paths())
        self.recent_files_checkbox.setChecked(self.app_state.should_track_recent_files())

    def save_preferences(self):
        """Save the current preference settings."""
        self.app_state.set_preference("track_session_restoration", self.session_tracking_checkbox.isChecked())
        self.app_state.set_preference("track_analysis_profiles", self.profile_tracking_checkbox.isChecked())
        self.app_state.set_preference("track_import_export_paths", self.path_tracking_checkbox.isChecked())
        self.app_state.set_preference("track_recent_files", self.recent_files_checkbox.isChecked())

        logging.info("Program preferences saved")
        self.preferences_changed.emit()

    def reset_to_defaults(self):
        """Reset all preferences to their default values."""
        reply = QMessageBox.question(
            self,
            "Reset Preferences",
            "Reset all program preferences to their default values?\n\n"
            "This will not clear any saved data, only the preference settings.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Set all checkboxes to default (True)
            self.session_tracking_checkbox.setChecked(True)
            self.profile_tracking_checkbox.setChecked(True)
            self.path_tracking_checkbox.setChecked(True)
            self.recent_files_checkbox.setChecked(True)

            logging.info("Program preferences reset to defaults")

    def clear_all_data(self):
        """Clear all saved tracking data."""
        reply = QMessageBox.question(
            self,
            "Clear All Saved Data",
            "This will permanently delete all saved data including:\n\n"
            "• Last session (experiment/dataset/session)\n"
            "• Analysis profile selections\n"
            "• Import/export folder paths\n"
            "• Recent files list\n"
            "• All other tracking data\n\n"
            "Are you sure you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.app_state.clear_all_tracked_data()
                QMessageBox.information(
                    self,
                    "Data Cleared",
                    "All saved tracking data has been cleared successfully.",
                )
                logging.info("All tracked user data cleared via preferences dialog")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while clearing data:\n{str(e)}")
                logging.error(f"Error clearing tracked data: {e}")

    def accept(self):
        """Accept the dialog and save preferences."""
        self.save_preferences()
        super().accept()

    def reject(self):
        """Reject the dialog without saving preferences."""
        super().reject()
