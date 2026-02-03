"""
Settings Dialog
Unified settings dialog for all application settings including display, UI scaling,
performance, and data tracking settings.
"""

import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from monstim_gui.core.application_state import ApplicationState
from monstim_gui.core.ui_config import ui_config
from monstim_gui.core.ui_scaling import ui_scaling


class ProgramSettingsDialog(QDialog):
    settings_changed = Signal()  # Signal emitted when settings change

    def __init__(self, parent=None):
        super().__init__(parent)
        self.app_state = ApplicationState()
        self._original_opengl_setting = None  # Track original OpenGL setting for restart warning
        self.setup_ui()
        self.load_current_settings()

    def setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(500, 600)

        # Main layout
        main_layout = QVBoxLayout(self)

        # Create setting groups in logical order
        main_layout.addWidget(self.create_display_group())
        main_layout.addWidget(self.create_ui_scaling_group())
        main_layout.addWidget(self.create_performance_group())
        main_layout.addWidget(self.create_tracking_group())
        main_layout.addWidget(self.create_data_management_group())

        # Button layout
        button_layout = QHBoxLayout()

        # Reset to defaults button
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.setToolTip(
            "Reset all settings to their default values. " "This will not clear any saved data, only the settings themselves."
        )
        self.reset_button.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.reset_button)

        button_layout.addStretch()

        # OK and Cancel buttons
        self.ok_button = QPushButton("OK")
        self.ok_button.setToolTip("Save all changed settings and close the dialog")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setToolTip("Discard all changes and close the dialog without saving")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        main_layout.addLayout(button_layout)

    def create_tracking_group(self):
        """Create the data tracking settings group."""
        group = QGroupBox("Data Tracking")
        layout = QFormLayout(group)

        # Session restoration tracking
        self.session_tracking_checkbox = QCheckBox()
        self.session_tracking_checkbox.setToolTip(
            "When enabled, automatically restores the last opened experiment, dataset, and session when the application starts. "
            "This provides seamless continuation of your work. Disable for enhanced privacy or to always start fresh."
        )
        layout.addRow("Track session restoration:", self.session_tracking_checkbox)

        # Analysis profiles tracking
        self.profile_tracking_checkbox = QCheckBox()
        self.profile_tracking_checkbox.setToolTip(
            "Remember the last selected analysis profile and automatically apply it when the application starts. "
            "Helps maintain consistent analysis parameters across sessions."
        )
        layout.addRow("Track analysis profiles:", self.profile_tracking_checkbox)

        # Import/Export paths tracking
        self.path_tracking_checkbox = QCheckBox()
        self.path_tracking_checkbox.setToolTip(
            "Remember the last used folders for importing data and exporting results. "
            "Saves time by opening file dialogs to your most recently used directories."
        )
        layout.addRow("Track import/export paths:", self.path_tracking_checkbox)

        # Recent files tracking
        self.recent_files_checkbox = QCheckBox()
        self.recent_files_checkbox.setToolTip(
            "Maintain a list of recently accessed experiments and files for quick access. "
            "Provides convenient shortcuts to your most frequently used datasets."
        )
        layout.addRow("Track recent files:", self.recent_files_checkbox)

        return group

    def create_ui_scaling_group(self):
        """Create the UI scaling settings group."""
        group = QGroupBox("UI Scaling")
        layout = QFormLayout(group)

        # Auto scale checkbox
        self.auto_scale_checkbox = QCheckBox()
        self.auto_scale_checkbox.setToolTip(
            "Automatically scale the user interface based on your screen's DPI and resolution. "
            "Recommended for most users. Disable only if you prefer manual control over interface scaling."
        )
        self.auto_scale_checkbox.toggled.connect(self._on_auto_scale_toggled)
        layout.addRow("Auto Scale:", self.auto_scale_checkbox)

        # Manual scale factor
        self.manual_scale_spinbox = QDoubleSpinBox()
        self.manual_scale_spinbox.setRange(0.5, 3.0)
        self.manual_scale_spinbox.setSingleStep(0.1)
        self.manual_scale_spinbox.setDecimals(1)
        self.manual_scale_spinbox.setSuffix("x")
        self.manual_scale_spinbox.setToolTip(
            "Manual scaling factor applied to the entire interface (0.5x - 3.0x). "
            "Only used when Auto Scale is disabled. 1.0x = normal size, values >1.0x make interface larger, <1.0x make it smaller."
        )
        self.manual_scale_spinbox.valueChanged.connect(self._update_current_scale)
        layout.addRow("Manual Scale Factor:", self.manual_scale_spinbox)

        # Current scale info
        self.current_scale_label = QLabel()
        layout.addRow("Current Scale Factor:", self.current_scale_label)

        # Base font size
        self.base_font_spinbox = QSpinBox()
        self.base_font_spinbox.setRange(6, 16)
        self.base_font_spinbox.setSuffix(" pt")
        self.base_font_spinbox.setToolTip(
            "Base font size for the application interface (6-16 points). "
            "Larger sizes improve readability on high-resolution displays or for accessibility. "
            "May require restart to fully apply to all interface elements."
        )
        layout.addRow("Base Font Size:", self.base_font_spinbox)

        # Max font scale
        self.max_font_scale_spinbox = QDoubleSpinBox()
        self.max_font_scale_spinbox.setRange(1.0, 2.0)
        self.max_font_scale_spinbox.setSingleStep(0.1)
        self.max_font_scale_spinbox.setDecimals(1)
        self.max_font_scale_spinbox.setSuffix("x")
        self.max_font_scale_spinbox.setToolTip(
            "Maximum scaling factor applied to fonts on high-DPI displays (1.0x - 2.0x). "
            "Prevents fonts from becoming too large on very high resolution screens while maintaining readability."
        )
        layout.addRow("Max Font Scale:", self.max_font_scale_spinbox)

        return group

    def _on_auto_scale_toggled(self, checked):
        """Handle auto scale checkbox toggle."""
        self.manual_scale_spinbox.setEnabled(not checked)
        self._update_current_scale()

    def _update_current_scale(self):
        """Update the current scale factor display."""
        if hasattr(self, "auto_scale_checkbox") and hasattr(self, "current_scale_label"):
            if self.auto_scale_checkbox.isChecked():
                scale = ui_scaling.scale_factor
                self.current_scale_label.setText(f"{scale:.1f}x (auto)")
            else:
                scale = self.manual_scale_spinbox.value()
                self.current_scale_label.setText(f"{scale:.1f}x (manual)")

    def _on_parallel_toggled(self, checked: bool):
        """When parallel loading is enabled, ensure lazy-open is also enabled."""
        if checked:
            # If user enables parallel loading, force lazy open on.
            if hasattr(self, "lazy_open_checkbox") and not self.lazy_open_checkbox.isChecked():
                self.lazy_open_checkbox.setChecked(True)

    def _on_lazy_toggled(self, checked: bool):
        """If lazy-open is turned off, disable/turn off parallel loading to keep them consistent."""
        if not checked:
            if hasattr(self, "parallel_load_checkbox") and self.parallel_load_checkbox.isChecked():
                # Turn off parallel loading if lazy open is disabled
                self.parallel_load_checkbox.setChecked(False)

    def create_performance_group(self):
        """Create the performance settings group."""
        group = QGroupBox("Performance Settings")
        layout = QFormLayout(group)

        # OpenGL acceleration
        self.opengl_checkbox = QCheckBox()
        self.opengl_checkbox.setToolTip(
            "Use hardware-accelerated OpenGL for plot rendering to improve performance and responsiveness. "
            "(Requires restart; may not be supported on all systems.)\n"
            "Warning: Enabling OpenGL may increase instability and trigger silent crashes, especially on Windows or with certain graphics drivers. Use at your own risk."
        )
        layout.addRow("Use OpenGL acceleration:", self.opengl_checkbox)

        # Lazy HDF5 open during experiment load
        self.lazy_open_checkbox = QCheckBox()
        self.lazy_open_checkbox.setToolTip(
            "When enabled, experiment loading avoids opening each .raw.h5 file up-front; metadata from .meta.json is used instead. "
            "This typically results in much faster initial load times for large experiments. Raw data is reopened lazily when required."
        )
        layout.addRow("Lazy open raw HDF5 files:", self.lazy_open_checkbox)

        # Parallel dataset loading
        self.parallel_load_checkbox = QCheckBox()
        self.parallel_load_checkbox.setToolTip(
            "Use multiple threads to load datasets in parallel. Defaults to the number of CPU cores minus one to leave one core free. "
            "Enabling this can speed up loading of experiments where datasets are independent (one dataset per animal)."
        )
        layout.addRow("Parallel dataset loading:", self.parallel_load_checkbox)

        # Keep parallel/lazy settings tied: parallel requires lazy_open to be True.
        # Connect signals to enforce the relationship in the UI.
        self.parallel_load_checkbox.toggled.connect(self._on_parallel_toggled)
        self.lazy_open_checkbox.toggled.connect(self._on_lazy_toggled)

        # Build index during experiment load (new)
        self.build_index_on_load_checkbox = QCheckBox()
        self.build_index_on_load_checkbox.setToolTip(
            "When enabled, the experiment index (.index.json) will be built/refreshed during load if missing or stale. "
            "Disabling avoids extra work during load; repositories will fall back to filesystem discovery."
        )
        layout.addRow("Build index during load:", self.build_index_on_load_checkbox)

        return group

    def create_display_group(self):
        """Create the display settings group."""
        group = QGroupBox("Display Settings")
        layout = QFormLayout(group)

        # Center windows setting
        self.center_windows_checkbox = QCheckBox()
        self.center_windows_checkbox.setToolTip(
            "Automatically center new windows on your screen when they open. "
            "Provides consistent window positioning across different screen sizes and resolutions."
        )
        layout.addRow("Center windows:", self.center_windows_checkbox)

        # Window max screen percent setting
        self.window_max_screen_percent_spinbox = QSpinBox()
        self.window_max_screen_percent_spinbox.setRange(50, 100)
        self.window_max_screen_percent_spinbox.setSuffix("%")
        self.window_max_screen_percent_spinbox.setToolTip(
            "Maximum percentage of screen space that application windows can occupy (50-100%). "
            "Lower values leave more space for other applications. Higher values maximize data visibility."
        )
        layout.addRow("Max window screen usage:", self.window_max_screen_percent_spinbox)

        # Combo box tooltip duration setting
        self.combo_tooltip_duration_spinbox = QSpinBox()
        self.combo_tooltip_duration_spinbox.setRange(500, 10000)
        self.combo_tooltip_duration_spinbox.setSingleStep(500)
        self.combo_tooltip_duration_spinbox.setSuffix(" ms")
        self.combo_tooltip_duration_spinbox.setToolTip(
            "How long to wait before showing help tooltips when hovering over dropdown menus (500-10000 ms). "
            "Shorter delays show help faster, longer delays reduce tooltip interruptions during navigation."
        )
        layout.addRow("Combo box tooltip duration:", self.combo_tooltip_duration_spinbox)

        # Left panel width percent
        self.panel_width_spinbox = QSpinBox()
        self.panel_width_spinbox.setRange(15, 40)
        self.panel_width_spinbox.setSuffix("%")
        self.panel_width_spinbox.setToolTip(
            "Width of the left control panel as a percentage of total window width (15-40%). "
            "Smaller values leave more space for plots, larger values provide more room for controls and data selection."
        )
        layout.addRow("Left Panel Width:", self.panel_width_spinbox)

        return group

    def create_data_management_group(self):
        """Create the data management group."""
        group = QGroupBox("Data Management")
        layout = QVBoxLayout(group)

        # Information label
        info_label = QLabel("Clear all saved user settings and tracking data:")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Clear all data button
        self.clear_data_button = QPushButton("Clear All Saved Data")
        self.clear_data_button.setToolTip(
            "Permanently remove all saved tracking data including session history, recent files, "
            "import/export paths, and user settings. This action cannot be undone. "
            "Use this to reset the application to its initial state or for privacy purposes."
        )
        self.clear_data_button.clicked.connect(self.clear_all_data)
        layout.addWidget(self.clear_data_button)

        return group

    def load_current_settings(self):
        """Load current setting values into the UI."""
        # Display settings
        self.center_windows_checkbox.setChecked(ui_config.get("center_windows", True))
        self.window_max_screen_percent_spinbox.setValue(ui_config.get("window_max_screen_percent", 80))
        self.combo_tooltip_duration_spinbox.setValue(ui_config.get("combo_tooltip_duration", 3000))

        # UI Scaling settings
        self.auto_scale_checkbox.setChecked(ui_config.get("auto_scale", True))
        self.manual_scale_spinbox.setValue(ui_config.get("manual_scale_factor", 1.0))
        self.panel_width_spinbox.setValue(ui_config.get("left_panel_preferred_width_percent", 22))
        self.base_font_spinbox.setValue(ui_config.get("base_font_size", 9))
        self.max_font_scale_spinbox.setValue(ui_config.get("max_font_scale", 1.5))
        self._on_auto_scale_toggled(self.auto_scale_checkbox.isChecked())

        # Performance settings
        opengl_enabled = self.app_state.should_use_opengl_acceleration()
        self.opengl_checkbox.setChecked(opengl_enabled)
        self._original_opengl_setting = opengl_enabled
        # New performance settings
        self.lazy_open_checkbox.setChecked(self.app_state.should_use_lazy_open_h5())
        self.parallel_load_checkbox.setChecked(self.app_state.should_use_parallel_loading())
        try:
            self.build_index_on_load_checkbox.setChecked(self.app_state.should_build_index_on_load())
        except Exception:
            self.build_index_on_load_checkbox.setChecked(True)

        # Tracking settings
        self.session_tracking_checkbox.setChecked(self.app_state.should_track_session_restoration())
        self.profile_tracking_checkbox.setChecked(self.app_state.should_track_analysis_profiles())
        self.path_tracking_checkbox.setChecked(self.app_state.should_track_import_export_paths())
        self.recent_files_checkbox.setChecked(self.app_state.should_track_recent_files())

    def save_settings(self):
        """Save the current setting values."""
        # Save display settings
        ui_config.set("center_windows", self.center_windows_checkbox.isChecked())
        ui_config.set("window_max_screen_percent", self.window_max_screen_percent_spinbox.value())
        ui_config.set("combo_tooltip_duration", self.combo_tooltip_duration_spinbox.value())

        # Save UI scaling settings
        ui_config.set("auto_scale", self.auto_scale_checkbox.isChecked())
        ui_config.set("manual_scale_factor", self.manual_scale_spinbox.value())
        ui_config.set("left_panel_preferred_width_percent", self.panel_width_spinbox.value())
        ui_config.set("base_font_size", self.base_font_spinbox.value())
        ui_config.set("max_font_scale", self.max_font_scale_spinbox.value())

        # Save performance settings
        self.app_state.set_setting("use_opengl_acceleration", self.opengl_checkbox.isChecked())
        self.app_state.set_setting("use_lazy_open_h5", self.lazy_open_checkbox.isChecked())
        self.app_state.set_setting("enable_parallel_loading", self.parallel_load_checkbox.isChecked())
        try:
            self.app_state.set_build_index_on_load(self.build_index_on_load_checkbox.isChecked())
        except Exception:
            logging.debug("Failed to save build_index_on_load setting", exc_info=True)

        # Save tracking settings
        self.app_state.set_setting("track_session_restoration", self.session_tracking_checkbox.isChecked())
        self.app_state.set_setting("track_analysis_profiles", self.profile_tracking_checkbox.isChecked())
        self.app_state.set_setting("track_import_export_paths", self.path_tracking_checkbox.isChecked())
        self.app_state.set_setting("track_recent_files", self.recent_files_checkbox.isChecked())

        # Check if settings that require restart have changed
        restart_required = False

        if self._original_opengl_setting != self.opengl_checkbox.isChecked():
            restart_required = True

        if restart_required:
            QMessageBox.information(
                self,
                "Restart Required",
                "Some settings have been changed that require a restart.\n\n"
                "Please restart the application for all changes to take effect.",
            )
        else:
            QMessageBox.information(
                self,
                "Settings Saved",
                "Settings have been saved successfully.",
            )

        logging.info("Program settings saved")
        self.settings_changed.emit()

    def reset_to_defaults(self):
        """Reset all settings to their default values."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all program settings to their default values?\n\n" "This will not clear any saved data, only the settings.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Reset display settings to defaults
            self.center_windows_checkbox.setChecked(True)
            self.window_max_screen_percent_spinbox.setValue(80)
            self.combo_tooltip_duration_spinbox.setValue(3000)

            # Reset UI scaling settings to defaults
            self.auto_scale_checkbox.setChecked(True)
            self.manual_scale_spinbox.setValue(1.0)
            self.panel_width_spinbox.setValue(22)
            self.base_font_spinbox.setValue(9)
            self.max_font_scale_spinbox.setValue(1.5)

            # Reset performance settings to defaults
            self.opengl_checkbox.setChecked(True)
            # New default: lazy open HDF5 and parallel loading enabled
            if hasattr(self, "lazy_open_checkbox"):
                self.lazy_open_checkbox.setChecked(True)
            if hasattr(self, "parallel_load_checkbox"):
                self.parallel_load_checkbox.setChecked(True)
            if hasattr(self, "build_index_on_load_checkbox"):
                self.build_index_on_load_checkbox.setChecked(True)

            # Reset tracking settings to defaults
            self.session_tracking_checkbox.setChecked(True)
            self.profile_tracking_checkbox.setChecked(True)
            self.path_tracking_checkbox.setChecked(True)
            self.recent_files_checkbox.setChecked(True)

            logging.info("Program settings reset to defaults")

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
                logging.info("All tracked user data cleared via settings dialog")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while clearing data:\n{str(e)}")
                logging.error(f"Error clearing tracked data: {e}")

    def accept(self):
        """Accept the dialog and save settings."""
        self.save_settings()
        super().accept()

    def reject(self):
        """Reject the dialog without saving settings."""
        super().reject()
