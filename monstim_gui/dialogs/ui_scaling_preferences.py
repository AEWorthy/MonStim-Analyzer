"""
UI scaling preferences dialog.
"""

from PyQt6.QtWidgets import (
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

from ..core.ui_config import ui_config
from ..core.ui_scaling import ui_scaling


class UIScalingPreferencesDialog(QDialog):
    """Dialog for configuring UI scaling preferences."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("UI Scaling Preferences")
        self.setModal(True)
        self.resize(400, 300)

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Scaling group
        scaling_group = QGroupBox("Scaling Settings")
        scaling_layout = QFormLayout(scaling_group)

        # Auto scale checkbox
        self.auto_scale_cb = QCheckBox()
        self.auto_scale_cb.setToolTip(
            "Automatically scale UI based on screen DPI and resolution"
        )
        self.auto_scale_cb.toggled.connect(self._on_auto_scale_toggled)
        scaling_layout.addRow("Auto Scale:", self.auto_scale_cb)

        # Manual scale factor
        self.manual_scale_spin = QDoubleSpinBox()
        self.manual_scale_spin.setRange(0.5, 3.0)
        self.manual_scale_spin.setSingleStep(0.1)
        self.manual_scale_spin.setDecimals(1)
        self.manual_scale_spin.setSuffix("x")
        self.manual_scale_spin.setToolTip(
            "Manual scaling factor when auto-scale is disabled"
        )
        scaling_layout.addRow("Manual Scale Factor:", self.manual_scale_spin)

        # Current scale info
        self.current_scale_label = QLabel()
        scaling_layout.addRow("Current Scale Factor:", self.current_scale_label)

        layout.addWidget(scaling_group)

        # Panel sizing group
        panel_group = QGroupBox("Panel Sizing")
        panel_layout = QFormLayout(panel_group)

        # Left panel width percent
        self.panel_width_spin = QSpinBox()
        self.panel_width_spin.setRange(15, 40)
        self.panel_width_spin.setSuffix("%")
        self.panel_width_spin.setToolTip(
            "Preferred width of left panel as percentage of screen width"
        )
        panel_layout.addRow("Left Panel Width:", self.panel_width_spin)

        layout.addWidget(panel_group)

        # Font group
        font_group = QGroupBox("Font Settings")
        font_layout = QFormLayout(font_group)

        # Base font size
        self.base_font_spin = QSpinBox()
        self.base_font_spin.setRange(6, 16)
        self.base_font_spin.setSuffix(" pt")
        self.base_font_spin.setToolTip("Base font size for the application")
        font_layout.addRow("Base Font Size:", self.base_font_spin)

        # Max font scale
        self.max_font_scale_spin = QDoubleSpinBox()
        self.max_font_scale_spin.setRange(1.0, 2.0)
        self.max_font_scale_spin.setSingleStep(0.1)
        self.max_font_scale_spin.setDecimals(1)
        self.max_font_scale_spin.setSuffix("x")
        self.max_font_scale_spin.setToolTip("Maximum font scaling factor")
        font_layout.addRow("Max Font Scale:", self.max_font_scale_spin)

        layout.addWidget(font_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self._reset_defaults)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        # Update current scale display
        self._update_current_scale()

    def _on_auto_scale_toggled(self, checked):
        """Handle auto scale checkbox toggle."""
        self.manual_scale_spin.setEnabled(not checked)
        self._update_current_scale()

    def _update_current_scale(self):
        """Update the current scale factor display."""
        if self.auto_scale_cb.isChecked():
            scale = ui_scaling.scale_factor
            self.current_scale_label.setText(f"{scale:.1f}x (auto)")
        else:
            scale = self.manual_scale_spin.value()
            self.current_scale_label.setText(f"{scale:.1f}x (manual)")

    def _load_settings(self):
        """Load current settings into the dialog."""
        self.auto_scale_cb.setChecked(ui_config.get("auto_scale", True))
        self.manual_scale_spin.setValue(ui_config.get("manual_scale_factor", 1.0))
        self.panel_width_spin.setValue(
            ui_config.get("left_panel_preferred_width_percent", 22)
        )
        self.base_font_spin.setValue(ui_config.get("base_font_size", 9))
        self.max_font_scale_spin.setValue(ui_config.get("max_font_scale", 1.5))

        self._on_auto_scale_toggled(self.auto_scale_cb.isChecked())

    def _reset_defaults(self):
        """Reset all settings to defaults."""
        result = QMessageBox.question(
            self,
            "Reset to Defaults",
            "Are you sure you want to reset all UI scaling settings to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if result == QMessageBox.StandardButton.Yes:
            self.auto_scale_cb.setChecked(True)
            self.manual_scale_spin.setValue(1.0)
            self.panel_width_spin.setValue(22)
            self.base_font_spin.setValue(9)
            self.max_font_scale_spin.setValue(1.5)

    def accept(self):
        """Save settings and close the dialog."""
        # Save all settings
        ui_config.set("auto_scale", self.auto_scale_cb.isChecked())
        ui_config.set("manual_scale_factor", self.manual_scale_spin.value())
        ui_config.set(
            "left_panel_preferred_width_percent", self.panel_width_spin.value()
        )
        ui_config.set("base_font_size", self.base_font_spin.value())
        ui_config.set("max_font_scale", self.max_font_scale_spin.value())

        # Show restart message
        QMessageBox.information(
            self,
            "Settings Saved",
            "UI scaling settings have been saved. Please restart the application for all changes to take effect.",
            QMessageBox.StandardButton.Ok,
        )

        super().accept()
