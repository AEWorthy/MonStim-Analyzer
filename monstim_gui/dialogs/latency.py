import copy
from typing import TYPE_CHECKING

from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from monstim_gui.commands import SetLatencyWindowsCommand
from monstim_gui.io.config_repository import ConfigRepository
from monstim_signals.core import LatencyWindow, get_config_path
from monstim_signals.domain.dataset import Dataset
from monstim_signals.domain.experiment import Experiment
from monstim_signals.domain.session import Session

from .base import COLOR_OPTIONS

if TYPE_CHECKING:
    from gui_main import MonstimGUI

# TODO: Fix widths and layout to be less janky.
MINIMUM_WIDTH = 600  # Minimum width for the dialog to accommodate two columns
COL_MIN_WIDTH = 200  # Minimum width for each column in the grid layout


class LatencyWindowsDialog(QDialog):
    """Dialog for editing multiple latency windows."""

    def __init__(self, data: Experiment | Dataset | Session, parent=None, config_repo=None):
        super().__init__(parent)
        self.data = data
        self.gui: MonstimGUI = parent
        self.setModal(False)  # Allow interaction with main window
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint
        )  # Make it a standalone window that stays on top
        self.setWindowTitle("Manage Latency Windows")
        self.window_entries = (
            []
        )  # type: list[tuple[QGroupBox, LatencyWindow, QLineEdit, QDoubleSpinBox, QDoubleSpinBox, QComboBox, QRadioButton, list[QDoubleSpinBox]]]
        self.config_repo = config_repo or ConfigRepository(get_config_path())
        self.init_ui()
        self._reposition_to_left_middle_of_parent()

    def _reposition_to_left_middle_of_parent(self):
        # ensure it's sized correctly first
        self.adjustSize()

        # Ensure minimum width for two columns
        if self.width() < MINIMUM_WIDTH:
            self.resize(MINIMUM_WIDTH, self.height())

        # parent should be your main window
        w = self.parentWidget() or self.window()
        # get its top-left in global coords
        top_left: QPoint = w.mapToGlobal(QPoint(0, 0))
        pw, ph = w.width(), w.height()

        # compute center of the *left half* of that parent
        cx = top_left.x() + pw // 4
        cy = top_left.y() + ph // 2

        # move this dialog so its center sits there
        x = cx - (self.width() // 2)
        y = cy - (self.height() // 2)  # Center vertically instead of top-aligning

        # Ensure dialog doesn't go off screen
        x = max(0, x)
        y = max(0, y)

        self.move(x, y)
        # ensure it’s sized correctly
        self.adjustSize()

        # parent should be your main window
        w = self.parentWidget() or self.window()
        # get its top-left in global coords
        top_left: QPoint = w.mapToGlobal(QPoint(0, 0))
        pw, ph = w.width(), w.height()

        # compute center of the *left half* of that parent
        cx = top_left.x() + pw // 4
        cy = top_left.y() + ph // 2

        # move this dialog so its center sits there
        x = cx - (self.width() // 2)
        y = cy - (self.height())
        self.move(x, y)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.setMaximumHeight(800)
        self.setMinimumWidth(MINIMUM_WIDTH)  # Increased to accommodate two columns
        self.setMaximumWidth(1000)  # Set a reasonable maximum width

        cfg = self.config_repo.read_config()
        self.presets = cfg.get("latency_window_presets", {})

        if self.presets:
            preset_row = QHBoxLayout()
            preset_label = QLabel("Preset:")
            self.preset_combo = QComboBox()
            for name in self.presets.keys():
                self.preset_combo.addItem(name)
            apply_btn = QPushButton("Apply Preset")
            apply_btn.clicked.connect(self._apply_preset)
            preset_row.addWidget(preset_label)
            preset_row.addWidget(self.preset_combo)
            preset_row.addWidget(apply_btn)
            layout.addLayout(preset_row)

        self.scroll: QScrollArea = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scroll.setMinimumWidth(MINIMUM_WIDTH)  # Ensure scroll area can accommodate both columns

        self.scroll_widget = QWidget()
        self.scroll_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.scroll_layout = QGridLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Set equal column stretch so both columns expand equally
        self.scroll_layout.setColumnStretch(0, 1)
        self.scroll_layout.setColumnStretch(1, 1)
        # Add spacing between columns and rows
        self.scroll_layout.setHorizontalSpacing(15)
        self.scroll_layout.setVerticalSpacing(10)
        # Ensure the layout has a minimum column width
        self.scroll_layout.setColumnMinimumWidth(0, COL_MIN_WIDTH)
        self.scroll_layout.setColumnMinimumWidth(1, COL_MIN_WIDTH)

        self.scroll.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll, 1)

        for window in self.data.latency_windows:
            self._add_window_group(window)

        add_button = QPushButton("Add Window")
        add_button.clicked.connect(lambda: self._add_window_group())
        layout.addWidget(add_button, 0)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply,
            self,
        )
        button_box.accepted.connect(self.save_windows)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply_changes)
        layout.addWidget(button_box, 0)
        self.adjustSize()

    def _add_window_group(self, window: LatencyWindow | None = None):
        num_channels = len(self.data.channel_names)
        if window is None:
            window = LatencyWindow(
                name=f"Window {len(self.window_entries)+1}",
                start_times=[0.0] * num_channels,
                durations=[1.0] * num_channels,
                color="black",
                linestyle=":",
            )

        # Ensure window data matches current channel count
        if len(window.start_times) != num_channels:
            # Extend or truncate start_times to match current channels
            if len(window.start_times) > 0:
                default_start = window.start_times[0]
            else:
                default_start = 0.0
            window.start_times = [default_start] * num_channels

        if len(window.durations) != num_channels:
            # Extend or truncate durations to match current channels
            if len(window.durations) > 0:
                default_duration = window.durations[0]
            else:
                default_duration = 1.0
            window.durations = [default_duration] * num_channels
        group = QGroupBox(window.name)
        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(group)

        # Basic form layout for name, duration, and color
        basic_form = QFormLayout()
        name_edit = QLineEdit(window.name)

        # Duration (always global)
        dur_spin = QDoubleSpinBox()
        dur_spin.setDecimals(2)
        dur_spin.setRange(0.0, 1000.0)
        dur_spin.setSingleStep(0.05)
        dur_spin.setValue(window.durations[0])
        dur_spin.setToolTip("Duration is applied globally to all channels")

        # Color
        color_combo = QComboBox()
        for color in COLOR_OPTIONS:
            display = color.replace("tab:", "")
            color_combo.addItem(display, userData=color)
        if window.color in COLOR_OPTIONS:
            color_combo.setCurrentIndex(COLOR_OPTIONS.index(window.color))

        basic_form.addRow("Name", name_edit)
        basic_form.addRow("Duration", dur_spin)
        basic_form.addRow("Color", color_combo)
        layout.addLayout(basic_form)

        # Start times section
        start_group = QGroupBox("Start Times")
        start_layout = QVBoxLayout(start_group)

        # Global/Per-channel toggle
        mode_layout = QVBoxLayout()
        global_radio = QRadioButton("Global")
        global_radio.setToolTip("Apply the same start time to all channels")
        per_channel_radio = QRadioButton("Per-channel")
        per_channel_radio.setToolTip("Set window start times for each individual channel")

        # Create button group to ensure mutual exclusivity
        radio_group = QButtonGroup()
        radio_group.addButton(global_radio)
        radio_group.addButton(per_channel_radio)

        global_radio.setChecked(True)  # Default to global
        mode_layout.addWidget(global_radio)
        mode_layout.addWidget(per_channel_radio)
        start_layout.addLayout(mode_layout)

        # Global start time control
        global_widget = QWidget()
        global_layout = QHBoxLayout(global_widget)
        global_layout.setContentsMargins(0, 0, 0, 0)
        global_layout.addWidget(QLabel("Start time:"))
        global_start_spin = QDoubleSpinBox()
        global_start_spin.setDecimals(2)
        global_start_spin.setRange(-1000.0, 1000.0)
        global_start_spin.setSingleStep(0.05)
        global_start_spin.setValue(window.start_times[0])
        global_layout.addWidget(global_start_spin)
        global_layout.addStretch()
        start_layout.addWidget(global_widget)

        # Per-channel start time controls
        per_channel_widget = QWidget()
        per_channel_widget.setMaximumHeight(200)  # Limit height to prevent dialog from becoming too tall

        # Use scroll area if there are many channels
        if len(self.data.channel_names) > 6:
            per_channel_scroll = QScrollArea()
            per_channel_scroll.setWidgetResizable(True)
            per_channel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            per_channel_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

            per_channel_content = QWidget()
            per_channel_layout = QVBoxLayout(per_channel_content)
            per_channel_scroll.setWidget(per_channel_content)

            # Add scroll area to main widget
            per_channel_main_layout = QVBoxLayout(per_channel_widget)
            per_channel_main_layout.addWidget(per_channel_scroll)
        else:
            per_channel_layout = QVBoxLayout(per_channel_widget)

        per_channel_spins = []

        for i, (channel_name, start_time) in enumerate(zip(self.data.channel_names, window.start_times)):
            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel(f"{channel_name}:"))
            spin = QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setRange(-1000.0, 1000.0)
            spin.setSingleStep(0.05)
            spin.setValue(start_time)
            per_channel_spins.append(spin)
            row_layout.addWidget(spin)
            per_channel_layout.addLayout(row_layout)

        start_layout.addWidget(per_channel_widget)
        per_channel_widget.setVisible(False)  # Hidden by default

        layout.addWidget(start_group)

        # Action buttons
        button_layout = QHBoxLayout()
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_window_group(group))
        button_layout.addWidget(remove_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Connect signals for mode switching
        def on_mode_changed():
            is_global = global_radio.isChecked()
            global_widget.setVisible(is_global)
            per_channel_widget.setVisible(not is_global)

            if is_global:
                # When switching to global, update all per-channel spins to match global value
                global_value = global_start_spin.value()
                for spin in per_channel_spins:
                    spin.setValue(global_value)
                window.start_times = [global_value] * num_channels
            else:
                # When switching to per-channel, update window with current per-channel values
                window.start_times = [spin.value() for spin in per_channel_spins]

            # Force layout update and resize with a slight delay to allow visibility changes to process
            QTimer.singleShot(
                10,
                lambda: [
                    self.updateGeometry(),
                    self.adjustSize(),
                    self.resize(self.sizeHint()),
                ],
            )

        def on_global_value_changed():
            if global_radio.isChecked():
                # Update all per-channel spins and window data
                global_value = global_start_spin.value()
                for spin in per_channel_spins:
                    spin.setValue(global_value)
                window.start_times = [global_value] * num_channels

        def on_per_channel_value_changed():
            if per_channel_radio.isChecked():
                # Update window data with current per-channel values
                window.start_times = [spin.value() for spin in per_channel_spins]
                # Also update global spin to first channel value for consistency
                global_start_spin.setValue(per_channel_spins[0].value())

        global_radio.toggled.connect(on_mode_changed)
        global_start_spin.valueChanged.connect(on_global_value_changed)
        for spin in per_channel_spins:
            spin.valueChanged.connect(on_per_channel_value_changed)

        # Add to grid layout - two columns
        num_windows = len(self.window_entries)
        row = num_windows // 2
        col = num_windows % 2
        self.scroll_layout.addWidget(group, row, col)

        self.window_entries.append(
            (
                group,
                window,
                name_edit,
                global_start_spin,
                dur_spin,
                color_combo,
                global_radio,
                per_channel_spins,
            )
        )

        # Adjust layout and ensure proper sizing
        self.updateGeometry()
        self.adjustSize()
        # Ensure minimum width is maintained
        if self.width() < MINIMUM_WIDTH:
            self.resize(MINIMUM_WIDTH, self.height())

    def _remove_window_group(self, group: QGroupBox):
        for i, (grp, *_) in enumerate(self.window_entries):
            if grp is group:
                self.window_entries.pop(i)
                break

        # Remove from layout and delete
        self.scroll_layout.removeWidget(group)
        group.setParent(None)
        group.deleteLater()

        # Reorganize remaining groups in grid layout
        self._reorganize_grid_layout()

        # Trigger resize after reorganization
        QTimer.singleShot(
            10,
            lambda: [
                self.updateGeometry(),
                self.adjustSize(),
                # Ensure minimum width is maintained
                (self.resize(max(MINIMUM_WIDTH, self.width()), self.height()) if self.width() < MINIMUM_WIDTH else None),
            ],
        )

    def _reorganize_grid_layout(self):
        """Reorganize all window groups in a 2-column grid layout."""
        # Remove all widgets from layout without deleting them
        for i in range(len(self.window_entries)):
            group = self.window_entries[i][0]
            self.scroll_layout.removeWidget(group)

        # Re-add them in grid positions
        for i, (group, *_) in enumerate(self.window_entries):
            row = i // 2
            col = i % 2
            self.scroll_layout.addWidget(group, row, col)

    def _apply_preset(self):
        name = self.preset_combo.currentText()
        if name not in self.presets:
            return

        # Clear existing entries
        for group, *_ in self.window_entries:
            group.setParent(None)
            group.deleteLater()
        self.window_entries.clear()

        num_channels = len(self.data.channel_names)
        for win in self.presets[name]:
            window = LatencyWindow(
                name=win.get("name", "Window"),
                start_times=[float(win.get("start", 0.0))] * num_channels,
                durations=[float(win.get("duration", 1.0))] * num_channels,
                color=win.get("color", "black"),
                linestyle=win.get("linestyle", ":"),
            )
            self._add_window_group(window)

    def save_windows(self):
        new_windows = []
        num_channels = len(self.data.channel_names)
        for (
            group,
            window,
            name_edit,
            global_start_spin,
            dur_spin,
            color_combo,
            global_radio,
            per_channel_spins,
        ) in self.window_entries:
            window.name = name_edit.text().strip() or "Window"

            # Handle start times based on radio button selection
            if global_radio.isChecked():
                # Apply start time globally to all channels
                window.start_times = [global_start_spin.value()] * num_channels
            else:
                # Use per-channel start times from individual spin boxes
                window.start_times = [spin.value() for spin in per_channel_spins]

            # Duration is ALWAYS applied globally - this is a requirement
            window.durations = [dur_spin.value()] * num_channels

            window.color = color_combo.currentData()
            new_windows.append(copy.deepcopy(window))

        if isinstance(self.data, Experiment):
            level = "experiment"
        elif isinstance(self.data, Dataset):
            level = "dataset"
        else:
            level = "session"

        command = SetLatencyWindowsCommand(self.gui, level, new_windows)
        self.gui.command_invoker.execute(command)

        # Show status message in main window
        if self.gui and hasattr(self.gui, "status_bar"):
            self.gui.status_bar.showMessage("Latency windows updated successfully.", 5000)

        # Clean up reference in parent
        if hasattr(self.gui, "_latency_dialog"):
            self.gui._latency_dialog = None

        self.accept()

    def reject(self):
        """Override reject to clean up parent reference."""
        # Clean up reference in parent
        if hasattr(self.gui, "_latency_dialog"):
            self.gui._latency_dialog = None
        super().reject()

    def closeEvent(self, event):
        """Override close event to clean up parent reference."""
        # Clean up reference in parent
        if hasattr(self.gui, "_latency_dialog"):
            self.gui._latency_dialog = None
        super().closeEvent(event)

    def apply_changes(self):
        """Apply current window settings and replot, but keep dialog open."""
        num_channels = len(self.data.channel_names)
        new_windows = []

        for (
            group,
            window,
            name_edit,
            global_start_spin,
            dur_spin,
            color_combo,
            global_radio,
            per_channel_spins,
        ) in self.window_entries:
            window.name = name_edit.text().strip() or "Window"

            # Handle start times based on radio button selection
            if global_radio.isChecked():
                # Apply start time globally to all channels
                window.start_times = [global_start_spin.value()] * num_channels
            else:
                # Use per-channel start times from individual spin boxes
                window.start_times = [spin.value() for spin in per_channel_spins]

            # Duration is ALWAYS applied globally - this is a requirement
            window.durations = [dur_spin.value()] * num_channels

            window.color = color_combo.currentData()
            new_windows.append(copy.deepcopy(window))

        if isinstance(self.data, Experiment):
            level = "experiment"
        elif isinstance(self.data, Dataset):
            level = "dataset"
        else:
            level = "session"

        command = SetLatencyWindowsCommand(self.gui, level, new_windows)
        self.gui.command_invoker.execute(command)

        # Trigger replot to show changes
        if self.gui:
            self.gui.plot_controller.plot_data()
