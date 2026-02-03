import copy
import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
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
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from monstim_gui.commands import InsertSingleLatencyWindowCommand, SetLatencyWindowsCommand
from monstim_gui.core.clipboard import LatencyWindowClipboard
from monstim_gui.io.config_repository import ConfigRepository
from monstim_signals.core import LatencyWindow, get_config_path
from monstim_signals.domain.dataset import Dataset
from monstim_signals.domain.experiment import Experiment
from monstim_signals.domain.session import Session

from .base import COLOR_OPTIONS

if TYPE_CHECKING:
    from gui_main import MonstimGUI

COL_MIN_WIDTH = 200  # Minimum width for each column in the grid layout


class NoScrollComboBox(QComboBox):
    """ComboBox that ignores scroll wheel events to prevent accidental value changes."""

    def wheelEvent(self, event):
        """Ignore wheel events to prevent accidental value changes."""
        event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """DoubleSpinBox that ignores scroll wheel events to prevent accidental value changes."""

    def wheelEvent(self, event):
        """Ignore wheel events to prevent accidental value changes."""
        event.ignore()


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
        # Get screen geometry
        screen = self.screen()
        if not screen:
            return

        screen_rect = screen.availableGeometry()

        # Position dialog's left edge at screen's left edge
        x = screen_rect.left()

        # Position dialog's vertical center at screen's vertical center
        screen_center_y = screen_rect.top() + screen_rect.height() // 2
        y = screen_center_y - (self.height() // 2)

        # Ensure dialog doesn't go off the edges
        x = max(screen_rect.left(), min(x, screen_rect.right() - self.width()))
        y = max(screen_rect.top(), min(y, screen_rect.bottom() - self.height()))

        self.move(x, y)

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Set minimum size but allow user to resize larger
        self.setMinimumSize(450, 200)  # Minimum size to ensure usability
        self.resize(550, 450)  # Default size that's comfortable but can be adjusted

        cfg = self.config_repo.read_config()
        self.presets = cfg.get("latency_window_presets", {})

        if self.presets:
            preset_row = QHBoxLayout()
            preset_row.setSpacing(5)  # Tight spacing between elements
            preset_label = QLabel("Preset:")
            self.preset_combo = NoScrollComboBox()
            self.preset_combo.setToolTip("Select a preset configuration to quickly apply predefined latency windows")
            self.preset_combo.setMinimumWidth(200)  # Make combo box wider for longer preset names
            for name in self.presets.keys():
                self.preset_combo.addItem(name)
            apply_btn = QPushButton("Apply Preset")
            apply_btn.setToolTip("Replace all current windows with the selected preset configuration")
            apply_btn.clicked.connect(self._apply_preset)
            preset_row.addStretch()  # Push everything to the right
            preset_row.addWidget(preset_label)
            preset_row.addWidget(self.preset_combo)
            preset_row.addWidget(apply_btn)
            layout.addLayout(preset_row)

        self.scroll: QScrollArea = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

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
        layout.addWidget(self.scroll, 1)  # Give the scroll area stretch factor of 1 to take up available space

        for window in self.data.latency_windows:
            self._add_window_group(window)

        # --- Action Row (Add / Copy / Paste) ---
        action_row = QHBoxLayout()
        add_button = QPushButton("Add Window")
        add_button.setToolTip("Create a new latency window with default settings")
        add_button.clicked.connect(lambda: self._add_window_group())
        action_row.addWidget(add_button)

        copy_button = QPushButton("Copy All")
        copy_button.setToolTip("Copy the current latency windows to a transient clipboard (not saved to disk)")
        copy_button.clicked.connect(self._copy_windows_to_clipboard)
        action_row.addWidget(copy_button)

        paste_button = QPushButton("Paste")
        paste_button.setToolTip("Paste from clipboard (handles both single and multiple windows)")
        paste_button.clicked.connect(self._paste_windows_from_clipboard)
        paste_button.setEnabled(LatencyWindowClipboard.has_any())
        self._paste_button = paste_button  # store for state updates
        action_row.addWidget(paste_button)

        action_row.addStretch()
        layout.addLayout(action_row)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply,
            self,
        )
        button_box.button(QDialogButtonBox.StandardButton.Ok).setToolTip("Save all changes and close the dialog")
        button_box.button(QDialogButtonBox.StandardButton.Cancel).setToolTip("Discard all changes and close the dialog")
        button_box.button(QDialogButtonBox.StandardButton.Apply).setToolTip(
            "Save changes and update plots, but keep dialog open"
        )
        button_box.accepted.connect(self.save_windows)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply_changes)
        layout.addWidget(button_box, 0)  # No stretch for the button box

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
        name_edit.setToolTip("Enter a descriptive name for this latency window")

        # Duration (always global)
        dur_spin = NoScrollDoubleSpinBox()
        dur_spin.setDecimals(2)
        dur_spin.setRange(0.0, 1000.0)
        dur_spin.setSingleStep(0.05)
        dur_spin.setValue(window.durations[0])
        dur_spin.setToolTip("Duration is applied globally to all channels (in milliseconds)")

        # Color
        color_combo = NoScrollComboBox()
        color_combo.setToolTip("Select the color for this window when displayed on plots")
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
        start_group.setToolTip("Configure when the latency windows begin relative to stimulus")
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

        # Don't set checked state yet; decide after building widgets based on data
        mode_layout.addWidget(global_radio)
        mode_layout.addWidget(per_channel_radio)
        start_layout.addLayout(mode_layout)

        # Global start time control
        global_widget = QWidget()
        global_layout = QHBoxLayout(global_widget)
        global_layout.setContentsMargins(0, 0, 0, 0)
        global_layout.addWidget(QLabel("Start time:"))
        global_start_spin = NoScrollDoubleSpinBox()
        global_start_spin.setDecimals(2)
        global_start_spin.setRange(-1000.0, 1000.0)
        global_start_spin.setSingleStep(0.05)
        global_start_spin.setValue(window.start_times[0])
        global_start_spin.setToolTip("Start time in milliseconds (applied to all channels when Global mode is selected)")
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
            spin = NoScrollDoubleSpinBox()
            spin.setDecimals(2)
            spin.setRange(-1000.0, 1000.0)
            spin.setSingleStep(0.05)
            spin.setValue(start_time)
            spin.setToolTip(f"Start time in milliseconds for channel {channel_name}")
            per_channel_spins.append(spin)
            row_layout.addWidget(spin)
            per_channel_layout.addLayout(row_layout)

        start_layout.addWidget(per_channel_widget)
        per_channel_widget.setVisible(False)  # Hidden by default

        # Determine default editing mode based on whether channels differ
        def _values_differ(vals: list[float], tol: float = 1e-9) -> bool:
            if not vals:
                return False
            return (max(vals) - min(vals)) > tol

        if _values_differ(window.start_times):
            # Default to per-channel if existing values differ to avoid accidental overwrite
            per_channel_radio.setChecked(True)
            per_channel_widget.setVisible(True)
            global_widget.setVisible(False)
        else:
            # Default to global when all values are equal
            global_radio.setChecked(True)

        layout.addWidget(start_group)

        # Action buttons
        button_layout = QHBoxLayout()
        copy_btn = QPushButton("Copy")
        copy_btn.setToolTip("Copy this latency window to clipboard for inserting elsewhere")
        copy_btn.clicked.connect(lambda: self._copy_single_window(group))
        button_layout.addWidget(copy_btn)
        remove_btn = QPushButton("Remove")
        remove_btn.setToolTip("Delete this latency window permanently")
        remove_btn.clicked.connect(lambda: self._remove_window_group(group))
        button_layout.addWidget(remove_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Connect signals for mode switching
        def on_mode_changed():
            is_global = global_radio.isChecked()

            # If switching to global and per-channel values differ, confirm to avoid accidental overwrite
            if is_global:
                channel_vals = [spin.value() for spin in per_channel_spins]

                def _vals_differ(vs: list[float], tol: float = 1e-9) -> bool:
                    if not vs:
                        return False
                    return (max(vs) - min(vs)) > tol

                if _vals_differ(channel_vals):
                    resp = QMessageBox.question(
                        self,
                        "Switch to Global?",
                        "Per-channel start times differ for this window. Switching to Global will overwrite them with a single value. Continue?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if resp != QMessageBox.StandardButton.Yes:
                        # Revert selection to per-channel and exit
                        per_channel_radio.setChecked(True)
                        return

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
                # Keep global spin in sync with first channel for consistency
                if per_channel_spins:
                    global_start_spin.setValue(per_channel_spins[0].value())

            # Force layout update with a slight delay to allow visibility changes to process
            QTimer.singleShot(10, self.updateGeometry)

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
        # After applying preset, any future paste is still valid
        self._update_paste_enabled()
        self._reorganize_grid_layout()

    # ---------------- Clipboard Support -----------------
    def _copy_windows_to_clipboard(self):
        """Copy current windows to the in-memory clipboard."""
        windows = []
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
            # Build a fresh LatencyWindow snapshot (respecting global/per-channel state)
            if global_radio.isChecked():
                start_times = [global_start_spin.value()] * num_channels
            else:
                start_times = [spin.value() for spin in per_channel_spins]
            durations = [dur_spin.value()] * num_channels
            win_copy = LatencyWindow(
                name=name_edit.text().strip() or "Window",
                start_times=start_times,
                durations=durations,
                color=color_combo.currentData(),
                linestyle=window.linestyle,
            )
            windows.append(win_copy)
        if windows:
            LatencyWindowClipboard.set_multiple(windows)
            if self.gui and hasattr(self.gui, "status_bar"):
                self.gui.status_bar.showMessage(f"Copied {len(windows)} latency window(s) to clipboard (transient).", 5000)
        self._update_paste_enabled()

    def _paste_windows_from_clipboard(self):
        """Paste windows from clipboard (handles both single and multi-window clipboards)."""
        # Get most recent clipboard data
        mode, data = LatencyWindowClipboard.get_most_recent()

        if mode == "none":
            QMessageBox.information(self, "Clipboard Empty", "There are no latency windows in the clipboard.")
            self._update_paste_enabled()
            return
        elif mode == "multiple":
            # Handle multi-window paste (replace all)
            self._paste_multi_windows(data)
        elif mode == "single":
            # Handle single-window paste (insert/replace by name)
            self._paste_single_window(data)

    def _paste_multi_windows(self, windows):
        """Paste multiple windows, replacing all current windows."""
        # Confirm replacement if existing windows present
        if self.window_entries:
            resp = QMessageBox.question(
                self,
                "Replace Existing Windows?",
                "Pasting will replace all currently displayed latency windows. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return

        # Clear existing entries
        for group, *_ in self.window_entries:
            self.scroll_layout.removeWidget(group)
            group.setParent(None)
            group.deleteLater()
        self.window_entries.clear()

        # Add new ones (ensure channel counts are reconciled automatically by _add_window_group)
        for w in windows:
            self._add_window_group(w)
        self._reorganize_grid_layout()
        self._update_paste_enabled()
        if self.gui and hasattr(self.gui, "status_bar"):
            self.gui.status_bar.showMessage(f"Pasted {len(windows)} latency window(s) from clipboard.", 5000)

    def _paste_single_window(self, window):
        """Paste a single window, appending or replacing by name in the dialog only."""
        # Check for duplicate names in current dialog
        existing_names = [name_edit.text().strip() for (_, _, name_edit, *_) in self.window_entries]

        if window.name in existing_names:
            # Ask user what to do
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setWindowTitle("Window Name Exists")
            msg.setText(f"A window named '{window.name}' already exists in this view.")
            msg.setInformativeText("Would you like to replace it or insert with a new name?")

            replace_btn = msg.addButton("Replace Existing", QMessageBox.ButtonRole.AcceptRole)
            rename_btn = msg.addButton("Insert as New", QMessageBox.ButtonRole.ActionRole)
            cancel_btn = msg.addButton(QMessageBox.StandardButton.Cancel)

            msg.exec()
            clicked = msg.clickedButton()

            if clicked == cancel_btn:
                return
            elif clicked == replace_btn:
                # Find and remove the existing window with that name
                for i, (grp, _, name_edit, *_) in enumerate(self.window_entries):
                    if name_edit.text().strip() == window.name:
                        self._remove_window_group(grp)
                        break
            elif clicked == rename_btn:
                # Generate a unique name
                base_name = window.name
                counter = 1
                while f"{base_name} ({counter})" in existing_names:
                    counter += 1
                window.name = f"{base_name} ({counter})"

        # Add the window to dialog
        self._add_window_group(window)
        self._reorganize_grid_layout()

        if self.gui and hasattr(self.gui, "status_bar"):
            self.gui.status_bar.showMessage(f"Pasted window '{window.name}' to dialog.", 3000)
        self._update_paste_enabled()

    def _update_paste_enabled(self):
        if hasattr(self, "_paste_button"):
            self._paste_button.setEnabled(LatencyWindowClipboard.has_any())

    def _copy_single_window(self, group: QGroupBox):
        """Copy a single window to the clipboard."""
        # Find the window entry for this group
        for (
            grp,
            window,
            name_edit,
            global_start_spin,
            dur_spin,
            color_combo,
            global_radio,
            per_channel_spins,
        ) in self.window_entries:
            if grp is group:
                # Build a fresh LatencyWindow snapshot
                num_channels = len(self.data.channel_names)
                if global_radio.isChecked():
                    start_times = [global_start_spin.value()] * num_channels
                else:
                    start_times = [spin.value() for spin in per_channel_spins]
                durations = [dur_spin.value()] * num_channels

                win_copy = LatencyWindow(
                    name=name_edit.text().strip() or "Window",
                    start_times=start_times,
                    durations=durations,
                    color=color_combo.currentData(),
                    linestyle=window.linestyle,
                )
                LatencyWindowClipboard.set_single(win_copy)

                if self.gui and hasattr(self.gui, "status_bar"):
                    self.gui.status_bar.showMessage(f"Copied '{win_copy.name}' to clipboard.", 3000)
                self._update_paste_enabled()
                return

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

        logging.info(f"Setting latency windows for {level}: {self.data.id}")
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

        logging.info(f"Setting latency windows for {level}: {self.data.id}")
        command = SetLatencyWindowsCommand(self.gui, level, new_windows)
        self.gui.command_invoker.execute(command)

        # Trigger replot to show changes
        if self.gui:
            self.gui.plot_controller.plot_data()


class AppendReplaceLatencyWindowDialog(QDialog):
    """Specialized dialog for appending or replacing a single latency window across hierarchy.

    This dialog applies changes immediately to the data (not just the UI), making it suitable
    for quick single-window operations without needing to review all windows.
    """

    def __init__(self, data: Experiment | Dataset | Session, parent=None):
        super().__init__(parent)
        self.data = data
        self.gui: MonstimGUI = parent
        self.setModal(True)
        self.setWindowTitle("Append/Replace Latency Window")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.setMinimumWidth(400)

        # Info label
        info_label = QLabel(
            "This action will append or replace latency window(s) across all "
            "sessions at the current level and below. Changes are applied immediately."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Clipboard status
        clipboard_group = QGroupBox("Clipboard Status")
        clipboard_layout = QVBoxLayout(clipboard_group)

        mode, data = LatencyWindowClipboard.get_most_recent()

        if mode == "single":
            clipboard_layout.addWidget(QLabel(f"✓ Single window (most recent): '{data.name}'"))
        elif mode == "multiple":
            count = len(data)
            names = ", ".join([w.name for w in data[:3]])
            if count > 3:
                names += f", ... ({count} total)"
            clipboard_layout.addWidget(QLabel(f"✓ Multiple windows (most recent): {names}"))
        else:
            clipboard_layout.addWidget(QLabel("✗ No clipboard data available"))
            clipboard_layout.addWidget(QLabel("Tip: Open the Latency Windows editor and use Copy or Copy All buttons"))

        layout.addWidget(clipboard_group)

        # Action buttons
        if mode != "none":
            # Determine action based on clipboard mode
            if mode == "single":
                self._add_single_window_actions(layout, data)
            else:  # mode == "multiple"
                self._add_multiple_windows_actions(layout, data)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Show message if no clipboard data
        if mode == "none":
            layout.addWidget(QLabel("Please copy latency window(s) first."))

    def _add_single_window_actions(self, layout, window: LatencyWindow):
        """Add action buttons for single window mode."""
        sessions_to_check = self._get_sessions_to_check()
        window_exists = any(any(w.name == window.name for w in s.annot.latency_windows) for s in sessions_to_check)

        action_group = QGroupBox("Action")
        action_layout = QVBoxLayout(action_group)

        if window_exists:
            action_layout.addWidget(QLabel(f"Window '{window.name}' exists in one or more sessions. Choose action:"))

            replace_btn = QPushButton(f"Replace '{window.name}' Windows")
            replace_btn.setToolTip(f"Replace all existing '{window.name}' windows with clipboard version")
            replace_btn.clicked.connect(lambda: self._execute_single_window_action(window, True))
            action_layout.addWidget(replace_btn)

            append_btn = QPushButton("Insert as New Window")
            append_btn.setToolTip("Add as a new window with a unique name, preserving existing windows")
            append_btn.clicked.connect(lambda: self._execute_single_window_action(window, False))
            action_layout.addWidget(append_btn)
        else:
            action_layout.addWidget(QLabel(f"Window '{window.name}' does not exist. It will be appended to all sessions."))

            append_btn = QPushButton(f"Append '{window.name}'")
            append_btn.setToolTip("Add this window to all sessions at the current level")
            append_btn.clicked.connect(lambda: self._execute_single_window_action(window, True))
            action_layout.addWidget(append_btn)

        layout.addWidget(action_group)

    def _add_multiple_windows_actions(self, layout, windows: list[LatencyWindow]):
        """Add action buttons for multiple windows mode."""
        sessions_to_check = self._get_sessions_to_check()

        # Check which windows exist
        existing_windows = []
        new_windows = []

        for w in windows:
            exists = any(any(sw.name == w.name for sw in s.annot.latency_windows) for s in sessions_to_check)
            if exists:
                existing_windows.append(w.name)
            else:
                new_windows.append(w.name)

        action_group = QGroupBox("Action")
        action_layout = QVBoxLayout(action_group)

        # Show status
        status_text = f"Processing {len(windows)} windows:\n"
        if existing_windows:
            status_text += f"  • {len(existing_windows)} will replace existing: {', '.join(existing_windows[:3])}"
            if len(existing_windows) > 3:
                status_text += "..."
            status_text += "\n"
        if new_windows:
            status_text += f"  • {len(new_windows)} will be appended: {', '.join(new_windows[:3])}"
            if len(new_windows) > 3:
                status_text += "..."

        action_layout.addWidget(QLabel(status_text))

        apply_btn = QPushButton(f"Apply {len(windows)} Windows")
        apply_btn.setToolTip("Apply all windows: replace existing by name, append new ones")
        apply_btn.clicked.connect(lambda: self._execute_multiple_windows_action(windows))
        action_layout.addWidget(apply_btn)

        layout.addWidget(action_group)

    def _get_sessions_to_check(self):
        """Get all sessions that will be affected by this operation."""
        if isinstance(self.data, Experiment):
            return [s for ds in self.data.datasets for s in ds.sessions]
        elif isinstance(self.data, Dataset):
            return list(self.data.sessions)
        else:
            return [self.data]

    def _execute_single_window_action(self, window: LatencyWindow, replace_mode: bool):
        """Execute the append/replace action for a single window."""
        # Determine level
        if isinstance(self.data, Experiment):
            level = "experiment"
        elif isinstance(self.data, Dataset):
            level = "dataset"
        else:
            level = "session"

        # If not replacing, generate unique name
        if not replace_mode:
            sessions_to_check = self._get_sessions_to_check()
            existing_names = set()
            for s in sessions_to_check:
                existing_names.update(w.name for w in s.annot.latency_windows)

            base_name = window.name
            counter = 1
            while f"{base_name} ({counter})" in existing_names:
                counter += 1
            window.name = f"{base_name} ({counter})"

        # Execute command
        command = InsertSingleLatencyWindowCommand(self.gui, level, window, replace_mode)
        self.gui.command_invoker.execute(command)

        # Trigger replot
        if self.gui:
            self.gui.plot_controller.plot_data()
            if hasattr(self.gui, "status_bar"):
                action = "replaced" if replace_mode else "appended"
                self.gui.status_bar.showMessage(f"Window '{window.name}' {action} successfully.", 5000)

        self.accept()

    def _execute_multiple_windows_action(self, windows: list[LatencyWindow]):
        """Execute append/replace for multiple windows."""
        # Determine level
        if isinstance(self.data, Experiment):
            level = "experiment"
        elif isinstance(self.data, Dataset):
            level = "dataset"
        else:
            level = "session"

        # Execute a command for each window (replace mode for all)
        for window in windows:
            command = InsertSingleLatencyWindowCommand(self.gui, level, window, replace_mode=True)
            self.gui.command_invoker.execute(command)

        # Trigger replot
        if self.gui:
            self.gui.plot_controller.plot_data()
            if hasattr(self.gui, "status_bar"):
                self.gui.status_bar.showMessage(f"{len(windows)} windows applied successfully.", 5000)

        self.accept()
