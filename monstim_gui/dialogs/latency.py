import logging
from html import escape

logger = logging.getLogger(__name__)
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from monstim_gui.commands import InsertSingleLatencyWindowCommand, SetLatencyWindowsCommand
from monstim_gui.core.clipboard import LatencyWindowClipboard
from monstim_gui.io.config_repository import ConfigRepository
from monstim_gui.widgets.latency_window_editor import LatencyWindowEditor
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
    """Live, non-modal editor for latency windows in the active data context."""

    def __init__(self, data: Experiment | Dataset | Session, parent=None, config_repo=None):
        super().__init__(parent)
        self.data = data
        self.gui: MonstimGUI = parent
        self._draft_dirty = False
        self.setModal(False)  # Allow interaction with main window
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)  # Make it a standalone window that stays on top
        self.setWindowTitle("Manage Latency Windows")
        self.window_entries = []  # type: list[tuple[QGroupBox, LatencyWindow, QLineEdit, QDoubleSpinBox, QDoubleSpinBox, QComboBox, QRadioButton, list[QDoubleSpinBox]]]
        self._move_buttons: dict[QGroupBox, tuple[QPushButton, QPushButton]] = {}
        self.config_repo = config_repo or ConfigRepository(get_config_path())
        self.init_ui()
        self.set_apply_level(self._level_for_data(data), reload=True)
        self._reposition_to_left_middle_of_parent()

    @staticmethod
    def _level_for_data(data: Experiment | Dataset | Session) -> str:
        if isinstance(data, Experiment):
            return "experiment"
        if isinstance(data, Dataset):
            return "dataset"
        return "session"

    def _target_for_level(self, level: str) -> Experiment | Dataset | Session | None:
        """Resolve a target from the current selection, with constructor fallback."""
        target = getattr(self.gui, f"current_{level}", None) if self.gui else None
        if target is not None:
            return target
        return self.data if self._level_for_data(self.data) == level else None

    def _representative_source_markup(self, target: Experiment | Dataset | Session) -> str:
        """Return a concise, structured description of the draft source."""
        if isinstance(target, Session):
            return f"<b>Values from:</b> <b>Representative:</b> Session annotation  |  <b>Session:</b> {escape(target.id, quote=False)}"
        if isinstance(target, Dataset):
            sessions = target.sessions
            return (
                f"<b>Values from:</b> <b>Representative:</b> Dataset  |  <b>Session:</b> {escape(sessions[0].id, quote=False)}"
                if sessions
                else "<b>Values from:</b> no active session is available"
            )
        datasets = target.datasets
        if not datasets:
            return "<b>Values from:</b> no active dataset is available"
        representative = max(datasets, key=lambda dataset: len(dataset.latency_windows))
        sessions = representative.sessions
        child = f"  |  <b>Session:</b> {escape(sessions[0].id, quote=False)}" if sessions else ""
        return f"<b>Values from:</b> <b>Representative:</b> Experiment  |  <b>Dataset:</b> {escape(representative.id, quote=False)}{child}"

    @staticmethod
    def _session_count(target: Experiment | Dataset | Session) -> int:
        if isinstance(target, Session):
            return 1
        if isinstance(target, Dataset):
            return len(target.sessions)
        return sum(len(dataset.sessions) for dataset in target.datasets)

    def _update_context_summary(self, target: Experiment | Dataset | Session | None) -> None:
        active = []
        for level in ("experiment", "dataset", "session"):
            item = getattr(self.gui, f"current_{level}", None) if self.gui else None
            if item is not None:
                active.append(f"<b>{level.title()}:</b> {escape(item.id, quote=False)}")
        self.active_context_label.setText("<b>Active:</b> " + "  |  ".join(active) if active else "<b>Active:</b> No active data selection")
        if target is None:
            self.value_source_label.setText("<b>Values from:</b> unavailable for this apply level")
            self.apply_summary_label.setText("<b>Apply target:</b> unavailable")
            self.heterogeneity_label.setText("")
            self.heterogeneity_label.setVisible(False)
            return
        level = self.apply_level_combo.currentData()
        self.value_source_label.setText(self._representative_source_markup(target))
        self.apply_summary_label.setText(
            f"<b>Apply target:</b> <b>Level:</b> {level.title()}  |  <b>ID:</b> {escape(target.id, quote=False)}  "
            f"|  <b>Updates:</b> {self._session_count(target)} session annotation(s)"
        )
        if not isinstance(target, Session) and target.has_heterogeneous_latency_windows:
            self.heterogeneity_label.setText(
                f"<b>Warning:</b> Child window sets differ; Apply replaces all {self._session_count(target)} affected session annotation(s)."
            )
            self.heterogeneity_label.setVisible(True)
        else:
            self.heterogeneity_label.setText("")
            self.heterogeneity_label.setVisible(False)

    def set_apply_level(self, level: str, *, reload: bool = True) -> None:
        """Select an apply scope and load that scope's current representative values."""
        index = self.apply_level_combo.findData(level)
        if index < 0:
            raise ValueError(f"Invalid latency-window apply level: {level}")
        self.apply_level_combo.blockSignals(True)
        self.apply_level_combo.setCurrentIndex(index)
        self.apply_level_combo.blockSignals(False)
        if reload:
            self.refresh_from_current_selection()

    def refresh_from_current_selection(self) -> None:
        """Refresh the draft after a main-window selection change or scope change."""
        level = self.apply_level_combo.currentData()
        target = self._target_for_level(level)
        replaced_unsaved_draft = self._draft_dirty
        self._update_context_summary(target)
        enabled = target is not None
        self.editor.setEnabled(enabled)
        self.apply_button.setEnabled(enabled)
        self.ok_button.setEnabled(enabled)
        if not enabled:
            self.draft_notice_label.setText("")
            self.draft_notice_label.setVisible(False)
            return
        self.data = target
        self.editor.set_channel_names(target.channel_names)
        self.editor.set_windows(target.latency_windows)
        self._draft_dirty = False
        self.draft_notice_label.setText(
            "Unsaved draft edits were discarded because the active selection or apply level changed." if replaced_unsaved_draft else ""
        )
        self.draft_notice_label.setVisible(replaced_unsaved_draft)

    def _on_apply_level_changed(self) -> None:
        self.refresh_from_current_selection()

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

        # Keep the editor small enough that users can inspect the plots behind it.
        self.setMinimumSize(650, 400)
        self.resize(820, 520)

        cfg = self.config_repo.read_config()
        self.presets = cfg.get("latency_window_presets", {})

        context_layout = QVBoxLayout()
        context_layout.setContentsMargins(0, 0, 0, 0)
        context_layout.setSpacing(1)
        scope_row = QHBoxLayout()
        self.apply_level_combo = QComboBox()
        for level in ("session", "dataset", "experiment"):
            self.apply_level_combo.addItem(level.title(), level)
        self.apply_level_combo.setToolTip("Choose the scope that receives changes and supplies the draft values.")
        self.active_context_label = QLabel()
        self.active_context_label.setWordWrap(True)
        self.value_source_label = QLabel()
        self.value_source_label.setWordWrap(True)
        self.apply_summary_label = QLabel()
        self.apply_summary_label.setWordWrap(True)
        self.heterogeneity_label = QLabel()
        self.heterogeneity_label.setWordWrap(True)
        self.heterogeneity_label.setStyleSheet("color: #b26a00;")
        self.draft_notice_label = QLabel()
        self.draft_notice_label.setWordWrap(True)
        self.draft_notice_label.setStyleSheet("color: #b26a00;")
        self.heterogeneity_label.setVisible(False)
        self.draft_notice_label.setVisible(False)
        scope_row.addWidget(QLabel("<b>Apply changes to</b>"))
        scope_row.addWidget(self.apply_level_combo)
        scope_row.addWidget(self.active_context_label, 1)
        context_layout.addLayout(scope_row)
        context_layout.addWidget(self.value_source_label)
        context_layout.addWidget(self.apply_summary_label)
        context_layout.addWidget(self.heterogeneity_label)
        context_layout.addWidget(self.draft_notice_label)
        layout.addLayout(context_layout)

        self.preset_button = QToolButton(self)
        self.preset_button.setText("Presets")
        self.preset_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.preset_button.setToolTip("Replace the draft with a saved latency-window preset")
        self.preset_button.setMinimumWidth(88)
        preset_menu = QMenu(self.preset_button)
        for name in self.presets:
            preset_menu.addAction(name, lambda checked=False, preset_name=name: self._apply_preset(preset_name))
        self.preset_button.setMenu(preset_menu)
        self.preset_button.setEnabled(bool(self.presets))
        if not self.presets:
            self.preset_button.setToolTip("No latency-window presets are configured")

        self.editor = LatencyWindowEditor(self.data.channel_names, self, minimal_toolbar=True, toolbar_extra=self.preset_button)
        self.editor.changed.connect(self._mark_draft_dirty)
        layout.addWidget(self.editor, 1)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Apply,
            self,
        )
        self.ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.apply_button = button_box.button(QDialogButtonBox.StandardButton.Apply)
        self.ok_button.setToolTip("Save all changes and close the dialog")
        button_box.button(QDialogButtonBox.StandardButton.Cancel).setToolTip("Discard all changes and close the dialog")
        self.apply_button.setToolTip("Save changes and update plots, but keep dialog open")
        self.ok_button.setDefault(False)
        self.apply_button.setDefault(True)
        button_box.accepted.connect(self.save_windows)
        button_box.rejected.connect(self.reject)
        self.apply_button.clicked.connect(self.apply_changes)
        self.apply_level_combo.currentIndexChanged.connect(self._on_apply_level_changed)
        layout.addWidget(button_box, 0)  # No stretch for the button box

    def _mark_draft_dirty(self) -> None:
        self._draft_dirty = True

    def _add_window_group(self, window: LatencyWindow | None = None):
        num_channels = len(self.data.channel_names)
        if window is None:
            window = LatencyWindow(
                name=f"Window {len(self.window_entries) + 1}",
                start_times=[0.0] * num_channels,
                durations=[1.0] * num_channels,
                color="black",
                linestyle=":",
            )

        # Ensure window data matches current channel count
        if len(window.start_times) != num_channels:
            # Extend or truncate start_times to match current channels
            default_start = window.start_times[0] if len(window.start_times) > 0 else 0.0
            window.start_times = [default_start] * num_channels

        if len(window.durations) != num_channels:
            # Extend or truncate durations to match current channels
            default_duration = window.durations[0] if len(window.durations) > 0 else 1.0
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

        for _i, (channel_name, start_time) in enumerate(zip(self.data.channel_names, window.start_times, strict=True)):
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

        move_up_btn = QPushButton("↑")
        move_up_btn.setAccessibleName("Move window up")
        move_up_btn.setToolTip("Move this latency window earlier in the list")
        move_up_btn.clicked.connect(lambda: self._move_window_group(group, -1))
        button_layout.addWidget(move_up_btn)

        move_down_btn = QPushButton("↓")
        move_down_btn.setAccessibleName("Move window down")
        move_down_btn.setToolTip("Move this latency window later in the list")
        move_down_btn.clicked.connect(lambda: self._move_window_group(group, 1))
        button_layout.addWidget(move_down_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.setToolTip("Delete this latency window permanently")
        remove_btn.clicked.connect(lambda: self._remove_window_group(group))
        button_layout.addWidget(remove_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self._move_buttons[group] = (move_up_btn, move_down_btn)

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
        self._reorganize_grid_layout()

    def _remove_window_group(self, group: QGroupBox):
        for i, (grp, *_) in enumerate(self.window_entries):
            if grp is group:
                self.window_entries.pop(i)
                break

        self._move_buttons.pop(group, None)

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

            move_buttons = self._move_buttons.get(group)
            if move_buttons:
                move_buttons[0].setEnabled(i > 0)
                move_buttons[1].setEnabled(i < len(self.window_entries) - 1)

    def _move_window_group(self, group: QGroupBox, direction: int):
        """Move a window one position earlier or later in the editor."""
        current_index = next((i for i, (grp, *_) in enumerate(self.window_entries) if grp is group), None)
        if current_index is None:
            return

        new_index = current_index + direction
        if not 0 <= new_index < len(self.window_entries):
            return

        self.window_entries[current_index], self.window_entries[new_index] = (
            self.window_entries[new_index],
            self.window_entries[current_index],
        )
        self._reorganize_grid_layout()

    def _apply_preset(self, name: str | None = None):
        if name is None:
            return
        if name not in self.presets:
            return

        num_channels = len(self.data.channel_names)
        windows = []
        for win in self.presets[name]:
            windows.append(
                LatencyWindow(
                    name=win.get("name", "Window"),
                    start_times=[float(win.get("start", 0.0))] * num_channels,
                    durations=[float(win.get("duration", 1.0))] * num_channels,
                    color=win.get("color", "black"),
                    linestyle=win.get("linestyle", ":"),
                )
            )
        self.editor.set_windows(windows)

    # ---------------- Clipboard Support -----------------
    def _copy_windows_to_clipboard(self):
        """Copy current windows to the in-memory clipboard."""
        windows = []
        num_channels = len(self.data.channel_names)
        for (
            _group,
            window,
            name_edit,
            global_start_spin,
            dur_spin,
            color_combo,
            global_radio,
            per_channel_spins,
        ) in self.window_entries:
            # Build a fresh LatencyWindow snapshot (respecting global/per-channel state)
            start_times = [global_start_spin.value()] * num_channels if global_radio.isChecked() else [spin.value() for spin in per_channel_spins]
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
        self._move_buttons.clear()

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
                for _i, (grp, _, name_edit, *_) in enumerate(self.window_entries):
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
                start_times = [global_start_spin.value()] * num_channels if global_radio.isChecked() else [spin.value() for spin in per_channel_spins]
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
        level = self.apply_level_combo.currentData()
        target = self._target_for_level(level)
        if target is None:
            return
        new_windows = self.editor.windows()
        logger.info("Setting latency windows for %s: %s", level, target.id)
        command = SetLatencyWindowsCommand(self.gui, level, new_windows)
        self.gui.command_invoker.execute(command)

        # Show status message in main window
        if self.gui and hasattr(self.gui, "status_bar"):
            self.gui.status_bar.showMessage("Latency windows updated successfully.", 5000)

        # Clean up reference in parent
        if getattr(self.gui, "_latency_dialog", None) is self:
            self.gui._latency_dialog = None

        self.accept()

    def reject(self):
        """Override reject to clean up parent reference."""
        # Clean up reference in parent
        if getattr(self.gui, "_latency_dialog", None) is self:
            self.gui._latency_dialog = None
        super().reject()

    def closeEvent(self, event):
        """Override close event to clean up parent reference."""
        # Clean up reference in parent
        if getattr(self.gui, "_latency_dialog", None) is self:
            self.gui._latency_dialog = None
        super().closeEvent(event)

    def apply_changes(self):
        """Apply current window settings and replot, but keep dialog open."""
        level = self.apply_level_combo.currentData()
        target = self._target_for_level(level)
        if target is None:
            return
        new_windows = self.editor.windows()
        logger.info("Setting latency windows for %s: %s", level, target.id)
        command = SetLatencyWindowsCommand(self.gui, level, new_windows)
        self.gui.command_invoker.execute(command)
        self._draft_dirty = False
        self._update_context_summary(target)

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
