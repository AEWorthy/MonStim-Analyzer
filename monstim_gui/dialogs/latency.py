from .base import *


class WindowStartDialog(QDialog):
    """Dialog for editing per-channel latency window start times."""

    def __init__(self, window: LatencyWindow, channel_names: list[str], gui, start_spin: QDoubleSpinBox, parent=None):
        super().__init__(parent)
        self.window = window
        self.channel_names = channel_names
        self.gui = gui  # EMGAnalysisGUI
        self.start_spin = start_spin
        self.setModal(True)
        self.setWindowTitle(f"Edit Start Times - {window.name}")
        self.spin_boxes: list[QDoubleSpinBox] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        for name, start in zip(self.channel_names, self.window.start_times):
            row = QHBoxLayout()
            row.addWidget(QLabel(name))
            spin = QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setRange(-1000.0, 1000.0)
            spin.setSingleStep(0.05)
            spin.setValue(start)
            row.addWidget(spin)
            self.spin_boxes.append(spin)
            layout.addLayout(row)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply,
            self,
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply_changes)
        layout.addWidget(button_box)

    def _on_accept(self):
        self.apply_changes()
        self.accept()

    def get_start_times(self) -> list[float]:
        return [sb.value() for sb in self.spin_boxes]

    def apply_changes(self):
        self.window.start_times = self.get_start_times()
        self.start_spin.setValue(self.window.start_times[0])
        if self.gui:
            self.gui.plot_controller.plot_data()

class LatencyWindowsDialog(QDialog):
    """Dialog for editing multiple latency windows."""

    def __init__(self, data: Experiment | Dataset | Session, parent=None):
        super().__init__(parent)
        self.data = data
        self.gui = parent  # EMGAnalysisGUI
        self.setModal(True)
        self.setWindowTitle("Manage Latency Windows")
        self.window_entries = []  # type: list[tuple[QGroupBox, LatencyWindow, QLineEdit, QDoubleSpinBox, QDoubleSpinBox, QComboBox]]
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll)

        for window in self.data.latency_windows:
            self._add_window_group(window)

        add_button = QPushButton("Add Window")
        add_button.clicked.connect(lambda: self._add_window_group())
        layout.addWidget(add_button)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        button_box.accepted.connect(self.save_windows)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _add_window_group(self, window: LatencyWindow | None = None):
        num_channels = len(self.data.channel_names)
        if window is None:
            window = LatencyWindow(
                name=f"Window {len(self.window_entries)+1}",
                start_times=[0.0] * num_channels,
                durations=[1.0] * num_channels,
                color="black",
                linestyle=":"
            )
        group = QGroupBox(window.name)
        form = QFormLayout()
        name_edit = QLineEdit(window.name)
        start_spin = QDoubleSpinBox()
        start_spin.setDecimals(2)
        start_spin.setRange(-1000.0, 1000.0)
        start_spin.setSingleStep(0.05)
        start_spin.setValue(window.start_times[0])
        dur_spin = QDoubleSpinBox()
        dur_spin.setDecimals(2)
        dur_spin.setRange(0.0, 1000.0)
        dur_spin.setSingleStep(0.05)
        dur_spin.setValue(window.durations[0])
        color_combo = QComboBox()
        for color in COLOR_OPTIONS:
            display = color.replace("tab:", "")
            color_combo.addItem(display, userData=color)
        if window.color in COLOR_OPTIONS:
            color_combo.setCurrentIndex(COLOR_OPTIONS.index(window.color))
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_window_group(group))
        edit_btn = QPushButton("Edit Starts...")
        edit_btn.clicked.connect(lambda: self._edit_window_starts(window, start_spin))
        form.addRow("Name", name_edit)
        form.addRow("Start (Ch 0)", start_spin)
        form.addRow("Duration", dur_spin)
        form.addRow("Color", color_combo)
        form.addRow(remove_btn, edit_btn)
        group.setLayout(form)
        self.scroll_layout.addWidget(group)
        self.window_entries.append((group, window, name_edit, start_spin, dur_spin, color_combo))

    def _remove_window_group(self, group: QGroupBox):
        for i, (grp, *_ ) in enumerate(self.window_entries):
            if grp is group:
                self.window_entries.pop(i)
                break
        group.setParent(None)

    def _edit_window_starts(self, window: LatencyWindow, start_spin: QDoubleSpinBox):
        dialog = WindowStartDialog(window, self.data.channel_names, self.gui, start_spin, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            window.start_times = dialog.get_start_times()
            start_spin.setValue(window.start_times[0])

    def save_windows(self):
        new_windows = []
        num_channels = len(self.data.channel_names)
        for group, window, name_edit, start_spin, dur_spin, color_combo in self.window_entries:
            window.name = name_edit.text().strip() or "Window"
            window.start_times[0] = start_spin.value()
            window.durations = [dur_spin.value()] * num_channels
            window.color = color_combo.currentData()
            new_windows.append(copy.deepcopy(window))

        if isinstance(self.data, Experiment):
            level = 'experiment'
        elif isinstance(self.data, Dataset):
            level = 'dataset'
        else:
            level = 'session'

        command = SetLatencyWindowsCommand(self.gui, level, new_windows)
        self.gui.command_invoker.execute(command)
        self.accept()
