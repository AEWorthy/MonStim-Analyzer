import logging

from PyQt6.QtCore import QEvent, QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from monstim_signals.domain.dataset import Dataset
from monstim_signals.domain.experiment import Experiment
from monstim_signals.domain.session import Session

"""LEGACY DIALOG: This dialog is used to update the reflex window settings for an Experiment, Dataset, or Session.
It allows users to set global or per-channel start times and durations for the reflex windows.
This dialog is no longer used in the main application, but is kept for reference and potential future use.
It is recommended to use the new ReflexWindowSettingsDialog in 'latency.py' instead, which provides a more flexible interface."""


class ReflexSettingsDialog(QDialog):
    def __init__(self, data: Experiment | Dataset | Session, parent=None):
        super().__init__(parent)
        self.data = data

        self.setModal(True)
        self.setWindowTitle(f"Update Reflex Window Settings: Dataset {self.data.formatted_name}")

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Toggle button for global or per-channel settings
        self.toggle_button = QPushButton("Switch to Per-Channel Start Times")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)  # Default to global settings
        self.toggle_button.toggled.connect(self.toggle_settings_mode)
        layout.addWidget(self.toggle_button)

        # Global settings layout
        self.global_layout = QVBoxLayout()
        self.global_durations_layout = QHBoxLayout()
        self.global_starts_layout = QHBoxLayout()

        self.global_durations_layout.addWidget(QLabel("m_duration:"))
        self.global_m_duration_entry = QLineEdit(str(self.data.m_duration[0]))
        self.global_m_duration_entry.installEventFilter(self)
        self.global_durations_layout.addWidget(self.global_m_duration_entry)

        self.global_durations_layout.addWidget(QLabel("h_duration:"))
        self.global_h_duration_entry = QLineEdit(str(self.data.h_duration[0]))
        self.global_h_duration_entry.installEventFilter(self)
        self.global_durations_layout.addWidget(self.global_h_duration_entry)

        self.global_starts_layout.addWidget(QLabel("m_start:"))
        self.global_m_start_entry = QLineEdit(str(self.data.m_start[0]))
        self.global_m_start_entry.installEventFilter(self)
        self.global_starts_layout.addWidget(self.global_m_start_entry)

        self.global_starts_layout.addWidget(QLabel("h_start:"))
        self.global_h_start_entry = QLineEdit(str(self.data.h_start[0]))
        self.global_h_start_entry.installEventFilter(self)
        self.global_starts_layout.addWidget(self.global_h_start_entry)

        self.global_layout.addLayout(self.global_durations_layout)
        self.global_layout.addLayout(self.global_starts_layout)
        layout.addLayout(self.global_layout)

        # Per-channel settings layout
        self.per_channel_layout = QVBoxLayout()
        self.entries: list[tuple[QLineEdit, QLineEdit]] = []
        self.labels: list[QLabel] = []
        self.sub_labels: list[QLabel] = []
        for i in range(len(self.data.m_start)):
            channel_label = QLabel(f"Channel {i}:")
            m_start_label = QLabel("m_start:")
            m_start_entry = QLineEdit(str(self.data.m_start[i]))
            m_start_entry.installEventFilter(self)
            h_start_label = QLabel("h_start:")
            h_start_entry = QLineEdit(str(self.data.h_start[i]))
            h_start_entry.installEventFilter(self)

            channel_layout = QHBoxLayout()
            channel_layout.addWidget(channel_label)
            channel_layout.addWidget(m_start_label)
            channel_layout.addWidget(m_start_entry)
            channel_layout.addWidget(h_start_label)
            channel_layout.addWidget(h_start_entry)

            self.per_channel_layout.addLayout(channel_layout)
            self.entries.append((m_start_entry, h_start_entry))
            self.labels.append(channel_label)
            self.sub_labels.extend([m_start_label, h_start_label])

        layout.addLayout(self.per_channel_layout)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.save_settings)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

        # Start with per-channel settings hidden
        self.toggle_settings_mode(True)

    def toggle_settings_mode(self, checked):
        if checked:
            self.toggle_button.setText("Switch to Per-Channel Start Times")
            self.global_layout.setEnabled(True)
            self.global_m_start_entry.setEnabled(True)
            self.global_h_start_entry.setEnabled(True)
            for entry in self.entries:
                for widget in entry:
                    widget.setVisible(False)
                    widget.setEnabled(False)
            for label in self.labels + self.sub_labels:
                label.setVisible(False)
        else:
            self.toggle_button.setText("Switch to Global Start Times")
            self.global_layout.setEnabled(True)
            self.global_m_start_entry.setEnabled(False)
            self.global_h_start_entry.setEnabled(False)
            for entry in self.entries:
                for widget in entry:
                    widget.setVisible(True)
                    widget.setEnabled(True)
            for label in self.labels + self.sub_labels:
                label.setVisible(True)

        # Activate and update the layout
        self.layout().activate()

        # Adjust the dialog size
        self.adjustSize()
        self.updateGeometry()

    def save_settings(self):
        try:
            m_duration = [float(self.global_m_duration_entry.text()) for _ in range(len(self.data.m_start))]
            h_duration = [float(self.global_h_duration_entry.text()) for _ in range(len(self.data.m_start))]
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Invalid input for durations. Please enter valid numbers.",
            )
            return

        if self.toggle_button.isChecked():
            # Global start times
            try:
                m_start = [float(self.global_m_start_entry.text()) for _ in range(len(self.data.m_start))]
                h_start = [float(self.global_h_start_entry.text()) for _ in range(len(self.data.m_start))]
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Invalid Input",
                    "Invalid input for global start times. Please enter valid numbers.",
                )
                return
        else:
            # Per-channel start times
            m_start = []
            h_start = []
            for i, (m_start_entry, h_start_entry) in enumerate(self.entries):
                try:
                    m_start.append(float(m_start_entry.text()))
                    h_start.append(float(h_start_entry.text()))
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Invalid Input",
                        f"Invalid input for channel {i}. Skipping.",
                    )
                    return

        try:
            self.data.change_reflex_latency_windows(m_start, m_duration, h_start, h_duration)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error saving settings: {str(e)}")
            logging.error(
                f"Error saving reflex settings: {str(e)}\n\tdata: {self.data}\n\tm_start: {m_start}\n\tm_duration: {m_duration}\n\th_start: {h_start}\n\th_duration: {h_duration}"
            )
            return

        self.data.update_latency_window_parameters()
        self.data.reset_all_caches()

        self.accept()

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.FocusIn and isinstance(source, QLineEdit):
            QTimer.singleShot(0, source.selectAll)
        return super().eventFilter(source, event)
