import logging
from typing import TYPE_CHECKING

from PyQt6.QtGui import QValidator
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from monstim_signals.core import get_main_window

if TYPE_CHECKING:
    from monstim_gui.gui_main import MonstimGUI


class CustomSpinBox(QSpinBox):
    # Custom SpinBox that wraps around when reaching the maximum or minimum value
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setKeyboardTracking(False)

    def validate(self, text, pos):
        if text == "" or text == "-":
            return QValidator.State.Intermediate, text, pos
        if text.lstrip("-").isdigit():
            return QValidator.State.Acceptable, text, pos
        return QValidator.State.Invalid, text, pos

    def textFromValue(self, value):
        return str(value)

    def valueFromText(self, text):
        try:
            value = int(text)
            if value > self.maximum():
                return self.maximum()
            elif value < self.minimum():
                return self.minimum()
            return value
        except ValueError:
            return self.value()

    def stepBy(self, steps):
        current_value = self.value()
        new_value = current_value + steps
        if new_value > self.maximum():
            self.setValue(self.minimum() + (new_value - self.maximum() - 1))
        elif new_value < self.minimum():
            self.setValue(self.maximum() - (self.minimum() - new_value - 1))
        else:
            self.setValue(new_value)

    def fixup(self, input):
        try:
            value = int(input)
            if value > self.maximum():
                return str(self.maximum())
            elif value < self.minimum():
                return str(self.minimum())
            return input
        except ValueError:
            return str(self.value())


class RecordingCyclerWidget(QGroupBox):
    def __init__(self, parent):
        super().__init__("Recording Cycler", parent)

        self.gui: "MonstimGUI" = get_main_window()
        if not self.gui.current_session:
            self.max_recording_idxs = 0
        else:
            self.max_recording_idxs = self.gui.current_session.num_all_recordings - 1

        # Set size policy to be fixed height
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)

        self.layout = QGridLayout()  # type: QGridLayout
        self.layout.setSpacing(6)  # Increased spacing for better appearance
        self.layout.setContentsMargins(8, 8, 8, 8)  # Increased padding to prevent border clipping
        self.setLayout(self.layout)

        self.prev_button = QPushButton("<--")
        self.next_button = QPushButton("-->")
        self.exclude_button = QPushButton("Exclude")
        self.recording_spinbox = CustomSpinBox()
        self.step_size = CustomSpinBox()

        # Set up the recording spinbox
        self.recording_spinbox.setMinimum(0)
        self.recording_spinbox.setMaximum(self.max_recording_idxs)
        self.recording_spinbox.setWrapping(True)

        # Set up the step size spinbox
        self.step_size.setMinimum(1)
        self.step_size.setMaximum(self.max_recording_idxs)
        self.step_size.setValue(1)

        # Simple horizontal layout
        step_label = QLabel("Step size:")
        rec_label = QLabel("Recording:")

        # First row
        hbox1 = QHBoxLayout()
        hbox1.addWidget(step_label)
        hbox1.addWidget(self.step_size)
        hbox1.addWidget(self.prev_button)
        hbox1.addWidget(self.next_button)

        # Second row
        hbox2 = QHBoxLayout()
        hbox2.addWidget(rec_label)
        hbox2.addWidget(self.recording_spinbox)
        hbox2.addWidget(self.exclude_button)

        # Add to main layout
        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.setSpacing(3)
        vbox.setContentsMargins(8, 8, 8, 8)

        # Replace grid layout with vbox
        QWidget().setLayout(self.layout)  # Clear existing layout
        self.layout = vbox
        self.setLayout(self.layout)

        self.prev_button.clicked.connect(self.on_previous)
        self.next_button.clicked.connect(self.on_next)
        self.exclude_button.clicked.connect(self.on_exclude)
        self.recording_spinbox.valueChanged.connect(self.on_recording_changed)

    def reset_max_recordings(self):
        if not self.gui.current_session:
            self.max_recording_idxs = 0
        else:
            self.max_recording_idxs = max(0, self.gui.current_session.num_recordings - 1)
        # Block signals so we don't trigger plot updates with transient invalid values
        self.recording_spinbox.blockSignals(True)
        old_val = self.recording_spinbox.value()
        self.recording_spinbox.setMaximum(self.max_recording_idxs)
        if old_val > self.max_recording_idxs:
            # Wrap to 0 (expected UX) rather than clamping to last
            self.recording_spinbox.setValue(0 if self.max_recording_idxs >= 0 else 0)
        new_val = self.recording_spinbox.value()
        self.recording_spinbox.blockSignals(False)
        # If the value actually changed while signals were blocked, manually propagate
        if new_val != old_val:
            self.on_recording_changed(new_val)

    def on_previous(self):
        if self.recording_spinbox.value() - self.step_size.value() < 0:
            self.recording_spinbox.setValue(self.recording_spinbox.maximum())
        else:
            self.recording_spinbox.setValue(self.recording_spinbox.value() - self.step_size.value())

    def on_next(self):
        if self.recording_spinbox.value() + self.step_size.value() > self.recording_spinbox.maximum():
            self.recording_spinbox.setValue(self.recording_spinbox.minimum())
        else:
            self.recording_spinbox.setValue(self.recording_spinbox.value() + self.step_size.value())

    def on_exclude(self):
        selected_recording_id = self.gui.current_session.all_recordings[self.recording_spinbox.value()].id
        logging.info(f"Excluding/including recording ID {selected_recording_id}")
        logging.info(f"Current excluded recordings: {self.gui.current_session.excluded_recordings}")
        if selected_recording_id in self.gui.current_session.excluded_recordings:
            self.exclude_button.setText("Exclude")
            self.gui.restore_recording(selected_recording_id)
            logging.info(f"Restored recording ID {selected_recording_id}")
        else:
            self.exclude_button.setText("Include")
            self.gui.exclude_recording(selected_recording_id)
            logging.info(f"Excluded recording ID {selected_recording_id}")

    def on_recording_changed(self, value):
        max_val = self.recording_spinbox.maximum()
        if max_val >= 0 and value > max_val:
            self.recording_spinbox.blockSignals(True)
            self.recording_spinbox.setValue(max_val)
            self.recording_spinbox.blockSignals(False)
            value = max_val
        self._refresh_exclude_button(value)

        self.gui.plot_controller.plot_data()

    def _refresh_exclude_button(self, recording_index):
        if self.gui.current_session and self.gui.current_session.all_recordings:
            # Translate index -> recording id before checking exclusion list
            if 0 <= recording_index < len(self.gui.current_session.all_recordings):
                rec_id = self.gui.current_session.all_recordings[recording_index].id
                if rec_id in self.gui.current_session.excluded_recordings:
                    self.exclude_button.setText("Include")
                else:
                    self.exclude_button.setText("Exclude")
            else:
                # Out-of-range index: disable exclude button defensively
                self.exclude_button.setEnabled(False)
                logging.warning("Recording index %s out of range after change; disabling exclude button", recording_index)
        else:
            # No active session or no recordings
            self.exclude_button.setEnabled(False)

    def get_current_recording(self):
        return self.recording_spinbox.value()

    def get_excluded_recordings(self):
        return self.gui.current_session.excluded_recordings
