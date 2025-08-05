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
    from monstim_gui import MonstimGUI  # noqa: F401


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

        self.main_gui = get_main_window()  # type: MonstimGUI
        if not self.main_gui.current_session:
            self.maximum_recordings = 0
        else:
            self.maximum_recordings = self.main_gui.current_session.num_recordings - 1

        # Set size policy to be fixed height
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

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
        self.recording_spinbox.setMaximum(self.maximum_recordings)
        self.recording_spinbox.setWrapping(True)

        # Set up the step size spinbox
        self.step_size.setMinimum(1)
        self.step_size.setMaximum(self.maximum_recordings)
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
        if not self.main_gui.current_session:
            self.maximum_recordings = 0
        else:
            self.maximum_recordings = self.main_gui.current_session.num_recordings - 1
        self.recording_spinbox.setMaximum(self.maximum_recordings)
        if self.recording_spinbox.value() > self.maximum_recordings:
            self.recording_spinbox.setValue(self.maximum_recordings)

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
        selected_recording_id = self.main_gui.current_session.recordings[self.recording_spinbox.value()].id
        if selected_recording_id in self.main_gui.current_session.excluded_recordings:
            self.exclude_button.setText("Exclude")
            self.main_gui.restore_recording(selected_recording_id)
        else:
            self.exclude_button.setText("Include")
            self.main_gui.exclude_recording(selected_recording_id)

    def on_recording_changed(self, value):
        if value in self.main_gui.current_session.excluded_recordings:
            self.exclude_button.setText("Include")
        else:
            self.exclude_button.setText("Exclude")
        self.main_gui.plot_controller.plot_data()

    def get_current_recording(self):
        return self.recording_spinbox.value()

    def get_excluded_recordings(self):
        return self.main_gui.current_session.excluded_recordings
