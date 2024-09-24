from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QGridLayout, QPushButton, QSpinBox, QLabel, QGroupBox
from PyQt6.QtGui import QValidator
from monstim_utils import get_main_window
if TYPE_CHECKING:
    from monstim_gui import EMGAnalysisGUI  # noqa: F401

class CustomSpinBox(QSpinBox):
    # Custom SpinBox that wraps around when reaching the maximum or minimum value
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setKeyboardTracking(False)

    def validate(self, text, pos):
        if text == "" or text == "-":
            return QValidator.State.Intermediate, text, pos
        if text.lstrip('-').isdigit():
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
        super().__init__("Cycle Through Recordings", parent)
        self.main_gui = get_main_window() # type: EMGAnalysisGUI
        self.maximum_recordings = self.main_gui.current_session.num_recordings - 1
        
        self.layout = QGridLayout() # type: QGridLayout
        self.setLayout(self.layout)

        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.exclude_button = QPushButton("Exclude")
        self.recording_label = QLabel("Recording:")
        self.recording_spinbox = CustomSpinBox()

        # Set up the CustomSpinBox
        self.recording_spinbox.setMinimum(0)
        self.recording_spinbox.setMaximum(self.maximum_recordings)
        self.recording_spinbox.setWrapping(True)
        

        self.layout.addWidget(self.prev_button, 0, 0)
        self.layout.addWidget(self.next_button, 0, 1)
        self.layout.addWidget(self.recording_label, 1, 0)
        self.layout.addWidget(self.recording_spinbox, 1, 1)
        self.layout.addWidget(self.exclude_button, 1, 2)

        self.prev_button.clicked.connect(self.on_previous)
        self.next_button.clicked.connect(self.on_next)
        self.exclude_button.clicked.connect(self.on_exclude)
        self.recording_spinbox.valueChanged.connect(self.on_recording_changed)

    def reset_max_recordings(self):
        self.maximum_recordings = self.main_gui.current_session.num_recordings - 1
        self.recording_spinbox.setMaximum(self.maximum_recordings)
        if self.recording_spinbox.value() > self.maximum_recordings:
            self.recording_spinbox.setValue(self.maximum_recordings)

    def on_previous(self):
        self.recording_spinbox.setValue(self.recording_spinbox.value() - 1)

    def on_next(self):
        self.recording_spinbox.setValue(self.recording_spinbox.value() + 1)

    def on_exclude(self):
        current = self.recording_spinbox.value()
        if current in self.main_gui.current_session.excluded_recordings:
            self.exclude_button.setText("Exclude")
            self.main_gui.restore_recording(current)
        else:
            self.exclude_button.setText("Include")
            self.main_gui.exclude_recording(current)

    def on_recording_changed(self, value):
        if value in self.main_gui.current_session.excluded_recordings:
            self.exclude_button.setText("Include")
        else:
            self.exclude_button.setText("Exclude")
        self.main_gui.plot_data()

    def get_current_recording(self):
        return self.recording_spinbox.value()

    def get_excluded_recordings(self):
        return self.main_gui.current_session.excluded_recordings