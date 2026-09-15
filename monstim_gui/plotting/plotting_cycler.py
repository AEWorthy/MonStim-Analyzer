import logging

logger = logging.getLogger(__name__)
from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QValidator
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
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


def _chevron_icon(points: tuple[tuple[float, float], ...]) -> QIcon:
    """Create a small, crisp navigation chevron for the recording cycler."""
    icon_size = QSize(16, 16)
    pixmap = QPixmap(icon_size)
    pixmap.fill(Qt.GlobalColor.transparent)

    path = QPainterPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(
        QPen(
            QColor("#cbd4dc"),
            1.8,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
    )
    painter.drawPath(path)
    painter.end()
    return QIcon(pixmap)


class RecordingCyclerWidget(QGroupBox):
    def __init__(self, parent):
        super().__init__("Recording Cycler", parent)

        self.gui: MonstimGUI = get_main_window()
        if not self.gui or not self.gui.current_session:
            self.max_recording_idxs = 0
        else:
            self.max_recording_idxs = self.gui.current_session.num_all_recordings - 1

        # Set size policy to be fixed height
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)

        self.prev_button = QToolButton()
        self.prev_button.setIcon(_chevron_icon(((10.5, 3.5), (5.5, 8.0), (10.5, 12.5))))
        self.prev_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.prev_button.setToolTip("Previous recording")
        self.prev_button.setAccessibleName("Previous recording")
        self.next_button = QToolButton()
        self.next_button.setIcon(_chevron_icon(((5.5, 3.5), (10.5, 8.0), (5.5, 12.5))))
        self.next_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.next_button.setToolTip("Next recording")
        self.next_button.setAccessibleName("Next recording")
        self.exclude_button = QPushButton("Exclude")
        self.recording_spinbox = CustomSpinBox()
        self.step_size = CustomSpinBox()

        # Use full-width navigation buttons instead of text glyphs.  Qt draws
        # QToolButton arrows as centered control primitives, so their visual
        # position is independent of font metrics and DPI.
        for button in (self.prev_button, self.next_button):
            button.setAutoRaise(False)
            button.setIconSize(QSize(16, 16))
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setStyleSheet(
                "QToolButton { padding: 0px; color: #cbd4dc; background: #2b2f33; "
                "border: 1px solid #4a5159; border-radius: 4px; }"
                "QToolButton:hover { color: #ffffff; background: #394149; border-color: #65717c; }"
                "QToolButton:pressed { background: #304b5d; border-color: #6d9fbe; }"
                "QToolButton:disabled { color: #717a82; background: #25282b; border-color: #373c41; }"
            )
        self.exclude_button.setProperty("plotOptionToggle", True)

        control_height = self.exclude_button.sizeHint().height()
        for control in (
            self.prev_button,
            self.next_button,
            self.exclude_button,
            self.recording_spinbox,
            self.step_size,
        ):
            control.setFixedHeight(control_height)

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
        vbox.setSpacing(4)
        # Keep the controls close to the group-box title.  The group box
        # already reserves space for its frame/title, so an additional 8 px
        # top margin makes the first row look unnecessarily detached.
        self.layout = vbox
        self.setLayout(vbox)
        # QGroupBox applies its default margins during setLayout(); set the
        # compact margins afterward so the first row stays close to the title.
        vbox.setContentsMargins(8, 0, 8, 8)

        self.prev_button.clicked.connect(self.on_previous)
        self.next_button.clicked.connect(self.on_next)
        self.exclude_button.clicked.connect(self.on_exclude)
        self.recording_spinbox.valueChanged.connect(self.on_recording_changed)

        # The initial recording value does not emit ``valueChanged`` when the
        # widget is created.  Synchronize the button immediately so a newly
        # selected Single EMG plot reflects the current recording's state.
        self._refresh_exclude_button(self.recording_spinbox.value())

    def reset_max_recordings(self):
        if not self.gui or not self.gui.current_session:
            self.max_recording_idxs = 0
        else:
            self.max_recording_idxs = max(0, self.gui.current_session.num_all_recordings - 1)
        # Block signals so we don't trigger plot updates with transient invalid values
        self.recording_spinbox.blockSignals(True)
        old_val = self.recording_spinbox.value()
        self.recording_spinbox.setMaximum(self.max_recording_idxs)
        if old_val > self.max_recording_idxs:
            # Wrap to 0 (expected UX) rather than clamping to last
            self.recording_spinbox.setValue(0)
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
        if not self.gui or not self.gui.current_session:
            return
        selected_recording_id = self.gui.current_session.all_recordings[self.recording_spinbox.value()].id
        logger.info(f"Excluding/including recording ID {selected_recording_id}")
        logger.info(f"Current excluded recordings: {self.gui.current_session.excluded_recordings}")
        if selected_recording_id in self.gui.current_session.excluded_recordings:
            self.exclude_button.setText("Exclude")
            self.exclude_button.setToolTip("Click to exclude this recording from plots")
            self.gui.restore_recording(selected_recording_id)
            logger.info(f"Restored recording ID {selected_recording_id}")
        else:
            self.exclude_button.setText("Include")
            self.exclude_button.setToolTip("Click to include this recording in plots")
            self.gui.exclude_recording(selected_recording_id)
            logger.info(f"Excluded recording ID {selected_recording_id}")

    def on_recording_changed(self, value):
        max_val = self.recording_spinbox.maximum()
        if max_val >= 0 and value > max_val:
            self.recording_spinbox.blockSignals(True)
            self.recording_spinbox.setValue(max_val)
            self.recording_spinbox.blockSignals(False)
            value = max_val
        self._refresh_exclude_button(value)

        if self.gui and self.gui.plot_controller:
            self.gui.plot_controller.plot_data()

    def _refresh_exclude_button(self, recording_index):
        if self.gui and self.gui.current_session and self.gui.current_session.all_recordings:
            # Translate index -> recording id before checking exclusion list
            if 0 <= recording_index < len(self.gui.current_session.all_recordings):
                rec_id = self.gui.current_session.all_recordings[recording_index].id
                self.exclude_button.setEnabled(True)
                if rec_id in self.gui.current_session.excluded_recordings:
                    self.exclude_button.setText("Include")
                    self.exclude_button.setToolTip("Click to include this recording in plots")
                else:
                    self.exclude_button.setText("Exclude")
                    self.exclude_button.setToolTip("Click to exclude this recording from plots")
            else:
                # Out-of-range index: disable exclude button defensively
                self.exclude_button.setEnabled(False)
                logger.warning("Recording index %s out of range after change; disabling exclude button", recording_index)
        else:
            # No active session or no recordings
            self.exclude_button.setEnabled(False)

    def get_current_recording(self):
        return self.recording_spinbox.value()

    def get_excluded_recordings(self):
        return self.gui.current_session.excluded_recordings
