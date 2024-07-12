from typing import TYPE_CHECKING
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QDialogButtonBox, QMessageBox, QHBoxLayout,
                             QTextEdit, QPushButton, QApplication)
from monstim_analysis.Plot_EMG import MatplotlibCanvas

from monstim_utils import DATA_PATH, OUTPUT_PATH, SAVED_DATASETS_PATH  # noqa: F401

if TYPE_CHECKING:
    from monstim_analysis import EMGSession, EMGDataset

class PlotWindowDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plot Window")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout(self)

        self.canvas = MatplotlibCanvas(self)
        self.layout.addWidget(self.canvas)

class ChangeChannelNamesDialog(QDialog):
    def __init__(self, channel_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Channel Names")
        self.setModal(True)
        layout = QGridLayout(self)
        
        self.channel_inputs = {}
        for i, channel_name in enumerate(channel_names):
            layout.addWidget(QLabel(f"Channel {i+1}:"), i, 0)
            self.channel_inputs[channel_name] = QLineEdit(channel_name)
            layout.addWidget(self.channel_inputs[channel_name], i, 1)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, len(channel_names), 0, 1, 2)

    def get_new_names(self):
        return {old: input.text() for old, input in self.channel_inputs.items()}

class ReflexSettingsDialog(QDialog):
    def __init__(self, session : 'EMGSession', dataset : 'EMGDataset', parent=None):
        super().__init__(parent)
        self.session = session
        self.dataset = dataset
        
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"Update Reflex Window Settings: Dataset {self.dataset.name}")
        layout = QVBoxLayout()

        # Duration
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("m_duration:"))
        self.m_duration_entry = QLineEdit(str(self.session.m_end[0] - self.session.m_start[0]))
        duration_layout.addWidget(self.m_duration_entry)

        duration_layout.addWidget(QLabel("h_duration:"))
        self.h_duration_entry = QLineEdit(str(self.session.h_end[0] - self.session.h_start[0]))
        duration_layout.addWidget(self.h_duration_entry)

        layout.addLayout(duration_layout)

        # Start times
        self.entries : list[tuple[QLineEdit, QLineEdit]] = []
        for i in range(self.session.num_channels):
            channel_layout = QHBoxLayout()
            channel_layout.addWidget(QLabel(f"Channel {i}:"))

            channel_layout.addWidget(QLabel("m_start:"))
            m_start_entry = QLineEdit(str(self.session.m_start[i]))
            channel_layout.addWidget(m_start_entry)

            channel_layout.addWidget(QLabel("h_start:"))
            h_start_entry = QLineEdit(str(self.session.h_start[i]))
            channel_layout.addWidget(h_start_entry)

            layout.addLayout(channel_layout)
            self.entries.append((m_start_entry, h_start_entry))

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.save_settings)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def save_settings(self):
        try:
            m_duration = float(self.m_duration_entry.text())
            h_duration = float(self.h_duration_entry.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Invalid input for durations. Please enter valid numbers.")
            return

        m_start = []
        h_start = []
        for i, (m_start_entry, h_start_entry) in enumerate(self.entries):
            try:
                m_start.append(float(m_start_entry.text()))
                h_start.append(float(h_start_entry.text()))
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", f"Invalid input for channel {i}. Skipping.")

        try:           
            self.dataset.set_reflex_settings(m_start, m_duration, h_start, h_duration)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error saving settings: {str(e)}")
            return

        self.accept()

class CopyableReportDialog(QDialog):
    def __init__(self, title, report, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setLayout(QVBoxLayout())

        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(report)
        self.text_edit.setReadOnly(True)
        self.layout().addWidget(self.text_edit)

        button_layout = QHBoxLayout()
        
        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(self.copy_to_clipboard)
        button_layout.addWidget(copy_button)
        
        done_button = QPushButton("Done")
        done_button.clicked.connect(self.close)
        button_layout.addWidget(done_button)
        
        self.layout().addLayout(button_layout)

        self.resize(300, 200)

    def copy_to_clipboard(self):
        QApplication.clipboard().setText(self.text_edit.toPlainText())


