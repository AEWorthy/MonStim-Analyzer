import logging
import os
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QDialogButtonBox, QMessageBox, QHBoxLayout,
                             QTextEdit, QPushButton, QApplication, QTextBrowser, QWidget)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt
from monstim_analysis.Plot_EMG import MatplotlibCanvas

from monstim_utils import get_source_path

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
            logging.error(f"Error saving reflex settings: {str(e)}\n\tdataset: {self.dataset}\n\tm_start: {m_start}\n\tm_duration: {m_duration}\n\th_start: {h_start}\n\th_duration: {h_duration}")
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

class HelpWindow(QWidget):
    def __init__(self, markdown_content, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help")
        self.resize(600, 400)
        
        layout = QVBoxLayout()
        self.text_browser = QTextBrowser()
        self.text_browser.setHtml(markdown_content)
        layout.addWidget(self.text_browser)
        self.setLayout(layout)

class InfoDialog(QWidget):
    logging.debug("Showing info dialog")
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Program Information")
        self.setFixedSize(400, 300)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Dialog)

        # Set white background
        self.setStyleSheet("background-color: white;")

        layout = QVBoxLayout(self)

        # Add logo
        logo_pixmap = QPixmap(os.path.join(get_source_path(), 'icon.png'))
        max_width = 100  # Set the desired maximum width
        max_height = 100  # Set the desired maximum height
        logo_pixmap = logo_pixmap.scaled(max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        logo_label = QLabel()
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo_label)

        font = QFont()
        font.setPointSize(12)

        program_name = QLabel("MonStim EMG Analyzer")
        program_name.setStyleSheet("font-weight: bold; color: #333333;")
        program_name.setFont(font)
        program_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(program_name)

        version = QLabel("Version 1.0")
        version.setStyleSheet("color: #666666;")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

        description = QLabel("Software for analyzing EMG data from LabView MonStim experiments.\n\n\nClick to dismiss...")
        description.setStyleSheet("color: #666666;")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)

        copyright = QLabel("© 2024 Andrew Worthy")
        copyright.setStyleSheet("color: #999999;")
        copyright.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(copyright)

        self.setLayout(layout)

    def mousePressEvent(self, event):
        self.close()


