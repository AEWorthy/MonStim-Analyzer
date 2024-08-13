from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QGroupBox, QPushButton, QGridLayout

if TYPE_CHECKING:
    from gui_main import EMGAnalysisGUI

class ReportsWidget(QGroupBox):
    def __init__(self, parent : 'EMGAnalysisGUI'):
        super().__init__("Reports", parent)
        self.parent = parent # type: EMGAnalysisGUI
        self.layout = QGridLayout(self)
        self.create_report_buttons()

    def create_report_buttons(self):
        buttons = [
            ("Session Info. Report", self.parent.show_session_report),
            ("Dataset Info. Report", self.parent.show_dataset_report),
            ("Experiment Info. Report", self.parent.show_experiment_report),
            ("M-max Report (RMS)", self.parent.show_mmax_report),
        ]

        for i, (text, callback) in enumerate(buttons):
            button = QPushButton(text)
            button.clicked.connect(callback)
            row = i // 3
            col = i % 3
            self.layout.addWidget(button, row, col)