from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QGroupBox, QHBoxLayout, QPushButton

if TYPE_CHECKING:
    from gui_main import EMGAnalysisGUI

class ReportsWidget(QGroupBox):
    def __init__(self, parent : 'EMGAnalysisGUI'):
        super().__init__("Reports", parent)
        self.parent = parent
        self.layout = QHBoxLayout(self)
        self.create_report_buttons()

    def create_report_buttons(self):
        self.mmax_report_button = QPushButton("M-max Report (RMS)")
        self.mmax_report_button.clicked.connect(self.parent.show_mmax_report)
        self.layout.addWidget(self.mmax_report_button)

        self.session_report_button = QPushButton("Session Info. Report")
        self.session_report_button.clicked.connect(self.parent.show_session_report)
        self.layout.addWidget(self.session_report_button)

        self.dataset_report_button = QPushButton("Dataset Info. Report")
        self.dataset_report_button.clicked.connect(self.parent.show_dataset_report)
        self.layout.addWidget(self.dataset_report_button)