from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QGroupBox, QPushButton, QSizePolicy

if TYPE_CHECKING:
    from gui_main import MonstimGUI


class ReportsWidget(QGroupBox):
    def __init__(self, parent: "MonstimGUI"):
        super().__init__("Reports", parent)
        self.parent = parent  # type: MonstimGUI
        self.layout = QGridLayout()
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setHorizontalSpacing(12)
        self.layout.setVerticalSpacing(8)
        self.setLayout(self.layout)
        self.create_report_buttons()

    def create_report_buttons(self):
        buttons = [
            (
                "Session Info. Report",
                self.parent.report_manager.show_session_report,
                "Generate a report for the currently selected session including recording statistics and analysis parameters",
            ),
            (
                "Dataset Info. Report",
                self.parent.report_manager.show_dataset_report,
                "Generate a report for the currently selected dataset including all sessions and their summary statistics",
            ),
            (
                "Experiment Info. Report",
                self.parent.report_manager.show_experiment_report,
                "Generate a report for the currently selected experiment including all datasets, sessions, and overall analysis summary",
            ),
            (
                "M-max Report (RMS)",
                self.parent.report_manager.show_mmax_report,
                "Generate a report for the currently selected data level's M-max analysis including RMS values, plateau detection results, and normalization data",
            ),
            # Add new buttons here as tuples in the format:
            # ("Button Label", callback_function, "Tooltip text").
            # Ensure the callback_function is a method of the parent class
            # (EMGAnalysisGUI) and is properly defined.
        ]

        # Number of columns in the grid layout
        N_COLS = 2

        # 1) add each button
        for idx, button_data in enumerate(buttons):
            # Handle both old format (text, callback) and new format (text, callback, tooltip)
            if len(button_data) == 3:
                text, callback, tooltip = button_data
            else:
                text, callback = button_data
                tooltip = f"Generate {text.lower()}"  # Fallback tooltip

            btn = QPushButton(text)
            btn.clicked.connect(callback)
            btn.setToolTip(tooltip)

            # 1) make it expand to fill whatever cell size you're given
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

            row = idx // N_COLS
            col = idx % N_COLS
            self.layout.addWidget(btn, row, col)

        # 2) make *every* column stretch equally
        for c in range(N_COLS):
            self.layout.setColumnStretch(c, 1)

        # 3) pin the whole grid at the top and center horizontally
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
