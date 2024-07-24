from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
from PyQt6.QtCore import Qt

if TYPE_CHECKING:
    from gui_main import EMGAnalysisGUI

class DataSelectionWidget(QGroupBox):
    def __init__(self, parent : 'EMGAnalysisGUI'):
        super().__init__("Data Selection", parent)
        self.parent = parent # type: EMGAnalysisGUI
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(5)
        self.create_dataset_selection()
        self.create_session_selection()
        self.setLayout(self.layout)
        self.create_dataset_options()
        self.update_ui()

    def create_dataset_selection(self):
        dataset_layout = QHBoxLayout()
        self.dataset_label = QLabel("Select Dataset:")
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self.parent.load_dataset)
        dataset_layout.addWidget(self.dataset_label)
        dataset_layout.addWidget(self.dataset_combo)
        self.layout.addLayout(dataset_layout)

    def create_session_selection(self):
        session_layout = QHBoxLayout()
        self.session_label = QLabel("Select Session:")
        self.session_combo = QComboBox()
        self.session_combo.currentIndexChanged.connect(self.parent.load_session)
        session_layout.addWidget(self.session_label)
        session_layout.addWidget(self.session_combo)
        self.layout.addLayout(session_layout)
    
    def update_dataset_combo(self):
        self.dataset_combo.clear()
        if self.parent.datasets:
            for dataset in self.parent.datasets:
                self.dataset_combo.addItem(dataset)
                index = self.dataset_combo.count() - 1
                self.dataset_combo.setItemData(index, dataset, role=Qt.ItemDataRole.ToolTipRole)

    def update_session_combo(self):
        self.session_combo.clear()
        if self.parent.current_dataset:
            for session in self.parent.current_dataset.emg_sessions:
                self.session_combo.addItem(session.session_id)
                index = self.session_combo.count() - 1
                self.session_combo.setItemData(index, session.session_id, role=Qt.ItemDataRole.ToolTipRole)

    def update_ui(self):
        self.update_dataset_combo()
        self.update_session_combo()
    
    def create_dataset_options(self):
        self.dataset_options = QHBoxLayout()
        self.dataset_options.setSpacing(5)

        self.dataset_options.addWidget(QLabel(""))

        # button to remove session from dataset
        self.remove_session_button = QPushButton("Remove Session")
        self.remove_session_button.clicked.connect(self.parent.remove_session)
        self.dataset_options.addWidget(self.remove_session_button)

        # button to reload dataset from file
        self.reload_dataset_button = QPushButton("Reload All Sessions")
        self.reload_dataset_button.clicked.connect(self.parent.reload_dataset)
        self.dataset_options.addWidget(self.reload_dataset_button)


        self.layout.addLayout(self.dataset_options)

