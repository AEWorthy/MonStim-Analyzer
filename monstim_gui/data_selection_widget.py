import logging
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PyQt6.QtCore import Qt

if TYPE_CHECKING:
    from gui_main import EMGAnalysisGUI

class DataSelectionWidget(QGroupBox):
    def __init__(self, parent : 'EMGAnalysisGUI'):
        super().__init__("Data Selection", parent)
        self.parent = parent # type: EMGAnalysisGUI
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(5)
        self.create_experiment_selection()
        self.create_dataset_selection()
        self.create_session_selection()
        self.setLayout(self.layout)

    def create_experiment_selection(self):
        experiment_layout = QHBoxLayout()
        self.experiment_label = QLabel("Select Experiment:")
        self.experiment_combo = QComboBox()
        self.experiment_combo.currentIndexChanged.connect(self.parent.load_experiment)
        experiment_layout.addWidget(self.experiment_label)
        experiment_layout.addWidget(self.experiment_combo)
        self.layout.addLayout(experiment_layout)

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
    
    def update_experiment_combo(self):
        self.experiment_combo.clear()
        if self.parent.expts_dict_keys:
            for experiment in self.parent.expts_dict_keys:
                self.experiment_combo.addItem(experiment)
                index = self.experiment_combo.count() - 1
                self.experiment_combo.setItemData(index, experiment, role=Qt.ItemDataRole.ToolTipRole)
        else:
            logging.warning("Cannot update experiments combo. No experiments loaded.")

    def update_dataset_combo(self):
        self.dataset_combo.clear()
        if self.parent.current_experiment:
            for name in self.parent.current_experiment.dataset_names:
                self.dataset_combo.addItem(name)
                index = self.dataset_combo.count() - 1
                self.dataset_combo.setItemData(index, name, role=Qt.ItemDataRole.ToolTipRole)
        else:
            logging.warning("Cannot update datasets combo. No experiment loaded.")

    def update_session_combo(self):
        self.session_combo.clear()
        if self.parent.current_dataset:
            for session in self.parent.current_dataset.emg_sessions:
                self.session_combo.addItem(session.session_id)
                index = self.session_combo.count() - 1
                self.session_combo.setItemData(index, session.session_id, role=Qt.ItemDataRole.ToolTipRole)
        else:
            logging.warning("Cannot update sessions combo. No dataset loaded.")

    def update_all_data_combos(self):
        self.update_experiment_combo()
        self.update_dataset_combo()
        self.update_session_combo()
    