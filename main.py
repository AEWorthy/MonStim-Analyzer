import sys
import os
import pickle
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QFileDialog, QComboBox, QMessageBox, QRadioButton, QButtonGroup,
                             QDialog, QMenuBar, QMenu, QGroupBox, QProgressDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal


from Analyze_EMG import EMGData, EMGDataset, EMGSession
from dialogs import ChangeChannelNamesDialog, ReflexSettingsDialog
from monstim_to_pickle import pickle_data
from monstim_utils import DATA_PATH, OUTPUT_PATH, SAVED_DATASETS_PATH, format_report  # noqa: F401


class EMGAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG Analysis Tool")
        self.setGeometry(100, 100, 600, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(10)  # Set spacing between layout items
        self.layout.setContentsMargins(20, 20, 20, 20)  # Set layout margins

        self.init_ui()

        self.dataset_dict = {}
        self.datasets = []
        self.current_dataset = None
        self.current_session = None
        self.channel_names = []
        self.plot_type_dict = {"EMG": "emg", "Suspected H-reflexes": "suspectedH", "Reflex Curves": "reflexCurves",
                               "M-max": "mmax", "Max H-reflex": "maxH"}

        # Set default paths
        self.csv_path = os.path.join(os.getcwd(), DATA_PATH)
        self.output_path = os.path.join(os.getcwd(), OUTPUT_PATH)

        # Load existing pickled datasets if available
        self.load_existing_datasets()

    def init_ui(self):
        self.create_menu_bar()
        self.create_data_selection_widgets()
        self.create_reports_widgets()
        self.create_plot_widgets()

        # Add a stretching space at the bottom
        self.layout.addStretch(1)

    def create_menu_bar(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        # File menu
        file_menu = QMenu("File", self)
        menu_bar.addMenu(file_menu)

        import_action = file_menu.addAction("Import New Data from CSV Files")
        import_action.triggered.connect(self.import_csv_data)

        # save_action = file_menu.addAction("Save Data")
        # save_action.triggered.connect(self.save_data)

        # load_action = file_menu.addAction("Load Data")
        # load_action.triggered.connect(self.load_data)

        # load existing datasets button
        load_datasets_action = file_menu.addAction("Refresh Datasets/Sessions Lists")
        load_datasets_action.triggered.connect(self.load_existing_datasets)

        # New Edit menu
        edit_menu = QMenu("Edit", self)
        menu_bar.addMenu(edit_menu)

        change_names_action = edit_menu.addAction("Change Channel Names")
        change_names_action.triggered.connect(self.change_channel_names)

        # Update window settings button
        update_window_action = edit_menu.addAction("Update Reflex Time Windows")
        update_window_action.triggered.connect(self.update_reflex_settings)


    def update_reflex_settings(self):
        if self.current_session:
            dialog = ReflexSettingsDialog(self.current_session, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                QMessageBox.information(self, "Success", "Window settings updated successfully.")
        else:
            QMessageBox.warning(self, "Warning", "Please select a session first.")

    def import_csv_data(self):
        self.csv_path = QFileDialog.getExistingDirectory(self, "Select CSV Directory")
        if self.csv_path:
            self.output_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if self.output_path:
                progress_dialog = QProgressDialog("Processing data...", "Cancel", 0, 100, self)
                progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
                progress_dialog.setAutoClose(False)
                progress_dialog.setAutoReset(False)
                progress_dialog.show()

                self.thread = DataProcessingThread(self.csv_path, self.output_path)
                self.thread.progress.connect(progress_dialog.setValue)
                self.thread.finished.connect(progress_dialog.close)
                self.thread.finished.connect(lambda: QMessageBox.information(self, "Success", "Data processed and imported successfully."))
                self.thread.finished.connect(self.load_existing_datasets)
                self.thread.error.connect(lambda e: QMessageBox.critical(self, "Error", f"An error occurred: {e}"))
                self.thread.canceled.connect(lambda: QMessageBox.information(self, "Canceled", "Data processing was canceled."))
                self.thread.start()

                progress_dialog.canceled.connect(self.thread.cancel)
            else:
                QMessageBox.warning(self, "Warning", "You must select an output directory.")
        else:
            QMessageBox.warning(self, "Warning", "You must select a CSV directory.")
        

    def save_data(self):
        if not self.current_dataset and not self.current_session:
            QMessageBox.warning(self, "Warning", "No data to save. Please load a dataset or session first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "Pickle Files (*.pkl)")
        if file_path:
            with open(file_path, 'wb') as f:
                if self.current_dataset:
                    pickle.dump(self.current_dataset, f)
                else:
                    pickle.dump(self.current_session, f)
            QMessageBox.information(self, "Success", "Data saved successfully.")

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "Pickle Files (*.pkl)")
        if file_path:
            with open(file_path, 'rb') as f:
                loaded_data = pickle.load(f)
                if isinstance(loaded_data, EMGDataset):
                    self.current_dataset = loaded_data
                    self.current_session = None
                    self.update_dataset_combo()
                elif isinstance(loaded_data, EMGSession):
                    self.current_session = loaded_data
                    self.current_dataset = None
                    self.update_session_combo()
                else:
                    QMessageBox.warning(self, "Warning", "Invalid data format.")
                    return
            QMessageBox.information(self, "Success", "Data loaded successfully.")

    def update_dataset_combo(self):
        self.dataset_combo.clear()
        if self.current_dataset:
            self.dataset_combo.addItem(self.current_dataset.dataset_name)
            self.dataset_combo.setCurrentIndex(0)
        self.load_dataset(0)

    def update_session_combo(self):
        self.session_combo.clear()
        if self.current_session:
            self.session_combo.addItem(self.current_session.session_id)
            self.session_combo.setCurrentIndex(0)
        self.load_session(0)

    def load_existing_datasets(self):
        if os.path.exists(self.output_path):
            self.dataset_dict, self.datasets = EMGData.unpackPickleOutput(self.output_path)
            self.dataset_combo.clear()
            self.dataset_combo.addItems(self.datasets)

    def change_channel_names(self):
        if not self.channel_names:
            QMessageBox.warning(self, "Warning", "Please load a dataset and session first.")
            return

        dialog = ChangeChannelNamesDialog(self.channel_names, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_names = dialog.get_new_names()
            if new_names:
                # Update channel names in the current session and dataset
                if self.current_dataset:
                    self.current_dataset.rename_channels(new_names)
                elif self.current_session:
                    self.current_session.rename_channels(new_names)
                
                # Update the channel_names list
                self.channel_names = list(new_names.values())
                
                QMessageBox.information(self, "Success", "Channel names updated successfully.")
            else:
                QMessageBox.warning(self, "Warning", "No changes made to channel names.")

    # Data selection pane and its functions.
    def create_data_selection_widgets(self):
        group_box = QGroupBox("Data Selection")
        layout = QVBoxLayout()
        layout.setSpacing(5)  # Set spacing between widgets in this group

        # Dataset selection
        dataset_layout = QHBoxLayout()
        self.dataset_label = QLabel("Select Dataset:")
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        dataset_layout.addWidget(self.dataset_label)
        dataset_layout.addWidget(self.dataset_combo)
        layout.addLayout(dataset_layout)

        # Session selection
        session_layout = QHBoxLayout()
        self.session_label = QLabel("Select Session:")
        self.session_combo = QComboBox()
        self.session_combo.currentIndexChanged.connect(self.load_session)
        session_layout.addWidget(self.session_label)
        session_layout.addWidget(self.session_combo)
        layout.addLayout(session_layout)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def load_dataset(self, index):
        if index >= 0:
            dataset_name = self.datasets[index]
            date, animal_id, condition = EMGDataset.getDatasetInfo(dataset_name)
            self.current_dataset = EMGDataset(self.dataset_dict[dataset_name], date, animal_id, condition)
            self.session_combo.clear()
            self.session_combo.addItems([session.session_id for session in self.current_dataset.emg_sessions])
            
            # Update channel names
            self.channel_names = self.current_dataset.channel_names

    def load_session(self, index):
        if self.current_dataset and index >= 0:
            self.current_session = self.current_dataset.get_session(index)
            
            # Update channel names
            self.channel_names = self.current_session.channel_names

    # Reports pane and its functions.
    def create_reports_widgets(self):
        group_box = QGroupBox("Reports")
        layout = QHBoxLayout()
        layout.setSpacing(10)  # Set spacing between widgets in this group

        # M-max report button
        self.mmax_report_button = QPushButton("M-max Report")
        self.mmax_report_button.clicked.connect(self.show_mmax_report)
        layout.addWidget(self.mmax_report_button)

        # Session Report button
        self.session_report_button = QPushButton("Session Report")
        self.session_report_button.clicked.connect(self.show_session_report)
        layout.addWidget(self.session_report_button)

        # Dataset Report button
        self.dataset_report_button = QPushButton("Dataset Report")
        self.dataset_report_button.clicked.connect(self.show_dataset_report)
        layout.addWidget(self.dataset_report_button)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def show_mmax_report(self):
        if self.current_session:
            report = self.current_session.m_max_report()
            report = format_report(report)
            QMessageBox.information(self, "M-max Report", report)

    def show_session_report(self):
        if self.current_session:
            report = self.current_session.session_parameters()
            report = format_report(report)
            QMessageBox.information(self, "Session Report", report)

    def show_dataset_report(self):
        if self.current_dataset:
            report = self.current_dataset.dataset_parameters()
            report = format_report(report)
            QMessageBox.information(self, "Dataset Report", report)


    # Plots pane and its functions.
    def create_plot_widgets(self):
        group_box = QGroupBox("Plotting")
        layout = QVBoxLayout()
        layout.setSpacing(5)  # Set spacing between widgets in this group

        # Session/Dataset selection
        view_layout = QHBoxLayout()
        self.view_group = QButtonGroup(self)
        self.session_radio = QRadioButton("Single Session")
        self.dataset_radio = QRadioButton("Enitre Dataset")
        self.view_group.addButton(self.session_radio)
        self.view_group.addButton(self.dataset_radio)
        self.session_radio.setChecked(True)
        view_layout.addWidget(self.session_radio)
        view_layout.addWidget(self.dataset_radio)
        layout.addLayout(view_layout)

        self.session_radio.toggled.connect(self.update_plot_types)
        self.dataset_radio.toggled.connect(self.update_plot_types)

        # Plot type selection
        plot_type_layout = QHBoxLayout()
        self.plot_type_label = QLabel("Select Plot Type:")
        self.plot_type_combo = QComboBox()
        plot_type_layout.addWidget(self.plot_type_label)
        plot_type_layout.addWidget(self.plot_type_combo)
        layout.addLayout(plot_type_layout)

        # Plot button
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot_data)
        layout.addWidget(self.plot_button)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

        # Initialize plot types
        self.update_plot_types()

    def update_plot_types(self):
        self.plot_type_combo.clear()
        if self.session_radio.isChecked():
            self.plot_type_combo.addItems(["EMG", "Suspected H-reflexes", "Reflex Curves", "M-max"])
        else:
            self.plot_type_combo.addItems(["Reflex Curves", "Max H-reflex"])

    def plot_data(self):
        # self.plot_window = PlotWindowDialog()
        # self.plot_window.exec()
        # don't forget to add the canvas to the plot args if you use the custom plot window.
        
        plot_type = self.plot_type_dict.get(self.plot_type_combo.currentText())

        if self.session_radio.isChecked():
            if self.current_session:
                self.current_session.plot(plot_type=plot_type)
            else:
                QMessageBox.warning(self, "Warning", "Please select a session first.")
        else:
            if self.current_dataset:
                self.current_dataset.plot(plot_type=plot_type)
            else:
                QMessageBox.warning(self, "Warning", "Please select a dataset first.")

class DataProcessingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(Exception)
    canceled = pyqtSignal()

    def __init__(self, csv_path, output_path):
        super().__init__()
        self.csv_path = csv_path
        self.output_path = output_path
        self._is_canceled = False
        self._is_canceled_handled = False

    def run(self):
        try:
            # Call your data processing function here with progress callback and cancel check
            pickle_data(self.csv_path, self.output_path, self.report_progress, self.is_canceled)
            if not self._is_canceled:
                self.finished.emit()
        except Exception as e:
            if not self._is_canceled:
                self.error.emit(e)

    def report_progress(self, value):
        self.progress.emit(value)

    def cancel(self):
        self._is_canceled = True
        if not self._is_canceled_handled:
            self._is_canceled_handled = True
            self.canceled.emit()

    def is_canceled(self):
        return self._is_canceled



if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = EMGAnalysisGUI()
    gui.show()
    sys.exit(app.exec())