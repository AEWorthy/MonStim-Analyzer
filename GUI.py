import sys
import os
import logging
import multiprocessing
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QFileDialog, QComboBox, QMessageBox, QRadioButton, QButtonGroup, QWidgetItem,
                             QDialog, QMenuBar, QMenu, QGroupBox, QProgressDialog, QCheckBox, QLineEdit, QLayout)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt


from Analyze_EMG import EMGData, EMGDataset#, EMGSession
from dialogs import ChangeChannelNamesDialog, ReflexSettingsDialog
from monstim_to_pickle import GUIDataProcessingThread
from monstim_utils import format_report, get_output_path, get_data_path, load_config
from custom_gui_elements import FloatLineEdit


class EMGAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MonStim Analyzer")
        self.setWindowIcon(QIcon("src/icon.png"))
        self.setGeometry(100, 100, 600, 400)

        self.config : dict = load_config() 

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(10)  # Set spacing between layout items
        self.layout.setContentsMargins(20, 20, 20, 20)  # Set layout margins

        self.dataset_dict = {}
        self.datasets = []
        self.current_dataset = None
        self.current_session = None
        self.channel_names = []
        self.plot_type_dict = {"EMG": "emg", "Suspected H-reflexes": "suspectedH", "Reflex Curves": "reflexCurves",
                               "M-max": "mmax", "Max H-reflex": "maxH"}
        self.plot_options = {"EMG": self.create_emg_options,
                             "Suspected H-reflexes": self.create_suspected_h_reflexes_options,
                             "Max H-reflex": self.create_max_h_reflex_options,
                             "Reflex Curves": self.create_reflex_curves_options,
                             "M-max": self.create_mmax_options}
        
        # Set default paths
        self.csv_path = get_data_path()
        self.output_path = get_output_path()
        
        self.init_ui()
       
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
        logging.debug("Updating reflex window settings.")
        if self.current_session:
            dialog = ReflexSettingsDialog(self.current_session, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                QMessageBox.information(self, "Success", "Window settings updated successfully.")
                logging.debug("Window settings updated successfully.")
        else:
            QMessageBox.warning(self, "Warning", "Please select a session first.")
            logging.warning("No session selected for updating reflex window settings.")

    def import_csv_data(self):
        logging.debug("Importing new data from CSV files.")
        self.csv_path = QFileDialog.getExistingDirectory(self, "Select CSV Directory", self.csv_path)
        if self.csv_path:
            # self.output_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if self.output_path:
                progress_dialog = QProgressDialog("Processing data...", "Cancel", 0, 100, self)
                progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
                progress_dialog.setAutoClose(False)
                progress_dialog.setAutoReset(False)
                progress_dialog.show()

                max_workers = max(1, multiprocessing.cpu_count() - 1)

                self.thread = GUIDataProcessingThread(self.csv_path, self.output_path, max_workers=max_workers)
                self.thread.progress.connect(progress_dialog.setValue)
                
                self.thread.finished.connect(progress_dialog.close)
                self.thread.finished.connect(lambda: QMessageBox.information(self, "Success", "Data processed and imported successfully."))
                self.thread.finished.connect(lambda: logging.info("Data processed and imported successfully."))
                self.thread.finished.connect(self.load_existing_datasets)
                
                
                self.thread.error.connect(lambda e: QMessageBox.critical(self, "Error", f"An error occurred: {e}"))
                self.thread.error.connect(lambda: logging.error)
                
                self.thread.canceled.connect(progress_dialog.close)
                self.thread.canceled.connect(lambda: QMessageBox.information(self, "Canceled", "Data processing was canceled."))
                self.thread.canceled.connect(lambda: logging.info("Data processing canceled."))
                self.thread.canceled.connect(self.load_existing_datasets)
                
                self.thread.start()

                progress_dialog.canceled.connect(self.thread.cancel)
            else:
                logging.warning("No output directory selected.")
                QMessageBox.warning(self, "Warning", "You must select an output directory.")
        else:
            logging.warning("No CSV directory selected.")
            QMessageBox.warning(self, "Warning", "You must select a CSV directory.")
        

    # def save_data(self):
    #     if not self.current_dataset and not self.current_session:
    #         QMessageBox.warning(self, "Warning", "No data to save. Please load a dataset or session first.")
    #         return

    #     file_path, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "Pickle Files (*.pkl)")
    #     if file_path:
    #         with open(file_path, 'wb') as f:
    #             if self.current_dataset:
    #                 pickle.dump(self.current_dataset, f)
    #             else:
    #                 pickle.dump(self.current_session, f)
    #         QMessageBox.information(self, "Success", "Data saved successfully.")

    # def load_data(self):
    #     file_path, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "Pickle Files (*.pkl)")
    #     if file_path:
    #         with open(file_path, 'rb') as f:
    #             loaded_data = pickle.load(f)
    #             if isinstance(loaded_data, EMGDataset):
    #                 self.current_dataset = loaded_data
    #                 self.current_session = None
    #                 self.update_dataset_combo()
    #             elif isinstance(loaded_data, EMGSession):
    #                 self.current_session = loaded_data
    #                 self.current_dataset = None
    #                 self.update_session_combo()
    #             else:
    #                 QMessageBox.warning(self, "Warning", "Invalid data format.")
    #                 return
    #         QMessageBox.information(self, "Success", "Data loaded successfully.")

    def update_dataset_combo(self):
        logging.debug("Updating dataset combo.")
        self.dataset_combo.clear()
        if self.current_dataset:
            self.dataset_combo.addItem(self.current_dataset.dataset_name)
            self.dataset_combo.setCurrentIndex(0)
        self.load_dataset(0)

    def update_session_combo(self):
        logging.debug("Updating session combo.")
        self.session_combo.clear()
        if self.current_session:
            self.session_combo.addItem(self.current_session.session_id)
            self.session_combo.setCurrentIndex(0)
        self.load_session(0)

    def load_existing_datasets(self):
        logging.debug("Loading existing datasets.")
        if os.path.exists(self.output_path):
            self.dataset_dict, self.datasets = EMGData.unpackPickleOutput(self.output_path)
            self.dataset_combo.clear()
            self.dataset_combo.addItems(self.datasets)

    def change_channel_names(self):
        logging.debug("Changing channel names.")
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
                logging.debug("Channel names updated successfully.")
            else:
                QMessageBox.warning(self, "Warning", "No changes made to channel names.")
                logging.debug("No changes made to channel names.")

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
        self.mmax_report_button = QPushButton("M-max Report (RMS)")
        self.mmax_report_button.clicked.connect(self.show_mmax_report)
        layout.addWidget(self.mmax_report_button)

        # Session Report button
        self.session_report_button = QPushButton("Session Info. Report")
        self.session_report_button.clicked.connect(self.show_session_report)
        layout.addWidget(self.session_report_button)

        # Dataset Report button
        self.dataset_report_button = QPushButton("Dataset Info. Report")
        self.dataset_report_button.clicked.connect(self.show_dataset_report)
        layout.addWidget(self.dataset_report_button)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def show_mmax_report(self):
        logging.debug("Showing M-max report.")
        if self.current_session:
            report = self.current_session.m_max_report()
            report = format_report(report)
            QMessageBox.information(self, "M-max Report (method = RMS)", report)

    def show_session_report(self):
        logging.debug("Showing session parameters report.")
        if self.current_session:
            report = self.current_session.session_parameters()
            report = format_report(report)
            QMessageBox.information(self, "Session Report", report)

    def show_dataset_report(self):
        logging.debug("Showing dataset parameters report.")
        if self.current_dataset:
            report = self.current_dataset.dataset_parameters()
            report = format_report(report)
            QMessageBox.information(self, "Dataset Report", report)

    # Plots pane and its functions.
    def create_plot_widgets(self):
        group_box = QGroupBox("Plotting")
        layout = QVBoxLayout()
        layout.setSpacing(5)

        # Session/Dataset selection
        view_layout = QHBoxLayout()
        self.view_group = QButtonGroup(self)
        self.session_radio = QRadioButton("Single Session")
        self.dataset_radio = QRadioButton("Entire Dataset")
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
        self.plot_type_combo.currentTextChanged.connect(self.update_plot_options)
        plot_type_layout.addWidget(self.plot_type_label)
        plot_type_layout.addWidget(self.plot_type_combo)
        layout.addLayout(plot_type_layout)

        # Additional options area
        self.additional_options_layout = QVBoxLayout()
        layout.addLayout(self.additional_options_layout)

        # Initialize plot types
        self.update_plot_types()

        # Plot button
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot_data)
        layout.addWidget(self.plot_button)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def update_plot_types(self):
        self.plot_type_combo.clear()
        if self.session_radio.isChecked():
            self.plot_type_combo.addItems(["EMG", "Suspected H-reflexes", "Reflex Curves", "M-max"])
        else:
            self.plot_type_combo.addItems(["Reflex Curves", "Max H-reflex"])

    def update_plot_options(self):
        # Clear existing widgets in the additional options layout
        self.clear_plot_options()

        # Get the current plot type
        plot_type = self.plot_type_combo.currentText()

        # If there are options for this plot type, create them
        if plot_type in self.plot_options:
            self.plot_options[plot_type]()

        self.additional_options_layout.update()
        
    def clear_plot_options(self):
        while self.additional_options_layout.count():
            item = self.additional_options_layout.takeAt(0)
            if isinstance(item, QWidgetItem):
                item.widget().deleteLater()
            elif isinstance(item, QLayout):
                self.clear_layout(item)
    
    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if isinstance(item, QWidgetItem):
                item.widget().deleteLater()
            elif isinstance(item, QLayout):
                self.clear_layout(item)

    def create_emg_options(self):
        # data type option
        emg_data_type_label = QLabel("Select EMG Data Type:")
        emg_data_type_combo = QComboBox()
        emg_data_type_combo.setObjectName("data_type")
        emg_data_type_combo.addItems(["filtered", "raw", "rectified_raw", "rectified_filtered"])

        # data options layout
        data_options_layout = QHBoxLayout()
        data_options_layout.addWidget(emg_data_type_label)
        data_options_layout.addWidget(emg_data_type_combo)

        # Create a horizontal layout for the flag options
        flags_layout = QHBoxLayout()

        # m_flags option (checkbox)
        m_flags_label = QLabel("Show M Flags:")
        m_flags_checkbox = QCheckBox()
        m_flags_checkbox.setObjectName("m_flags")
        m_flags_checkbox.setChecked(True)  # Set the initial state to True
        flags_layout.addWidget(m_flags_label)
        flags_layout.addWidget(m_flags_checkbox)
        flags_layout.addSpacing(30)

        # h_flags option (checkbox)
        h_flags_label = QLabel("Show H Flags:")
        h_flags_checkbox = QCheckBox()
        h_flags_checkbox.setObjectName("h_flags")
        h_flags_checkbox.setChecked(True)  # Set the initial state to True
        flags_layout.addWidget(h_flags_label)
        flags_layout.addWidget(h_flags_checkbox)

        # Add some stretching to layouts
        data_options_layout.addStretch(1)
        data_options_layout.addSpacing(10)
        flags_layout.addStretch(1)
        

        # Add the widgets to the layout
        self.additional_options_layout.addLayout(data_options_layout)
        self.additional_options_layout.addLayout(flags_layout)

    def create_suspected_h_reflexes_options(self):
        # H threshold option
        h_threshold_label = QLabel("H Threshold:")
        h_threshold_input = FloatLineEdit(default_value=self.config.get("h_threshold", 0.5))
        h_threshold_input.setObjectName("h_threshold")
        h_threshold_input.setPlaceholderText("H-relex Threshold (mV):")

        # H-reflex calculation method option
        h_method_label = QLabel("H-reflex Calculation Method:")
        h_method_combo = QComboBox()
        h_method_combo.setObjectName("method")
        h_method_combo.addItems(["rms", "average_rectified", "average_unrectified", "peak_to_trough"])
        h_method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        
        # Create a horizontal layout for the threshold option
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(h_threshold_label)
        threshold_layout.addWidget(h_threshold_input)
        threshold_layout.addWidget(h_method_label)
        threshold_layout.addWidget(h_method_combo)

        # plot_legend option (checkbox)
        plot_legend_label = QLabel("Show Plot Legend:")
        plot_legend_checkbox = QCheckBox()
        plot_legend_checkbox.setObjectName("plot_legend")
        plot_legend_checkbox.setChecked(True)  # Set the initial state to True
        
        
        self.additional_options_layout.addLayout(threshold_layout)
        self.additional_options_layout.addWidget(plot_legend_label)
        self.additional_options_layout.addWidget(plot_legend_checkbox)

    def create_reflex_curves_options(self):
        # method option
        method_label = QLabel("Reflex Amplitude Calculation Method:")
        method_combo = QComboBox()
        method_combo.setObjectName("method")
        method_combo.addItems(["rms", "average_rectified", "average_unrectified", "peak_to_trough"])
        method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"

        # Create a horizontal layout for the method option
        method_layout = QHBoxLayout()
        method_layout.setSpacing(10)
        method_layout.addWidget(method_label)
        method_layout.addWidget(method_combo)
        method_layout.addStretch(1)

        # relative_to_mmax checkbox
        relative_to_mmax_label = QLabel("Relative to M-max:")
        relative_to_mmax_checkbox = QCheckBox()
        relative_to_mmax_checkbox.setObjectName("relative_to_mmax")
        relative_to_mmax_checkbox.setChecked(False)  # Set the initial state to False

        # relative_to_mmax layout
        relative_to_mmax_layout = QHBoxLayout()
        relative_to_mmax_layout.setSpacing(10)
        relative_to_mmax_layout.addWidget(relative_to_mmax_label)
        relative_to_mmax_layout.addWidget(relative_to_mmax_checkbox)
        relative_to_mmax_layout.addStretch(1)

        # Add the widgets to the layout
        self.additional_options_layout.addLayout(method_layout)
        self.additional_options_layout.addLayout(relative_to_mmax_layout)

    def create_mmax_options(self):
        # method option
        method_label = QLabel("Reflex Amplitude Calculation Method:")
        method_combo = QComboBox()
        method_combo.setObjectName("method")
        method_combo.addItems(["rms", "average_rectified", "average_unrectified", "peak_to_trough"])
        method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"

        # Create a horizontal layout for the method option
        method_layout = QHBoxLayout()
        method_layout.setSpacing(10)
        method_layout.addWidget(method_label)
        method_layout.addWidget(method_combo)
        method_layout.addStretch(1)

        # add to additional options layout
        self.additional_options_layout.addLayout(method_layout)

    def create_max_h_reflex_options(self):
        # method option
        method_label = QLabel("Reflex Amplitude Calculation Method:")
        method_combo = QComboBox()
        method_combo.setObjectName("method")
        method_combo.addItems(["rms", "average_rectified", "average_unrectified", "peak_to_trough"])
        method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"

        # Create a horizontal layout for the method option
        method_layout = QHBoxLayout()
        method_layout.setSpacing(10)
        method_layout.addWidget(method_label)
        method_layout.addWidget(method_combo)
        method_layout.addStretch(1)

        # relative_to_mmax checkbox
        relative_to_mmax_label = QLabel("Relative to M-max:")
        relative_to_mmax_checkbox = QCheckBox()
        relative_to_mmax_checkbox.setObjectName("relative_to_mmax")
        relative_to_mmax_checkbox.setChecked(False)  # Set the initial state to False

        # relative_to_mmax layout
        relative_to_mmax_layout = QHBoxLayout()
        relative_to_mmax_layout.setSpacing(10)
        relative_to_mmax_layout.addWidget(relative_to_mmax_label)
        relative_to_mmax_layout.addWidget(relative_to_mmax_checkbox)
        relative_to_mmax_layout.addStretch(1)

        # Add the widgets to the layout
        self.additional_options_layout.addLayout(method_layout)
        self.additional_options_layout.addLayout(relative_to_mmax_layout)

    def plot_data(self):
        plot_type_raw = self.plot_type_combo.currentText()
        plot_type = self.plot_type_dict.get(plot_type_raw)

        # Gather additional kwarg options based on selected plot type
        kwargs = {}

        # Get the values from the additional options widgets
        for i in range(self.additional_options_layout.count()):
            item = self.additional_options_layout.itemAt(i)
            if isinstance(item, QHBoxLayout):
                # Handle the horizontal layout
                for j in range(item.count()):
                    widget = item.itemAt(j).widget()
                    if isinstance(widget, (QComboBox, QLineEdit, QCheckBox)):
                        value = self.get_widget_value(widget)
                        if value is not None:
                            kwargs[widget.objectName()] = value
            elif isinstance(item, QWidgetItem):
                widget = item.widget()
                if isinstance(widget, (QComboBox, QLineEdit, QCheckBox)):
                    value = self.get_widget_value(widget)
                    if value is not None:
                        kwargs[widget.objectName()] = value

        try:
            logging.debug(f'Plotting {plot_type} with kwargs: {kwargs}.')
            if self.session_radio.isChecked():
                if self.current_session:
                    self.current_session.plot(plot_type=plot_type, **kwargs)
                else:
                    QMessageBox.warning(self, "Warning", "Please select a session first.")
            else:
                if self.current_dataset:
                    self.current_dataset.plot(plot_type=plot_type, **kwargs)
                else:
                    QMessageBox.warning(self, "Warning", "Please select a dataset first.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
    
    def get_widget_value(self, widget):
        if isinstance(widget, QComboBox):
            return widget.currentText()
        elif isinstance(widget, FloatLineEdit):
            return widget.value()
        elif isinstance(widget, QLineEdit):
            return widget.text()
        elif isinstance(widget, QCheckBox):
            return widget.isChecked()
        return None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = EMGAnalysisGUI()
    gui.show()
    sys.exit(app.exec())