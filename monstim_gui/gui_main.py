import sys
import os
import logging
import traceback
import multiprocessing
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QFileDialog, QMessageBox, 
                             QDialog, QProgressDialog, QSplashScreen, QLabel, QHBoxLayout)
from PyQt6.QtGui import QPixmap, QFont, QIcon
from PyQt6.QtCore import Qt
import markdown

from monstim_analysis import EMGData, EMGDataset#, EMGSession
from monstim_converter import GUIDataProcessingThread
from monstim_utils import (format_report, get_output_path, get_data_path, get_output_bin_path, 
                           get_source_path, get_docs_path, get_config_path)

from .dialogs import (ChangeChannelNamesDialog, ReflexSettingsDialog, CopyableReportDialog, 
                      LatexHelpWindow, InfoDialog, HelpWindow, PreferencesDialog)
from .menu_bar import MenuBar
from .data_selection_widget import DataSelectionWidget
from .reports_widget import ReportsWidget
from .plotting_widget import PlotWidget, PlotPane
from .commands import RemoveSessionCommand, CommandInvoker

class SplashScreen(QSplashScreen):
    def __init__(self):
        logging.debug("Creating splash screen.")
        pixmap = QPixmap(400, 300)
        pixmap.fill(Qt.GlobalColor.white)
        
        super().__init__(pixmap, Qt.WindowType.WindowStaysOnTopHint)
        
        # Add program information
        layout = self.layout()
        if layout is None:
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
        
        description = QLabel("Software for analyzing EMG data\nfrom LabView MonStim experiments.\n\n\nClick to dismiss...")
        description.setStyleSheet("color: #666666;")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)
        
        copyright = QLabel("© 2024 Andrew Worthy")
        copyright.setStyleSheet("color: #999999;")
        copyright.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(copyright)

class EMGAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MonStim Analyzer")
        self.setWindowIcon(QIcon(os.path.join(get_source_path(), 'icon.png')))
        self.setGeometry(100, 100, 1000, 600)
    
        # Initialize variables
        self.dataset_dict = {}
        self.datasets = []
        self.current_dataset = None # Type: EMGDataset
        self.current_session = None # Type: EMGSession
        self.channel_names = []
        self.plot_type_dict = {"EMG": "emg", "Suspected H-reflexes": "suspectedH", "Reflex Curves": "reflexCurves",
                               "M-max": "mmax", "Max H-reflex": "maxH", "Average Reflex Curves": "reflexCurves"}
        
        # Set default paths
        self.csv_path = get_data_path()
        self.output_path = get_output_path()
        self.config_file = get_config_path()

        self.init_ui()

        # Load existing pickled datasets if available
        self.load_existing_datasets()
        self.data_selection_widget.update_ui()

        self.command_invoker = CommandInvoker()
    
    def init_ui(self):
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Create widgets
        self.menu_bar = MenuBar(self)
        self.data_selection_widget = DataSelectionWidget(self)
        self.reports_widget = ReportsWidget(self)
        self.plot_pane = PlotPane(self)
        self.plot_widget = PlotWidget(self)
        
        # Left panel widget to hold existing widgets
        left_panel = QWidget()
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setSpacing(10)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_panel_layout.addWidget(self.data_selection_widget)
        left_panel_layout.addWidget(self.reports_widget)
        left_panel_layout.addWidget(self.plot_widget)
        left_panel_layout.addStretch(1)  # Pushes the plot button to the bottom

        # Add left panel and plot widget to the main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.plot_pane)

        self.setMenuBar(self.menu_bar)
   
    # Command functions
    def undo(self):
        self.command_invoker.undo()

    def redo(self):
        self.command_invoker.redo()

    def remove_session(self):
        command = RemoveSessionCommand(self)
        self.command_invoker.execute(command)

    def load_existing_datasets(self):
        logging.debug("Loading existing datasets.")
        if os.path.exists(self.output_path):
            self.dataset_dict, self.datasets = EMGData.unpackPickleOutput(self.output_path)
    
    # Menu bar functions
    def update_reflex_settings(self):
        logging.debug("Updating reflex window settings.")
        if self.current_session:
            dialog = ReflexSettingsDialog(self.current_session, self.current_dataset, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                QMessageBox.information(self, "Success", "Window settings updated successfully.")
                logging.debug("Window settings updated successfully.")
        else:
            QMessageBox.warning(self, "Warning", "Please select a session first.")

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
                self.thread.finished.connect(self.refresh_existing_datasets)
                
                self.thread.error.connect(lambda e: QMessageBox.critical(self, "Error", f"An error occurred: {e}"))
                self.thread.error.connect(lambda e: logging.error(f"An error occurred while importing CSVs: {e}"))
                self.thread.error.connect(lambda: logging.error(traceback.format_exc()))
                
                self.thread.canceled.connect(progress_dialog.close)
                self.thread.canceled.connect(lambda: QMessageBox.information(self, "Canceled", "Data processing was canceled."))
                self.thread.canceled.connect(lambda: logging.info("Data processing canceled."))
                self.thread.canceled.connect(self.refresh_existing_datasets)
                
                self.thread.start()

                progress_dialog.canceled.connect(self.thread.cancel)
            else:
                QMessageBox.warning(self, "Warning", "You must select an output directory.")
        else:
            QMessageBox.warning(self, "Warning", "You must select a CSV directory.")
    
    def refresh_existing_datasets(self):
        logging.debug("Refreshing existing datasets.")
        self.load_existing_datasets()
        self.data_selection_widget.update_ui()

    def show_preferences_window(self):
        logging.debug("Showing preferences window.")
        window = PreferencesDialog(self.config_file, parent=self)
        if window.exec() == QDialog.DialogCode.Accepted:
            # Apply preferences to data
            if self.current_dataset:
                self.current_dataset.apply_preferences()
            QMessageBox.information(self, "Success", "Preferences applied successfully.")
            logging.debug("Preferences applied successfully.")
        pass

    def change_channel_names(self):
        # Check if a dataset and session are selected
        logging.debug("Changing channel names.")
        if not self.channel_names:
            QMessageBox.warning(self, "Warning", "Please load a dataset and session first.")
            return

        # Open dialog to change channel names
        dialog = ChangeChannelNamesDialog(self.channel_names, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_names = dialog.get_new_names()
            if new_names:
                # Update channel names in the current session and dataset
                if self.current_dataset:
                    self.current_dataset.rename_channels(new_names)
                    self.current_dataset.save_dataset()

                # Update the channel_names list
                self.channel_names = list(new_names.values())
                
                QMessageBox.information(self, "Success", "Channel names updated successfully.")
                logging.debug("Channel names updated successfully.")
            else:
                QMessageBox.warning(self, "Warning", "No changes made to channel names.")
                logging.debug("No changes made to channel names.")

    def show_about_screen(self):
        dialog = InfoDialog(self)
        dialog.show()

    def show_help_dialog(self, topic=None, latex=False):
        logging.debug(f"Showing help dialog for topic: {topic}.")

        match topic:
            case 'help':
                file = 'readme.md'
                title = 'Help'
            case 'emg_processing':
                file = 'Transform_EMG.md'
                latex = True
                title = 'EMG Processing and Analysis Info'
            case _:
                file = 'readme.md'

        file_path = os.path.join(get_docs_path(), file)
        with open(file_path, 'r', encoding='utf-8') as file:
            markdown_content = file.read()
            html_content = markdown.markdown(markdown_content)
            if latex:
                self.help_window = LatexHelpWindow(html_content, title)
            else:
                self.help_window = HelpWindow(html_content, title)
            self.help_window.show()

    # Data selection widget functions
    def load_dataset(self, index):
        if index >= 0:
            dataset_name = self.datasets[index]
            date, animal_id, condition = EMGDataset.getDatasetInfo(dataset_name)  
            save_path = os.path.join(get_output_bin_path(),(f"{date}_{animal_id}_{condition}.pickle"))
            logging.debug(f"Loading dataset: '{dataset_name}'.")

            # Load existing dataset if available
            if os.path.exists(save_path):
                self.current_dataset = EMGDataset.load_dataset(save_path) # Type: EMGDataset
                logging.debug(f"Dataset '{dataset_name}' loaded successfully from '{save_path}'.")
            else:
                self.current_dataset = EMGDataset(self.dataset_dict[dataset_name], date, animal_id, condition) # Type: EMGDataset
                logging.debug(f"Dataset '{dataset_name}' created successfully to '{save_path}'.")         
            # Update channel names
            self.channel_names = self.current_dataset.channel_names
            self.current_dataset.apply_preferences()
            self.data_selection_widget.update_session_combo()

    def load_session(self, index):
        if self.current_dataset and index >= 0:
            self.current_session = self.current_dataset.get_session(index) # Type: EMGSession
            
            # Update channel names
            self.channel_names = self.current_session.channel_names

    def reload_dataset(self):
        self.current_dataset.reload_dataset_sessions()
        self.current_dataset.apply_preferences()
        self.data_selection_widget.update_session_combo()

    # Reports functions.
    def show_mmax_report(self):
        logging.debug("Showing M-max report.")
        if self.current_session:
            report = self.current_session.m_max_report()
            report = format_report(report)
            dialog = CopyableReportDialog("M-max Report (method = RMS)", report, self)
            dialog.exec()
        else:
            QMessageBox.warning(self, "Warning", "Please select a session first.")

    def show_session_report(self):
        logging.debug("Showing session parameters report.")
        if self.current_session:
            report = self.current_session.session_parameters()
            report = format_report(report)
            dialog = CopyableReportDialog("Session Report", report, self)
            dialog.exec()
        else:
            QMessageBox.warning(self, "Warning", "Please select a session first.")

    def show_dataset_report(self):
        logging.debug("Showing dataset parameters report.")
        if self.current_dataset:
            report = self.current_dataset.dataset_parameters()
            report = format_report(report)
            dialog = CopyableReportDialog("Dataset Report", report, self)
            dialog.exec()
        else:
            QMessageBox.warning(self, "Warning", "Please select a dataset first.")

    # Plotting functions
    def plot_data(self):
        self.plot_widget.canvas.show()
        plot_type_raw = self.plot_widget.plot_type_combo.currentText()
        plot_type = self.plot_type_dict.get(plot_type_raw)
        plot_options = self.plot_widget.get_plot_options()
        
        # Plot the data      
        try:
            logging.debug(f'Plotting {plot_type} with options: {plot_options}.')
            if self.plot_widget.session_radio.isChecked():
                if self.current_session:
                    # Assuming plot_type corresponds to the data_type in plot_emg
                    self.current_session.plot(
                        plot_type=plot_type,
                        **plot_options,
                        canvas = self.plot_widget.canvas
                    )
                else:
                    QMessageBox.warning(self, "Warning", "Please select a session first.")
                    return
            else:
                if self.current_dataset:
                    # If you have a similar plot method for dataset, use it here
                    self.current_dataset.plot(
                        plot_type=plot_type,
                        **plot_options,
                        canvas = self.plot_widget.canvas
                    )
                else:
                    QMessageBox.warning(self, "Warning", "Please select a dataset first.")
                    return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
            logging.error(f"An error occurred while plotting: {e}")
            logging.error(f"Plot type: {plot_type}, options: {plot_options}")
            logging.error(f"Current session: {self.current_session}, current dataset: {self.current_dataset}")
            logging.error(traceback.format_exc())
        
        self.plot_pane.layout.update()  # Refresh the layout of the plot pane

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = EMGAnalysisGUI()
    gui.show()
    sys.exit(app.exec())