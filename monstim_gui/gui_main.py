import sys
import os
import re
import shutil
import logging
import traceback
import multiprocessing

import markdown
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QFileDialog, QMessageBox, 
                             QDialog, QProgressDialog, QHBoxLayout, QStatusBar)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt

# Adds the parent directory to the path so that the monstim_analysis and monstim_utils modules can be imported
if __name__ == '__main__':
    top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if top_level_dir not in sys.path:
        print(f"Adding {top_level_dir} to sys.path.")
        sys.path.insert(0, top_level_dir)
from monstim_analysis import EMGExperiment
from monstim_converter import GUIExptImportingThread
from monstim_gui.splash import SPLASH_INFO
from monstim_utils import (format_report, get_output_path, get_data_path, get_output_bin_path, 
                           get_source_path, get_docs_path, get_config_path)
from monstim_gui.dialogs import (ChangeChannelNamesDialog, ReflexSettingsDialog, CopyableReportDialog, 
                                 LatexHelpWindow, AboutDialog, HelpWindow, PreferencesDialog, InvertChannelPolarityDialog)
from monstim_gui.menu_bar import MenuBar
from monstim_gui.data_selection_widget import DataSelectionWidget
from monstim_gui.reports_widget import ReportsWidget
from monstim_gui.plotting_widget import PlotWidget, PlotPane
from monstim_gui.commands import (RemoveSessionCommand, CommandInvoker, ExcludeRecordingCommand, 
                       RestoreRecordingCommand, InvertChannelPolarityCommand)
from monstim_gui.dataframe_exporter import DataFrameDialog

class EMGAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MonStim Analyzer")
        self.setWindowIcon(QIcon(os.path.join(get_source_path(), 'icon.png')))
        self.setGeometry(100, 100, 1000, 600)
    
        # Initialize variables
        self.expts_dict = {}
        self.expts_dict_keys = [] # Type: List[str]
        self.current_experiment = None # Type: EMGExperiment
        self.current_dataset = None # Type: EMGDataset
        self.current_session = None # Type: EMGSession
        self.channel_names = []
        self.plot_type_dict = {"EMG": "emg", "Suspected H-reflexes": "suspectedH", "Reflex Curves": "reflexCurves",
                               "M-max": "mmax", "Max H-reflex": "maxH", "Average Reflex Curves": "reflexCurves",
                               "Single EMG Recordings": "singleEMG"}
        
        # Set default paths
        self.output_path = get_output_path()
        self.config_file = get_config_path()

        self.init_ui()

        # Load existing pickled experiments if available
        self.unpack_existing_experiments()
        self.data_selection_widget.update_experiment_combo()

        self.command_invoker = CommandInvoker(self)
    
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

        # Add status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Add a temporary message to visualize the status bar
        self.status_bar.showMessage(f"Welcome to MonStim Analyzer, {SPLASH_INFO['version']}", 10000)
   
    # Command functions
    def undo(self):
        self.command_invoker.undo()

    def redo(self):
        self.command_invoker.redo()

    def exclude_recording(self, recording_index):
        command = ExcludeRecordingCommand(self, recording_index)
        self.command_invoker.execute(command)
    
    def restore_recording(self, recording_index):
        command = RestoreRecordingCommand(self, recording_index)
        self.command_invoker.execute(command)

    def remove_session(self):
        command = RemoveSessionCommand(self)
        self.command_invoker.execute(command)

    def unpack_existing_experiments(self):
        logging.debug("Unpacking existing experiments.")
        if os.path.exists(self.output_path):
            try:
                self.expts_dict = EMGExperiment.unpackPickleOutput(self.output_path)
                self.expts_dict_keys = list(self.expts_dict.keys())
                logging.debug("Existing experiments unpacked successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while unpacking existing experiments: {e}")
                logging.error(f"An error occurred while unpacking existing experiments: {e}")
                logging.error(traceback.format_exc())
    
    # Menu bar functions
    def change_reflex_window_settings(self):
        logging.debug("Updating reflex window settings.")
        if self.current_session and self.current_dataset:
            dialog = ReflexSettingsDialog(self.current_session, self.current_dataset, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.current_experiment.save_experiment()
                self.status_bar.showMessage("Window settings updated successfully.", 5000)  # Show message for 5 seconds
                logging.debug("Window settings updated successfully.")
        else:
            QMessageBox.warning(self, "Warning", "Please select a session first.")

    def invert_channel_polarity(self, channel_indexes_to_invert):
        logging.debug("Inverting channel polarity.")

        if self.current_dataset:
            dialog = InvertChannelPolarityDialog(self.current_dataset, self)

            if dialog.exec():  # Show the dialog and wait for the user's response
                channel_indexes_to_invert = dialog.get_selected_channel_indexes()
                if not channel_indexes_to_invert:
                    QMessageBox.warning(self, "Warning", "Please select at least one channel to invert.")
                    return
                else:
                    command = InvertChannelPolarityCommand(self, channel_indexes_to_invert)
                    self.command_invoker.execute(command)
                    self.status_bar.showMessage("Channel polarity inverted successfully.", 5000)
        else:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")

    def import_expt_data(self):
        logging.info("Importing new experiment data from CSV files.")
        expt_path = QFileDialog.getExistingDirectory(self, "Select Experiment Directory", get_data_path())
        expt_name = os.path.splitext(os.path.basename(expt_path))[0]

        if expt_path and expt_name:
            # Ensure that the experiment directory does not already exist. If so, prompt the user to confirm overwriting.
            if os.path.exists(os.path.join(self.output_path, expt_name)):
                overwrite = QMessageBox.question(self, "Warning", "This experiment already exists in your 'data' folder. Do you want to continue the importation process and overwrite the existing data?\n\nNote: This will also reset and changes you made to the datasets in this experiment (e.g., channel names, latency time windows, etc.)", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if overwrite == QMessageBox.StandardButton.Yes: # Overwrite existing experiment.
                    logging.info(f"Overwriting existing experiment '{expt_name}' in the output folder.")                

                    # Delete existing experiment folder in the data output folder.
                    shutil.rmtree(os.path.join(self.output_path, expt_name))
                    logging.info(f"Deleted existing experiment '{expt_name}' in 'data' folder.")

                    # Delete existing bin file in the output folder.
                    logging.info(f"Checking for bin file: {os.path.join(get_output_bin_path(),(f'{expt_name}.pickle'))}.")
                    bin_file = os.path.join(get_output_bin_path(),(f"{expt_name}.pickle"))
                    if os.path.exists(bin_file):
                        os.remove(bin_file)
                        logging.info(f"Deleted bin file: {bin_file}.")
                    else:
                        logging.info(f"Bin file not found: {bin_file}. Could not delete existing experiment if it exists.")  
                else: # Cancel importing the experiment.
                    logging.info(f"User chose not to overwrite existing experiment '{expt_name}' in the output folder.")
                    QMessageBox.warning(self, "Canceled", "The importation of your data was canceled.")
                    return
                
            # Validate the dataset dir names and confirm that each dataset dir contains at least one .csv file.
            try:
                dataset_dirs_without_csv = []
                for dataset_dir in os.listdir(expt_path):
                    dataset_path = os.path.join(expt_path, dataset_dir)
                    if os.path.isdir(dataset_path):
                        self.validate_dataset_name(dataset_path)

                        # Check if the dataset directory contains any .csv files.
                        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
                        if not csv_files:
                            dataset_dirs_without_csv.append(dataset_dir)
                
                if dataset_dirs_without_csv:
                    raise FileNotFoundError(f"The following dataset directories do not contain any .csv files: {dataset_dirs_without_csv}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while validating your experiment: {e}.\n\nImportation was canceled.")
                logging.error(f"An error occurred while validating dataset names: {e}. Importation was canceled.")
                logging.error(traceback.format_exc())
                return


            # Create a progress dialog to show the importing progress.
            progress_dialog = QProgressDialog("Processing data...", "Cancel", 0, 100, self)
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dialog.setAutoClose(False)
            progress_dialog.setAutoReset(False)
            progress_dialog.show()

            # Initialize the importing thread.
            max_workers = max(1, multiprocessing.cpu_count() - 1)
            self.thread : GUIExptImportingThread = GUIExptImportingThread(expt_name, expt_path, self.output_path, max_workers=max_workers)
            self.thread.progress.connect(progress_dialog.setValue)
            
            # Finished signal.
            self.thread.finished.connect(progress_dialog.close)
            self.thread.finished.connect(self.refresh_existing_experiments)
            # self.thread.finished.connect(lambda: QMessageBox.information(self, "Success", "Data processed and imported successfully."))
            self.thread.finished.connect(lambda: self.status_bar.showMessage("Data processed and imported successfully.", 5000))  # Show message for 5 seconds
            self.thread.finished.connect(lambda: logging.info("Data processed and imported successfully."))
            
            # Error signal.
            self.thread.error.connect(lambda e: QMessageBox.critical(self, "Error", f"An error occurred: {e}"))
            self.thread.error.connect(lambda e: logging.error(f"An error occurred while importing CSVs: {e}"))
            self.thread.error.connect(lambda: logging.error(traceback.format_exc()))
            
            # Canceled signal.
            self.thread.canceled.connect(progress_dialog.close)
            # self.thread.canceled.connect(lambda: QMessageBox.information(self, "Canceled", "Data processing was canceled."))
            self.thread.canceled.connect(lambda: self.status_bar.showMessage("Data processing canceled.", 5000))  # Show message for 5 seconds
            self.thread.canceled.connect(lambda: logging.info("Data processing canceled."))
            self.thread.canceled.connect(self.refresh_existing_experiments)
            
            # Start the thread.
            self.thread.start()
            progress_dialog.canceled.connect(self.thread.cancel)
        else: # No directory selected.
            QMessageBox.warning(self, "Warning", "You must select a CSV directory.")
            logging.info("No CSV directory selected. Import canceled.")
    
    def reload_current_session(self):
        logging.debug("Reloading current session.")
        if self.current_dataset and self.current_session:
            self.current_dataset.reload_session(self.current_session.session_id)
            self.current_dataset.apply_preferences()
            self.current_experiment.save_experiment()
            self.plot_widget.update_plot_options() # Reset plot options

            self.status_bar.showMessage("Session reloaded successfully.", 5000)  # Show message for 5 seconds
            logging.debug("Session reloaded successfully.")
        else:
            QMessageBox.warning(self, "Warning", "Please select a session first.")

    def reload_current_dataset(self):
        logging.debug("Reloading current dataset.")
        if self.current_dataset:
            self.current_dataset.reload_dataset_sessions()
            self.data_selection_widget.update_session_combo()
            self.current_experiment.save_experiment()
            self.plot_widget.update_plot_options() # Reset plot options

            self.status_bar.showMessage("Dataset reloaded successfully.", 5000)  # Show message for 5 seconds
            logging.debug("Dataset reloaded successfully.")
        else:
            QMessageBox.warning(self, "Warning", "Please select a dataset first.")

    def reload_current_experiment(self):
        logging.debug(f"Reloading current experiment: {self.current_experiment.expt_id}.") 
        if self.current_experiment:
            # delete bin file in output folder
            bin_file = self.current_experiment.save_path
            if os.path.exists(bin_file):
                os.remove(bin_file)
                logging.debug(f"Deleted bin file: {bin_file}.")
            else:
                logging.error(f"Bin file not found: {bin_file}. Could not delete current experiment.")
            
            self.current_experiment = None
            self.current_dataset = None
            self.current_session = None
            
            self.refresh_existing_experiments()
            
            logging.debug("Experiment reloaded successfully.")
            self.status_bar.showMessage("Experiment reloaded successfully.", 5000)  # Show message for 5 seconds

    def refresh_existing_experiments(self):
        logging.debug("Refreshing existing experiments.")
        self.unpack_existing_experiments()
        self.data_selection_widget.update_experiment_combo()
        self.plot_widget.update_plot_options() # Reset plot options
        logging.debug("Existing experiments refreshed successfully.")

    def show_preferences_window(self):
        logging.debug("Showing preferences window.")
        window = PreferencesDialog(self.config_file, parent=self)
        if window.exec() == QDialog.DialogCode.Accepted:
            # Apply preferences to data
            if self.current_experiment:
                self.current_experiment.apply_preferences()
                self.current_experiment.save_experiment()
            self.status_bar.showMessage("Preferences applied successfully.", 5000)  # Show message for 5 seconds
            logging.debug("Preferences applied successfully.")
        else:
            QMessageBox.warning(self, "Warning", "No changes made to preferences.")
            logging.debug("No changes made to preferences.")

    def change_channel_names(self):
        # Check if a dataset and session are selected
        logging.debug("Changing channel names.")
        if not self.channel_names:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return

        # Open dialog to change channel names
        dialog = ChangeChannelNamesDialog(self.channel_names, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_names = dialog.get_new_names()
            if new_names:
                # Update channel names in the current session and dataset
                if self.current_dataset and self.current_experiment:
                    self.current_dataset.rename_channels(new_names)
                    self.current_experiment.save_experiment()

                # Update the channel_names list
                self.channel_names = self.current_dataset.channel_names
                
                self.status_bar.showMessage("Channel names updated successfully.", 5000)  # Show message for 5 seconds
                logging.debug("Channel names updated successfully.")
            else:
                QMessageBox.warning(self, "Warning", "No changes made to channel names.")
                logging.debug("No changes made to channel names.")

    def show_about_screen(self):
        dialog = AboutDialog(self)
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

    # Data selection widget functions - will be called whenever their index changes.
    def load_experiment(self, index):
        if index >= 0:
            experiment_name = self.expts_dict_keys[index]
            save_path = os.path.join(get_output_bin_path(),(f"{experiment_name}.pickle"))
            logging.debug(f"Loading experiment: '{experiment_name}'.")
            
            try:
                # Load existing experiment if available
                if os.path.exists(save_path):
                    self.current_experiment = EMGExperiment.load_experiment(save_path) # Type: EMGExperiment
                    logging.debug(f"Experiment '{experiment_name}' loaded successfully from bin.")
                else:
                    self.current_experiment = EMGExperiment(experiment_name, self.expts_dict, save_path=save_path) # Type: EMGExperiment
                    logging.debug(f"Experiment '{experiment_name}' created to bin and loaded successfully.")

                # Update current expt/dataset/session
                self.data_selection_widget.update_dataset_combo()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while loading experiment '{experiment_name}': {e}")
                logging.error(f"An error occurred while loading experiment: {e}")
                logging.error(traceback.format_exc())

    def load_dataset(self, index):
        if index >= 0:
            # date, animal_id, condition = EMGDataset.getDatasetInfo(dataset_name)  
            logging.debug(f"Loading dataset [{index}] from experiment '{self.current_experiment.expt_id}'.")

            if self.current_experiment:
                self.current_dataset = self.current_experiment.get_dataset(index) # Type: EMGDataset
            else:
                logging.error("No current experiment to load dataset from.")     

            # Update current dataset/session and channel names
            self.channel_names = self.current_dataset.channel_names
            self.data_selection_widget.update_session_combo()

    def load_session(self, index):
        if self.current_dataset and index >= 0:
            logging.debug(f"Loading session [{index}] from dataset '{self.current_dataset.dataset_id}'.")
            self.current_session = self.current_dataset.get_session(index) # Type: EMGSession
            if hasattr(self.plot_widget.current_option_widget, 'recording_cycler'):
                self.plot_widget.current_option_widget.recording_cycler.reset_max_recordings()

    # Reports functions.
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

    def show_experiment_report(self):
        logging.debug("Showing experiment parameters report.")
        if self.current_experiment:
            report = self.current_experiment.experiment_parameters()
            report = format_report(report)
            dialog = CopyableReportDialog("Experiment Report", report, self)
            dialog.exec()
        else:
            QMessageBox.warning(self, "Warning", "Please select an experiment first.")

    def show_mmax_report(self):
            logging.debug("Showing M-max report.")
            if self.current_session:
                report = self.current_session.m_max_report()
                report = format_report(report)
                dialog = CopyableReportDialog("M-max Report (method = RMS)", report, self)
                dialog.exec()
            else:
                QMessageBox.warning(self, "Warning", "Please select a session first.")

    # Plotting functions
    def plot_data(self, return_raw_data : bool = False):
        self.plot_widget.canvas.show()
        plot_type_raw = self.plot_widget.plot_type_combo.currentText()
        plot_type = self.plot_type_dict.get(plot_type_raw)
        plot_options = self.plot_widget.get_plot_options()
        raw_data = None # Type: pd.DataFrame
        
        # Plot the data      
        try:
            logging.debug(f'Plotting {plot_type} with options: {plot_options}.')
            if self.plot_widget.session_radio.isChecked():
                logging.debug("Plotting session data.")
                if self.current_session:
                    # Assuming plot_type corresponds to the data_type in plot_emg
                    raw_data = self.current_session.plot(
                        plot_type=plot_type,
                        **plot_options,
                        canvas = self.plot_widget.canvas
                    )
                else:
                    QMessageBox.warning(self, "Warning", "Please select a session first.")
                    return
            else:
                logging.debug("Plotting dataset data.")
                if self.current_dataset:
                    # If you have a similar plot method for dataset, use it here
                    raw_data = self.current_dataset.plot(
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
        logging.debug("Plotting complete.")

        self.plot_pane.layout.update()  # Refresh the layout of the plot pane

        if return_raw_data:
            logging.debug("Returning raw data.")
            return raw_data
        else:
            logging.debug("Not returning raw data.")
            return

    def get_raw_data(self):
        raw_data = self.plot_data(return_raw_data=True)
        if raw_data is not None:
            dialog = DataFrameDialog(raw_data, self)
            dialog.exec()
        else:
            QMessageBox.warning(self, "Warning", "No data to display.")

    @staticmethod
    def validate_dataset_name(dataset_path):
        # strip the dataset name from the provided path
        original_dataset_name = os.path.basename(dataset_path)
        # get the path before the dataset name for later reconstruction
        dataset_basepath = os.path.dirname(dataset_path)

        def get_new_dataset_name(dataset_name, validity_check_dict):
            if 'PyQt6' in sys.modules: # Check if PyQt6 is imported. If not, use input() instead of QDialog.
                from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout
                # check if there is an app running
                app = QApplication.instance()
                if app is None:
                    app = QApplication(sys.argv)

                dialog = QDialog()
                dialog.setWindowTitle('Rename Dataset')
                dialog.setModal(True)
                dialog.resize(500, 200)
                layout = QVBoxLayout()

                # Add labels based on the validity check dictionary.
                main_text = QLabel(f'The dataset name "{dataset_name}" is not valid. Dataset folder names should be formatted like "[YYMMDD] [Animal ID] [Experimental Condition]". Please confirm the following fields, and that they are separated by single spaces:')
                main_text.setWordWrap(True)
                layout.addWidget(main_text)
                if not validity_check_dict['Date']:
                    layout.addWidget(QLabel('- Date (YYYYMMDD)'))
                if not validity_check_dict['Animal ID']:
                    layout.addWidget(QLabel('- Animal ID (e.g., XX000.0)'))
                if not validity_check_dict['Condition']:
                    layout.addWidget(QLabel('- Experimental Condition (Any string. Spaces allowed.)'))
                        
                # Add a line edit and button to the layout
                layout.addWidget(QLabel('\nRename your dataset:'))
                line_edit = QLineEdit(dataset_name)
                layout.addWidget(line_edit)

                # Rename button
                button = QPushButton('Rename')
                button.clicked.connect(dialog.accept)
                layout.addWidget(button)

                dialog.setLayout(layout)
                result = dialog.exec()

                if result == QDialog.DialogCode.Accepted:
                    dataset_name = line_edit.text()
                    return dataset_name
                else:
                    raise ValueError('User canceled dataset renaming.')
            else:
                print(f'The dataset name "{dataset_name}" is not valid. Please confirm the following fields:')
                if not validity_check_dict['Date']:
                    print('\t- Date (YYYYMMDD)')
                if not validity_check_dict['Animal ID']:
                    print('\t- Animal ID (e.g., XX000.0)')
                if not validity_check_dict['Condition']:
                    print('\t- Experimental Condition (Any string. Spaces allowed.)')
                dataset_name = input('Rename your dataset: > ')
                if not dataset_name:
                    raise ValueError('User canceled dataset renaming.')
                return dataset_name

        def check_name(dataset_name, name_changed = None):
            date_valid, animal_id_valid, condition_valid = False, False, False
            if not name_changed:
                name_changed = False
            
            try: # check if the dataset name is in the correct format
                pattern = r'^(\d+)\s([a-zA-Z0-9.]+)\s(.+)$'
                match = re.match(pattern, dataset_name)
                if match:
                    date_string = match.group(1)
                    animal_id = match.group(2)
                    condition = match.group(3)
                else:
                    raise AttributeError
            except AttributeError:
                new_dataset_name = get_new_dataset_name(dataset_name, validity_check_dict = {'Date': date_valid, 'Animal ID': animal_id_valid, 'Condition': condition_valid})
                return check_name(new_dataset_name, name_changed=True)
            
            # 1) confirm date field is valid
            if len(date_string) == 6 or len(date_string) == 8:
                date_valid = True
            # 2) confirm animal id is valid
            if len(animal_id) > 0:
                animal_id_valid = True
            # 3) confirm condition is valid
            if len(condition) > 0:
                condition_valid = True

            if date_valid and animal_id_valid and condition_valid:
                return dataset_name, name_changed
            else:
                new_dataset_name = get_new_dataset_name(dataset_name, validity_check_dict = {'Date': date_valid, 'Animal ID': animal_id_valid, 'Condition': condition_valid})
                return check_name(new_dataset_name, name_changed=True)
            
            

        validated_dataset_name, name_changed = check_name(original_dataset_name)
        if name_changed: # if the name was changed, rename the dataset folder
            logging.info(f'Dataset name changed from "{original_dataset_name}" to "{validated_dataset_name}".')
            validated_dataset_path = os.path.join(dataset_basepath, validated_dataset_name)
            os.rename(dataset_path, validated_dataset_path)
            logging.info(f'Dataset folder renamed from "{dataset_path}" to "{validated_dataset_path}".')
        else: # if the name was not changed, return the original, now-validated dataset path.
            validated_dataset_path = dataset_path

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = EMGAnalysisGUI()
    gui.show()
    sys.exit(app.exec())