import sys
import os
import re
import shutil
import logging
import traceback
import multiprocessing

import markdown
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QFileDialog, QMessageBox, 
                             QDialog, QProgressDialog, QHBoxLayout, QStatusBar, QInputDialog)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt

from monstim_signals.domain.experiment import Experiment
from monstim_signals.domain.dataset import Dataset
from monstim_signals.domain.session import Session
from monstim_signals.io.repositories import ExperimentRepository
from monstim_signals.io.csv_importer import GUIExptImportingThread
from monstim_signals.core.utils import (format_report, get_output_path, get_data_path, get_output_bin_path, 
                           get_source_path, get_docs_path, get_config_path, BIN_EXTENSION)

from monstim_gui.splash import SPLASH_INFO
from monstim_gui.dialogs import (
    ChangeChannelNamesDialog,
    ReflexSettingsDialog,
    CopyableReportDialog,
    SelectChannelsDialog,
    LatexHelpWindow,
    AboutDialog,
    HelpWindow,
    PreferencesDialog,
    InvertChannelPolarityDialog,
    LatencyWindowsDialog,
)
from monstim_gui.menu_bar import MenuBar
from monstim_gui.data_selection_widget import DataSelectionWidget
from monstim_gui.reports_widget import ReportsWidget
from monstim_gui.plotting.plotting_widget import PlotWidget, PlotPane
from monstim_gui.commands import (
    ExcludeSessionCommand,
    ExcludeDatasetCommand,
    RestoreSessionCommand,
    RestoreDatasetCommand,
    CommandInvoker,
    ExcludeRecordingCommand,
    RestoreRecordingCommand,
    InvertChannelPolarityCommand,
)
from monstim_gui.dataframe_exporter import DataFrameDialog

class EMGAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.has_unsaved_changes = False
        self.setWindowTitle("MonStim Analyzer")
        self.setWindowIcon(QIcon(os.path.join(get_source_path(), 'icon.png')))
        self.setGeometry(30, 30, 800, 770)
    
        # Initialize variables
        self.expts_dict = {}
        self.expts_dict_keys = []  # type: list[str]
        self.current_experiment: Experiment | None = None
        self.current_dataset: Dataset | None = None
        self.current_session: Session | None = None
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

        self.plot_widget.initialize_plot_widget()

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
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
            self.command_invoker.undo()
        finally:
            QApplication.restoreOverrideCursor()

    def redo(self):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
            self.command_invoker.redo()
        finally:
            QApplication.restoreOverrideCursor()

    def exclude_recording(self, recording_index):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
            command = ExcludeRecordingCommand(self, recording_index)
            self.command_invoker.execute(command)
        finally:
            QApplication.restoreOverrideCursor()
    
    def restore_recording(self, recording_index):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
            command = RestoreRecordingCommand(self, recording_index)
            self.command_invoker.execute(command)
        finally:
            QApplication.restoreOverrideCursor()

    def exclude_session(self):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
            command = ExcludeSessionCommand(self)
            self.command_invoker.execute(command)
        finally:
            QApplication.restoreOverrideCursor()

    def exclude_dataset(self):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
            command = ExcludeDatasetCommand(self)
            self.command_invoker.execute(command)
        finally:
            QApplication.restoreOverrideCursor()

    def restore_session(self, session_id: str):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            command = RestoreSessionCommand(self, session_id)
            self.command_invoker.execute(command)
        finally:
            QApplication.restoreOverrideCursor()

    def restore_dataset(self, dataset_id: str):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            command = RestoreDatasetCommand(self, dataset_id)
            self.command_invoker.execute(command)
        finally:
            QApplication.restoreOverrideCursor()

    def prompt_restore_session(self):
        if not self.current_dataset:
            QMessageBox.warning(self, "Warning", "Please select a dataset first.")
            return
        excluded = list(self.current_dataset.excluded_sessions)
        if not excluded:
            QMessageBox.information(self, "Info", "No excluded sessions to restore.")
            return
        session_id, ok = QInputDialog.getItem(
            self,
            "Restore Session",
            "Select session to restore:",
            excluded,
            0,
            False,
        )
        if ok and session_id:
            self.restore_session(session_id)

    def prompt_restore_dataset(self):
        if not self.current_experiment:
            QMessageBox.warning(self, "Warning", "Please select an experiment first.")
            return
        excluded = list(self.current_experiment.excluded_datasets)
        if not excluded:
            QMessageBox.information(self, "Info", "No excluded datasets to restore.")
            return
        dataset_id, ok = QInputDialog.getItem(
            self,
            "Restore Dataset",
            "Select dataset to restore:",
            excluded,
            0,
            False,
        )
        if ok and dataset_id:
            self.restore_dataset(dataset_id)

    def unpack_existing_experiments(self):
        logging.debug("Unpacking existing experiments.")
        if os.path.exists(self.output_path):
            try:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
                self.expts_dict = {
                    name: os.path.join(self.output_path, name)
                    for name in os.listdir(self.output_path)
                    if os.path.isdir(os.path.join(self.output_path, name))
                }
                self.expts_dict_keys = sorted(self.expts_dict.keys())
                logging.debug("Existing experiments unpacked successfully.")
            except Exception as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(self, "Error", f"An error occurred while unpacking existing experiments: {e}")
                logging.error(f"An error occurred while unpacking existing experiments: {e}")
                logging.error(traceback.format_exc())
            finally:
                QApplication.restoreOverrideCursor()
    
    # Menu bar functions
    def update_reflex_time_windows(self, level : str):
        logging.debug("Updating reflex window settings.")
        match level: # Check the level of the reflex window settings update.
            case 'experiment':
                if not self.current_experiment:
                    QMessageBox.warning(self, "Warning", "Please select an experiment first.")
                    return
                else:
                    emg_data = self.current_experiment
            case 'dataset':
                if not self.current_dataset:
                    QMessageBox.warning(self, "Warning", "Please load a dataset first.")
                    return
                else:
                    emg_data = self.current_dataset
            case 'session':
                if not self.current_session:
                    QMessageBox.warning(self, "Warning", "Please select a session first.")
                    return
                else:
                    emg_data = self.current_session
            case _:
                QMessageBox.warning(self, "Warning", "Invalid level for updating reflex window settings.")
                return
        
        try:
            if self.current_session and self.current_dataset:
                dialog = ReflexSettingsDialog(emg_data, self)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    try:
                        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
                        self.has_unsaved_changes = True
                        self.status_bar.showMessage("Window settings updated successfully.", 5000)
                        logging.debug("Window settings updated successfully.")
                    finally:
                        QApplication.restoreOverrideCursor()
            else:
                QMessageBox.warning(self, "Warning", "Please select a session first.")
        finally:
            QApplication.restoreOverrideCursor()

    def manage_latency_windows(self, level: str):
        logging.debug("Managing latency windows.")
        match level:
            case 'experiment':
                if not self.current_experiment:
                    QMessageBox.warning(self, "Warning", "Please select an experiment first.")
                    return
                emg_data = self.current_experiment
            case 'dataset':
                if not self.current_dataset:
                    QMessageBox.warning(self, "Warning", "Please load a dataset first.")
                    return
                emg_data = self.current_dataset
            case 'session':
                if not self.current_session:
                    QMessageBox.warning(self, "Warning", "Please select a session first.")
                    return
                emg_data = self.current_session
            case _:
                QMessageBox.warning(self, "Warning", "Invalid level for managing latency windows.")
                return

        dialog = LatencyWindowsDialog(emg_data, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.has_unsaved_changes = True
            self.status_bar.showMessage("Latency windows updated successfully.", 5000)

    def invert_channel_polarity(self, level : str):
        logging.debug("Inverting channel polarity.")

        match level: # Check the level of the channel polarity inversion.
            case 'experiment':
                if not self.current_experiment:
                    QMessageBox.warning(self, "Warning", "Please select an experiment first.")
                    return
                else:
                    dialog = InvertChannelPolarityDialog(self.current_experiment, self)
            case 'dataset':
                if not self.current_dataset:
                    QMessageBox.warning(self, "Warning", "Please load a dataset first.")
                    return
                else:
                    dialog = InvertChannelPolarityDialog(self.current_dataset, self)
            case 'session':
                if not self.current_session:
                    QMessageBox.warning(self, "Warning", "Please select a session first.")
                    return
                else:
                    dialog = InvertChannelPolarityDialog(self.current_session, self)
            case _:
                QMessageBox.warning(self, "Warning", "Invalid level for inverting channel polarity.")
                return

        try:
            if dialog.exec():  # Show the dialog and wait for the user's response
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
                channel_indexes_to_invert = dialog.get_selected_channel_indexes()
                if not channel_indexes_to_invert:
                    QMessageBox.warning(self, "Warning", "Please select at least one channel to invert.")
                    return
                else:
                    command = InvertChannelPolarityCommand(self, level, channel_indexes_to_invert)
                    self.command_invoker.execute(command)
                    self.status_bar.showMessage("Channel polarity inverted successfully.", 5000)
            else:
                QMessageBox.warning(self, "Warning", "Please load a dataset first.")
        finally:
            QApplication.restoreOverrideCursor()

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
                    logging.info(f"Checking for bin file: {os.path.join(get_output_bin_path(),(f'{expt_name}{BIN_EXTENSION}'))}.")
                    bin_file = os.path.join(get_output_bin_path(),(f"{expt_name}{BIN_EXTENSION}"))
                    if os.path.exists(bin_file):
                        os.remove(bin_file)
                        logging.info(f"Deleted bin file: {bin_file}.")
                    else:
                        logging.warning(f"Bin file not found: {bin_file}. Could not delete existing experiment if it exists.")  
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
            progress_dialog = QProgressDialog("Processing...", "Cancel", 0, 100, self)
            progress_dialog.setWindowTitle("Importing Data")
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
            self.thread.finished.connect(lambda: self.data_selection_widget.experiment_combo.setCurrentIndex(0))
            self.thread.finished.connect(lambda: self.status_bar.showMessage("Data processed and imported successfully.", 5000))  # Show message for 5 seconds
            self.thread.finished.connect(lambda: logging.info("Data processed and imported successfully."))

            # Error signal.
            self.thread.error.connect(lambda e: QMessageBox.critical(self, "Error", f"An error occurred: {e}"))
            self.thread.error.connect(lambda e: logging.error(f"An error occurred while importing CSVs: {e}"))
            self.thread.error.connect(lambda: logging.error(traceback.format_exc()))

            # Canceled signal.
            self.thread.canceled.connect(progress_dialog.close)
            self.thread.canceled.connect(lambda: self.status_bar.showMessage("Data processing canceled.", 5000))  # Show message for 5 seconds
            self.thread.canceled.connect(lambda: logging.info("Data processing canceled."))
            self.thread.canceled.connect(self.refresh_existing_experiments)
            
            # Start the thread.
            self.thread.start()
            progress_dialog.canceled.connect(self.thread.cancel)
        else: # No directory selected.
            QMessageBox.warning(self, "Warning", "You must select a CSV directory.")
            logging.warning("No CSV directory selected. Import canceled.")
    
    def rename_experiment(self):
        logging.debug("Renaming experiment.")
        if self.current_experiment:
            new_name, ok = QInputDialog.getText(self, "Rename Experiment", "Enter new experiment name:", text=self.current_experiment.formatted_name)

            if ok and new_name:
                try:
                    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
                    # Rename bin file in output folder
                    bin_file = self.current_experiment.save_path
                    new_bin_file = os.path.join(get_output_bin_path(), f"{new_name}{BIN_EXTENSION}")
                    if os.path.exists(bin_file):
                        os.rename(bin_file, new_bin_file)
                        logging.debug(f"Renamed bin file: {bin_file} -> {new_bin_file}.")
                    else:
                        logging.warning(f"Bin file not found: {bin_file}. Could not rename current experiment.")

                    # Rename experiment folder in data folder
                    old_folder = os.path.join(self.output_path, self.current_experiment.formatted_name)
                    new_folder = os.path.join(self.output_path, new_name)
                    if os.path.exists(old_folder):
                        os.rename(old_folder, new_folder)
                        logging.debug(f"Renamed experiment folder: {old_folder} -> {new_folder}.")
                    else:
                        logging.warning(f"Experiment folder not found: {old_folder}. Could not rename current experiment.")
                except FileExistsError:
                    QMessageBox.critical(self, "Error", f"An experiment with the name '{new_name}' already exists. Please choose a different name.")
                    logging.warning(f"An experiment with the name '{new_name}' already exists. Could not rename current experiment.")
                    return
                finally:
                    QApplication.restoreOverrideCursor()
                
                # Update experiment name in the experiment object
                self.current_experiment.rename_experiment(new_name)
                self.has_unsaved_changes = True
                self.refresh_existing_experiments()
                self.status_bar.showMessage("Experiment renamed successfully.", 5000)

    def delete_experiment(self):
        # warning message box
        logging.debug("Deleting experiment.")
        if self.current_experiment:
            delete = QMessageBox.warning(self, "Delete Experiment", f"Are you sure you want to delete the experiment '{self.current_experiment.id}'?\n\nWARNING: This action cannot be undone.", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if delete == QMessageBox.StandardButton.Yes:
                # delete bin file in output folder
                shutil.rmtree(os.path.join(self.output_path, self.current_experiment.id))
                logging.debug(f"Deleted experiment folder: {os.path.join(self.output_path, self.current_experiment.id)}.")

                self.current_experiment = None
                self.current_dataset = None
                self.current_session = None
                self.refresh_existing_experiments()
                self.status_bar.showMessage("Experiment deleted successfully.", 5000)

    def reload_current_session(self):
        logging.debug("Reloading current session.")
        if self.current_dataset and self.current_session:
            if self.current_session.repo is not None:
                if self.current_session.repo.session_js.exists():
                    self.current_session.repo.session_js.unlink()
                new_sess = self.current_session.repo.load()
                idx = self.current_dataset._all_sessions.index(self.current_session)
                self.current_dataset._all_sessions[idx] = new_sess
                self.current_session = new_sess
            self.has_unsaved_changes = True
            self.plot_widget.on_data_selection_changed() # Alert plot widget to update plot options, etc.

            self.status_bar.showMessage("Session reloaded successfully.", 5000)  # Show message for 5 seconds
            logging.debug("Session reloaded successfully.")
        else:
            QMessageBox.warning(self, "Warning", "Please select a session first.")

    def reload_current_dataset(self):
        logging.debug("Reloading current dataset.")
        if self.current_dataset:
            if self.current_dataset.repo is not None:
                if self.current_dataset.repo.dataset_js.exists():
                    self.current_dataset.repo.dataset_js.unlink()
                for sess in self.current_dataset._all_sessions:
                    if sess.repo and sess.repo.session_js.exists():
                        sess.repo.session_js.unlink()
                new_ds = self.current_dataset.repo.load()
                idx = self.current_experiment._all_datasets.index(self.current_dataset)
                self.current_experiment._all_datasets[idx] = new_ds
                self.current_dataset = new_ds
            self.data_selection_widget.update_session_combo()
            self.has_unsaved_changes = True
            self.plot_widget.on_data_selection_changed() # Reset plot options

            self.status_bar.showMessage("Dataset reloaded successfully.", 5000)  # Show message for 5 seconds
            logging.debug("Dataset reloaded successfully.")
        else:
            QMessageBox.warning(self, "Warning", "Please select a dataset first.")

    def reload_current_experiment(self):
        """Reload the current experiment from disk."""
        logging.debug(f"Reloading current experiment: {self.current_experiment.id}.")
        current_experiment_combo_index = self.data_selection_widget.experiment_combo.currentIndex()
        if self.current_experiment:
            if self.current_experiment.repo is not None:
                if self.current_experiment.repo.expt_js.exists():
                    self.current_experiment.repo.expt_js.unlink()
                else:
                    logging.warning(f"Experiment JS file does not exist: {self.current_experiment.repo.expt_js}. Cannot unlink.")
                for ds in self.current_experiment._all_datasets:
                    if ds.repo and ds.repo.dataset_js.exists():
                        ds.repo.dataset_js.unlink()
                    for sess in ds._all_sessions:
                        if sess.repo and sess.repo.session_js.exists():
                            sess.repo.session_js.unlink()
                new_expt = self.current_experiment.repo.load()
                self.current_experiment = new_expt
            else:
                self.current_experiment = None
                logging.warning("No repository found for the current experiment. Cannot reload.")
            self.current_dataset = None
            self.current_session = None

            self.refresh_existing_experiments()
            self.data_selection_widget.experiment_combo.setCurrentIndex(current_experiment_combo_index)

            self.current_experiment.apply_preferences(reset_properties=False)
            self.has_unsaved_changes = True
            self.plot_widget.on_data_selection_changed()

            logging.debug("Experiment reloaded successfully.")
            self.status_bar.showMessage("Experiment reloaded successfully.", 5000)  # Show message for 5 seconds

    def refresh_existing_experiments(self):
        logging.debug("Refreshing existing experiments.")
        current_experiment_combo_index = self.data_selection_widget.experiment_combo.currentIndex()
        self.unpack_existing_experiments()
        self.data_selection_widget.update_experiment_combo()
        self.plot_widget.on_data_selection_changed() # Reset plot options
        self.data_selection_widget.experiment_combo.setCurrentIndex(current_experiment_combo_index)
        logging.debug("Existing experiments refreshed successfully.")

    def show_preferences_window(self):
        logging.debug("Showing preferences window.")
        window = PreferencesDialog(self.config_file, parent=self)
        if window.exec() == QDialog.DialogCode.Accepted:
            # Apply preferences to data
            if self.current_experiment:
                self.current_experiment.apply_preferences(reset_properties=False)
                self.has_unsaved_changes = True
            self.status_bar.showMessage("Preferences applied successfully.", 5000)  # Show message for 5 seconds
            logging.debug("Preferences applied successfully.")
        else:
            QMessageBox.warning(self, "Warning", "No changes made to preferences.")
            logging.debug("No changes made to preferences.")

    def select_channels(self, level : str):
        logging.debug("Selecting channels.")
        match level: # Check the level of the channel selection.
            case 'experiment':
                if not self.current_experiment:
                    QMessageBox.warning(self, "Warning", "Please select an experiment first.")
                    return
                else:
                    dialog = SelectChannelsDialog(self.current_experiment, self)
            case 'dataset':
                if not self.current_dataset:
                    QMessageBox.warning(self, "Warning", "Please load a dataset first.")
                    return
                else:
                    dialog = SelectChannelsDialog(self.current_dataset, self)
            case 'session':
                if not self.current_session:
                    QMessageBox.warning(self, "Warning", "Please select a session first.")
                    return
                else:
                    dialog = SelectChannelsDialog(self.current_session, self)
            case _:
                QMessageBox.warning(self, "Warning", "Invalid level for selecting channels.")
                return
        try:    
            if dialog.exec():
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
                selected_channel_indexes = dialog.get_selected_channel_indexes()
                if not selected_channel_indexes:
                    QMessageBox.warning(self, "Warning", "Please select at least one channel.")
                    return
                else:
                    match level:
                        case 'experiment':
                            self.current_experiment.select_channels(selected_channel_indexes)
                        case 'dataset':
                            self.current_dataset.select_channels(selected_channel_indexes)
                        case 'session':
                            self.current_session.select_channels(selected_channel_indexes)
                        case _:
                            QMessageBox.warning(self, "Warning", "Invalid level for selecting channels.")
                            return
                    self.has_unsaved_changes = True
        finally:
            QApplication.restoreOverrideCursor()
            self.status_bar.showMessage("Channels selected successfully.", 5000)

    def change_channel_names(self, level : str):
        logging.debug("Changing channel names.")

        match level: # Check the level of the channel name change and set the channel names accordingly.
            case 'experiment':
                if not self.current_experiment:
                    QMessageBox.warning(self, "Warning", "Please select an experiment first.")
                    return
                else:
                    self.channel_names = self.current_experiment.channel_names
            case 'dataset':
                if not self.current_dataset:
                    QMessageBox.warning(self, "Warning", "Please load a dataset first.")
                    return
                else:
                    self.channel_names = self.current_dataset.channel_names
            case 'session':
                if not self.current_session:
                    QMessageBox.warning(self, "Warning", "Please select a session first.")
                    return
                else:
                    self.channel_names = self.current_session.channel_names
            case _:
                QMessageBox.warning(self, "Warning", "Invalid level for changing channel names.")
                return

        # Open dialog to change channel names
        dialog = ChangeChannelNamesDialog(self.channel_names, self)
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
                new_names = dialog.get_new_names()
                if new_names:
                    match level:
                        case 'experiment':
                            self.current_experiment.rename_channels(new_names)
                        case 'dataset':
                            self.current_dataset.rename_channels(new_names)
                        case 'session':
                            self.current_session.rename_channels(new_names)
                        case _:
                            QMessageBox.warning(self, "Warning", "Invalid level for changing channel names.")
                            return
                        
                    self.has_unsaved_changes = True
                    self.status_bar.showMessage("Channel names updated successfully.", 5000)  # Show message for 5 seconds
                    logging.debug("Channel names updated successfully.")
                else:
                    QMessageBox.warning(self, "Warning", "No changes made to channel names.")
                    logging.debug("No changes made to channel names.")
        finally:
            QApplication.restoreOverrideCursor()

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
    def save_experiment(self):
        if self.current_experiment:
            self.current_experiment.save()
            self.status_bar.showMessage("Experiment saved successfully.", 5000)
            logging.debug("Experiment saved successfully.")
            self.has_unsaved_changes = False

    def load_experiment(self, index):
        if index >= 0:
            experiment_name = self.expts_dict_keys[index]
            exp_path = os.path.join(self.output_path, experiment_name)
            logging.debug(f"Loading experiment: '{experiment_name}'.")
            try:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
                if os.path.exists(exp_path):
                    repo = ExperimentRepository(Path(exp_path))
                    self.current_experiment = repo.load()
                    logging.debug(f"Experiment '{experiment_name}' loaded successfully.")
                else:
                    QMessageBox.warning(self, "Warning", f"Experiment folder '{exp_path}' not found.")
                    return

                # Update current expt/dataset/session
                self.data_selection_widget.update_dataset_combo()
                self.plot_widget.on_data_selection_changed() # Reset plot options
                self.status_bar.showMessage(f"Experiment '{experiment_name}' loaded successfully.", 5000)  # Show message for 5 seconds
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while loading experiment '{experiment_name}': {e}")
                logging.error(f"An error occurred while loading experiment: {e}")
                logging.error(traceback.format_exc())
            finally:
                QApplication.restoreOverrideCursor()

    def load_dataset(self, index):
        if index >= 0:
            logging.debug(f"Loading dataset [{index}] from experiment '{self.current_experiment.id}'.")

            if self.current_experiment:
                self.current_dataset = self.current_experiment.datasets[index]
            else:
                logging.error("No current experiment to load dataset from.")

            # Update current dataset/session and channel names
            self.channel_names = self.current_dataset.channel_names
            self.data_selection_widget.update_session_combo()
            self.plot_widget.on_data_selection_changed() # Reset plot options

    def load_session(self, index):
        if self.current_dataset and index >= 0:
            logging.debug(f"Loading session [{index}] from dataset '{self.current_dataset.id}'.")
            self.current_session = self.current_dataset.sessions[index]
            if hasattr(self.plot_widget.current_option_widget, 'recording_cycler'):
                self.plot_widget.current_option_widget.recording_cycler.reset_max_recordings()
            self.plot_widget.on_data_selection_changed() # Reset plot options

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

        if self.plot_widget.session_radio.isChecked():
            level = 'session'
            level_object = self.current_session
        elif self.plot_widget.dataset_radio.isChecked():
            level = 'dataset'
            level_object = self.current_dataset
        elif self.plot_widget.experiment_radio.isChecked():
            level = 'experiment'
            level_object = self.current_experiment
        else:
            QMessageBox.warning(self, "Warning", "Please select a level to plot data from (session, dataset, or experiment).")
            logging.warning("No level selected for plotting data.")
            return
        
        if level_object is None:
            QMessageBox.warning(self, "Warning", f"No {level} data exists to plot. Please try importing experiment data first.")
            logging.warning(f"No {level} data exists to plot. Please try importing experiment data first.")
            return

        # Plot the data      
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # Set cursor to busy
            raw_data = level_object.plot(
                plot_type=plot_type,
                **plot_options,
                canvas = self.plot_widget.canvas)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
            logging.error(f"An error occurred while plotting: {e}")
            logging.error(f"Plot type: {plot_type}, options: {plot_options}")
            logging.error(f"Current session: {self.current_session}, current dataset: {self.current_dataset}")
            logging.error(traceback.format_exc())
        finally:
            QApplication.restoreOverrideCursor()

        logging.info(f"Plot Created. level: {level} type: {plot_type}, options: {plot_options}, return_raw_data: {return_raw_data}.")
        self.plot_pane.layout.update()  # Refresh the layout of the plot pane

        if return_raw_data:
            return raw_data
        else:
            return

    def _get_raw_data(self):
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

    def show_save_confirmation_dialog(self):
        """Show dialog asking user if they want to save before closing"""
        if not self.current_experiment or not self.has_unsaved_changes:
            return True
            
        reply = QMessageBox.question(
        self,
        'Save Changes?',
        'Do you want to save the current experiment before closing?',
        QMessageBox.StandardButton.Save | 
        QMessageBox.StandardButton.Discard | 
        QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Save
    )
        
        if reply == QMessageBox.StandardButton.Save:
            saved = self.save_experiment()
            if saved:
                QApplication.quit() # Close the application after saving.
        elif reply == QMessageBox.StandardButton.Cancel:
            return False
        return True

    def closeEvent(self, event):
        """Handle application closing"""
        if self.show_save_confirmation_dialog():
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = EMGAnalysisGUI()
    gui.show()
    sys.exit(app.exec())