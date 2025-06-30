import sys
import os
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from monstim_gui.widgets.gui_layout import MenuBar, DataSelectionWidget, ReportsWidget, PlotPane, PlotWidget
    from PyQt6.QtWidgets import QStatusBar

import markdown
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QDialog,
    QInputDialog,
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt

from monstim_signals.domain.experiment import Experiment
from monstim_signals.domain.dataset import Dataset
from monstim_signals.domain.session import Session
from monstim_signals.core.utils import (get_output_path, get_source_path, get_docs_path, get_config_path)

from monstim_gui.core.splash import SPLASH_INFO
from monstim_gui.dialogs import (
    ChangeChannelNamesDialog,
    LatexHelpWindow,
    AboutDialog,
    HelpWindow,
    InvertChannelPolarityDialog,
    LatencyWindowsDialog,
)
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
from monstim_gui.widgets.gui_layout import setup_main_layout
from monstim_gui.managers.data_manager import DataManager
from monstim_gui.managers.report_manager import ReportManager
from monstim_gui.managers.plot_controller import PlotController
from monstim_gui.io.config_repository import ConfigRepository
from monstim_gui.io.help_repository import HelpFileRepository

class MonstimGUI(QMainWindow):
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
        self.PLOT_TYPE_DICT = {"EMG": "emg", "Suspected H-reflexes": "suspectedH", "Reflex Curves": "reflexCurves",
                               "M-max": "mmax", "Max H-reflex": "maxH", "Average Reflex Curves": "reflexCurves",
                               "Single EMG Recordings": "singleEMG"}
        
        # Set default paths
        self.output_path = get_output_path()
        self.config_file = get_config_path()


        # Helper managers
        self.data_manager = DataManager(self)
        self.report_manager = ReportManager(self)
        self.plot_controller = PlotController(self)
        self.config_repo = ConfigRepository(get_config_path())
        self.help_repo = HelpFileRepository(get_docs_path())

        self.init_ui()

        # Load existing pickled experiments if available
        self.data_manager.unpack_existing_experiments()
        self.data_selection_widget.update_experiment_combo()

        self.plot_widget.initialize_plot_widget()

        self.command_invoker = CommandInvoker(self)

    def init_ui(self):
        widgets = setup_main_layout(self)
        self.menu_bar : 'MenuBar' = widgets["menu_bar"]
        self.data_selection_widget : 'DataSelectionWidget' = widgets["data_selection_widget"]
        self.reports_widget : 'ReportsWidget' = widgets["reports_widget"]
        self.plot_pane : 'PlotPane' = widgets["plot_pane"]
        self.plot_widget : 'PlotWidget' = widgets["plot_widget"]
        self.status_bar : 'QStatusBar' = widgets["status_bar"]

        self.plot_widget.import_canvas()

        self.status_bar.showMessage(
            f"Welcome to MonStim Analyzer, {SPLASH_INFO['version']}", 10000
        )
   
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
    
    # Menu bar functions
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
        """Show help dialog using HelpFileRepository."""
        file = 'readme.md' if topic is None else topic
        markdown_content = self.help_repo.read_help_file(file)
        html_content = markdown.markdown(markdown_content)
        if latex:
            self.help_window = LatexHelpWindow(html_content, topic)
        else:
            self.help_window = HelpWindow(html_content, topic)
        self.help_window.show()

    def update_domain_configs(self, config=None):
        """Propagate the current config to all loaded domain objects."""
        if config is None:
            config = self.config_repo.read_config()
        if self.current_experiment:
            self.current_experiment.set_config(config)
        if self.current_dataset:
            self.current_dataset.set_config(config)
        if self.current_session:
            self.current_session.set_config(config)

    def set_current_experiment(self, experiment : 'Experiment'):
        """Set the current experiment and ensure config is injected."""
        config = self.config_repo.read_config()
        experiment.set_config(config)
        self.current_experiment = experiment

    def set_current_dataset(self, dataset: 'Dataset'):
        """Set the current dataset and ensure config is injected."""
        config = self.config_repo.read_config()
        dataset.set_config(config)
        self.current_dataset = dataset

    def set_current_session(self, session: 'Session'):
        """Set the current session and ensure config is injected."""
        config = self.config_repo.read_config()
        session.set_config(config)
        self.current_session = session

    # Close event handling
    def show_save_confirmation_dialog(self):
        """Show dialog asking user if they want to save before closing"""
        # Shouldn't be needed anymore for most changes, but keeping it in case it's needed for future changes.
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
            saved = self.data_manager.save_experiment()
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
    gui = MonstimGUI()
    gui.show()
    sys.exit(app.exec())