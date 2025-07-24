import abc
from collections import deque
import copy
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMessageBox

if TYPE_CHECKING:
    from gui_main import MonstimGUI

class Command(abc.ABC):
    command_name : str = None

    @abc.abstractmethod
    def execute(self):
        pass

    @abc.abstractmethod
    def undo(self):
        pass

class CommandInvoker:
    def __init__(self, parent : 'MonstimGUI'):
        self.parent = parent # type: MonstimGUI
        self.history = deque() # type: deque[Command]
        self.redo_stack = deque() # type: deque[Command]

    def execute(self, command : Command):
        command.execute()
        self.history.append(command)
        self.redo_stack.clear()
        self.parent.menu_bar.update_undo_redo_labels()
        # Set self.parent._has_unsaved_changes to True if needed

    def undo(self):
        if self.history:
            command = self.history.pop()
            command.undo()
            self.redo_stack.append(command)
            self.parent.menu_bar.update_undo_redo_labels()
            # Set self.parent._has_unsaved_changes to True if needed

    def redo(self):
        if self.redo_stack:
            command = self.redo_stack.pop()
            command.execute()
            self.history.append(command)
            self.parent.menu_bar.update_undo_redo_labels()
            # Set self.parent._has_unsaved_changes to True if needed

    def get_undo_command_name(self):
        if self.history:
            return self.history[-1].command_name
        return None

    def get_redo_command_name(self):
        if self.redo_stack:
            return self.redo_stack[-1].command_name
        return None
    
    def remove_command_by_name(self, command_name : str):
        # Remove all occurrences from history
        self.history = deque(command for command in self.history if command.command_name != command_name)
        
        # Remove all occurrences from redo_stack
        self.redo_stack = deque(command for command in self.redo_stack if command.command_name != command_name)
                
# GUI command classes
class ExcludeRecordingCommand(Command):
    def __init__(self, gui, recording_id : str):
        self.command_name : str = "Exclude Recording"
        self.gui : 'MonstimGUI' = gui
        self.recording_id : str = recording_id

    def execute(self):
        try:
            self.gui.current_session.exclude_recording(self.recording_id)
            # Recording exclusion doesn't affect dataset/session selections, so use sync instead
            self.gui.data_selection_widget.sync_combo_selections()
        except ValueError as e:
            QMessageBox.critical(self.gui, "Error", str(e))
    
    def undo(self):
        try:
            self.gui.current_session.restore_recording(self.recording_id)
            self.gui.data_selection_widget.sync_combo_selections()
        except ValueError as e:
            QMessageBox.critical(self.gui, "Error", str(e))

class RestoreRecordingCommand(Command):
    def __init__(self, gui, recording_id : str):
        self.command_name : str = "Restore Recording"
        self.gui : 'MonstimGUI' = gui
        self.recording_id = recording_id

    def execute(self):
        try:
            self.gui.current_session.restore_recording(self.recording_id)
            self.gui.data_selection_widget.sync_combo_selections()
        except ValueError as e:
            QMessageBox.critical(self.gui, "Error", str(e))

    def undo(self):
        try:
            self.gui.current_session.exclude_recording(self.recording_id)
            self.gui.data_selection_widget.sync_combo_selections()
        except ValueError as e:
            QMessageBox.critical(self.gui, "Error", str(e))

class ExcludeSessionCommand(Command):
    """Exclude the currently selected session."""

    def __init__(self, gui):
        self.command_name = "Exclude Session"
        self.gui : 'MonstimGUI' = gui
        self.removed_session = None
        self.session_id = None
        self.idx = None
        self.previous_dataset = None

    def execute(self):
        self.removed_session = self.gui.current_session
        self.session_id = self.gui.current_session.id
        self.idx = self.gui.current_dataset.sessions.index(self.gui.current_session)
        self.previous_dataset = self.gui.current_dataset  # Preserve dataset selection
        
        self.gui.current_dataset.exclude_session(self.session_id)
        self.gui.current_session = None
        
        # Update combos and keep dataset selection
        self.gui.data_selection_widget.update_session_combo()

    def undo(self):
        self.gui.current_dataset.restore_session(self.session_id)
        self.gui.current_session = self.removed_session
        # Ensure we maintain the correct dataset selection
        if self.previous_dataset and self.gui.current_dataset != self.previous_dataset:
            self.gui.current_dataset = self.previous_dataset
        # Update session combo and set the correct selection
        self.gui.data_selection_widget.update_session_combo()
        if self.removed_session:
            try:
                session_index = self.gui.current_dataset.sessions.index(self.removed_session)
                self.gui.data_selection_widget.session_combo.blockSignals(True)
                self.gui.data_selection_widget.session_combo.setCurrentIndex(session_index)
                self.gui.data_selection_widget.session_combo.blockSignals(False)
            except ValueError:
                pass  # Session not found in list

class ExcludeDatasetCommand(Command):
    """Exclude the currently selected dataset."""

    def __init__(self, gui):
        self.command_name = "Exclude Dataset"
        self.gui : 'MonstimGUI' = gui
        self.removed_dataset = None
        self.dataset_id = None
        self.idx = None
        self.previous_experiment = None

    def execute(self):
        self.removed_dataset = self.gui.current_dataset
        self.dataset_id = self.gui.current_dataset.id
        self.idx = self.gui.current_experiment.datasets.index(self.gui.current_dataset)
        self.previous_experiment = self.gui.current_experiment  # Preserve experiment selection
        
        self.gui.current_experiment.exclude_dataset(self.dataset_id)
        self.gui.current_dataset = None
        self.gui.current_session = None
        
        # Update combos and keep experiment selection
        self.gui.data_selection_widget.update_dataset_combo()
        self.gui.data_selection_widget.update_session_combo()

    def undo(self):
        self.gui.current_experiment.restore_dataset(self.dataset_id)
        self.gui.current_dataset = self.removed_dataset
        # Ensure we maintain the correct experiment selection
        if self.previous_experiment and self.gui.current_experiment != self.previous_experiment:
            self.gui.current_experiment = self.previous_experiment
        # Update dataset combo and set the correct selection
        self.gui.data_selection_widget.update_dataset_combo()
        if self.removed_dataset:
            try:
                dataset_index = self.gui.current_experiment.datasets.index(self.removed_dataset)
                self.gui.data_selection_widget.dataset_combo.blockSignals(True)
                self.gui.data_selection_widget.dataset_combo.setCurrentIndex(dataset_index)
                self.gui.data_selection_widget.dataset_combo.blockSignals(False)
            except ValueError:
                pass  # Dataset not found in list
        # Update session combo since dataset changed
        self.gui.data_selection_widget.update_session_combo()

class RestoreSessionCommand(Command):
    """Restore an excluded session by ID."""

    def __init__(self, gui, session_id: str):
        self.command_name = "Restore Session"
        self.gui: 'MonstimGUI' = gui
        self.session_id = session_id
        self.session_obj = None

    def execute(self):
        self.session_obj = next(
            (s for s in self.gui.current_dataset._all_sessions if s.id == self.session_id),
            None
        )
        self.gui.current_dataset.restore_session(self.session_id)
        self.gui.current_session = self.session_obj
        # Only update session combo and sync its selection
        self.gui.data_selection_widget.update_session_combo()
        if self.session_obj:
            # Find the index of the restored session
            try:
                session_index = self.gui.current_dataset.sessions.index(self.session_obj)
                self.gui.data_selection_widget.session_combo.blockSignals(True)
                self.gui.data_selection_widget.session_combo.setCurrentIndex(session_index)
                self.gui.data_selection_widget.session_combo.blockSignals(False)
            except ValueError:
                pass  # Session not found in list

    def undo(self):
        self.gui.current_dataset.exclude_session(self.session_id)
        self.gui.current_session = None
        self.gui.data_selection_widget.update_session_combo()

class RestoreDatasetCommand(Command):
    """Restore an excluded dataset by ID."""

    def __init__(self, gui, dataset_id: str):
        self.command_name = "Restore Dataset"
        self.gui: 'MonstimGUI' = gui
        self.dataset_id = dataset_id
        self.dataset_obj = None

    def execute(self):
        self.dataset_obj = next(
            (ds for ds in self.gui.current_experiment._all_datasets if ds.id == self.dataset_id),
            None
        )
        self.gui.current_experiment.restore_dataset(self.dataset_id)
        self.gui.current_dataset = self.dataset_obj
        # Update dataset combo and sync its selection
        self.gui.data_selection_widget.update_dataset_combo()
        if self.dataset_obj:
            # Find the index of the restored dataset
            try:
                dataset_index = self.gui.current_experiment.datasets.index(self.dataset_obj)
                self.gui.data_selection_widget.dataset_combo.blockSignals(True)
                self.gui.data_selection_widget.dataset_combo.setCurrentIndex(dataset_index)
                self.gui.data_selection_widget.dataset_combo.blockSignals(False)
            except ValueError:
                pass  # Dataset not found in list
        # Update session combo since dataset changed
        self.gui.data_selection_widget.update_session_combo()

    def undo(self):
        self.gui.current_experiment.exclude_dataset(self.dataset_id)
        self.gui.current_dataset = None
        self.gui.current_session = None
        self.gui.data_selection_widget.update_dataset_combo()
        self.gui.data_selection_widget.update_session_combo()
        
class InvertChannelPolarityCommand(Command):
    def __init__(self, gui, level : str, channel_indexes_to_invert : list[int]):
        self.command_name = "Invert Channel Polarity"
        self.gui : 'MonstimGUI' = gui # type: EMGAnalysisGUI
        self.channel_indexes_to_invert = channel_indexes_to_invert
        
        match level:
            case 'experiment':
                self.level = self.gui.current_experiment
            case 'dataset':
                self.level = self.gui.current_dataset
            case 'session':
                self.level = self.gui.current_session
            case _:
                raise ValueError(f"Invalid level: {level}")

    def execute(self):
        for channel_index in self.channel_indexes_to_invert:
            self.level.invert_channel_polarity(channel_index)

    def undo(self):
        for channel_index in self.channel_indexes_to_invert:
            self.level.invert_channel_polarity(channel_index)

class SetLatencyWindowsCommand(Command):
    def __init__(self, gui, level: str, new_windows: list):
        self.command_name : str = "Set Latency Windows"
        self.gui : 'MonstimGUI' = gui
        match level:
            case 'experiment':
                self.level = self.gui.current_experiment
                self.sessions = [s for ds in self.level.datasets for s in ds.sessions]
            case 'dataset':
                self.level = self.gui.current_dataset
                self.sessions = list(self.level.sessions)
            case 'session':
                self.level = self.gui.current_session
                self.sessions = [self.level]
            case _:
                raise ValueError(f"Invalid level: {level}")
        self.new_windows = [copy.deepcopy(w) for w in new_windows]
        self.old_windows = {s.id: copy.deepcopy(s.annot.latency_windows) for s in self.sessions}

    def _apply(self, windows):
        import copy
        for s in self.sessions:
            s.annot.latency_windows = [copy.deepcopy(w) for w in windows]
            s.update_latency_window_parameters()
            if s.repo is not None:
                s.repo.save(s)
        if hasattr(self.level, 'update_latency_window_parameters'):
            if isinstance(self.level, list):
                for obj in self.level:
                    obj.update_latency_window_parameters()
            else:
                self.level.update_latency_window_parameters()

    def execute(self):
        self._apply(self.new_windows)

    def undo(self):
        for s in self.sessions:
            windows = self.old_windows[s.id]
            s.annot.latency_windows = windows
            s.update_latency_window_parameters()
            if s.repo is not None:
                s.repo.save(s)
        if hasattr(self.level, 'update_latency_window_parameters'):
            if isinstance(self.level, list):
                for obj in self.level:
                    obj.update_latency_window_parameters()
            else:
                self.level.update_latency_window_parameters()

class ChangeChannelNamesCommand(Command):
    def __init__(self, gui, level: str, new_names: dict):
        self.command_name : str = "Change Channel Names"
        self.gui : 'MonstimGUI' = gui
        self.new_names = copy.deepcopy(new_names)
        
        match level:
            case 'experiment':
                self.level = self.gui.current_experiment
            case 'dataset':
                self.level = self.gui.current_dataset
            case 'session':
                self.level = self.gui.current_session
            case _:
                raise ValueError(f"Invalid level: {level}")
        
        # Store old channel names for undo - create reverse mapping
        self.old_names = {new_name: old_name for old_name, new_name in new_names.items()}

    def execute(self):
        self.level.rename_channels(self.new_names)

    def undo(self):
        self.level.rename_channels(self.old_names)

