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
        self.parent.has_unsaved_changes = True
        self.history.append(command)
        self.redo_stack.clear()
        self.parent.menu_bar.update_undo_redo_labels()

    def undo(self):
        if self.history:
            command = self.history.pop()
            command.undo()
            self.parent.has_unsaved_changes = True
            self.redo_stack.append(command)
    
    def get_undo_command_name(self):
        if self.history:
            return self.history[-1].command_name
        return None
    
    def redo(self):
        if self.redo_stack:
            command = self.redo_stack.pop()
            command.execute()
            self.parent.has_unsaved_changes = True
            self.history.append(command)

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
    def __init__(self, gui, recording_index):
        self.command_name = "Exclude Recording"
        self.gui : 'MonstimGUI' = gui
        self.recording_index = recording_index
    
    def execute(self):
        try:
            self.gui.current_session.exclude_recording(self.recording_index)
            self.gui.data_selection_widget.update_all_data_combos()
        except ValueError as e:
            QMessageBox.critical(self.gui, "Error", str(e))
    
    def undo(self):
        try:
            self.gui.current_session.restore_recording(self.recording_index)
            self.gui.data_selection_widget.update_all_data_combos()
        except ValueError as e:
            QMessageBox.critical(self.gui, "Error", str(e))

class RestoreRecordingCommand(Command):
    def __init__(self, gui, original_recording_index):
        self.command_name = "Restore Recording"
        self.gui : 'MonstimGUI' = gui
        self.recording_index = original_recording_index

    def execute(self):
        try:
            self.gui.current_session.restore_recording(self.recording_index)
            self.gui.data_selection_widget.update_all_data_combos()
        except ValueError as e:
            QMessageBox.critical(self.gui, "Error", str(e))

    def undo(self):
        try:
            self.gui.current_session.exclude_recording(self.recording_index)
            self.gui.data_selection_widget.update_all_data_combos()
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

    def execute(self):
        self.removed_session = self.gui.current_session
        self.session_id = self.gui.current_session.id
        self.idx = self.gui.current_dataset.sessions.index(self.gui.current_session)
        self.gui.current_dataset.exclude_session(self.session_id)
        self.gui.current_session = None
        self.gui.data_selection_widget.update_all_data_combos()

    def undo(self):
        self.gui.current_dataset.restore_session(self.session_id)
        self.gui.current_session = self.removed_session
        self.gui.data_selection_widget.update_all_data_combos()

class ExcludeDatasetCommand(Command):
    """Exclude the currently selected dataset."""

    def __init__(self, gui):
        self.command_name = "Exclude Dataset"
        self.gui : 'MonstimGUI' = gui
        self.removed_dataset = None
        self.dataset_id = None
        self.idx = None

    def execute(self):
        self.removed_dataset = self.gui.current_dataset
        self.dataset_id = self.gui.current_dataset.id
        self.idx = self.gui.current_experiment.datasets.index(self.gui.current_dataset)
        self.gui.current_experiment.exclude_dataset(self.dataset_id)
        self.gui.current_dataset = None
        self.gui.data_selection_widget.update_all_data_combos()

    def undo(self):
        self.gui.current_experiment.restore_dataset(self.dataset_id)
        self.gui.current_dataset = self.removed_dataset
        self.gui.data_selection_widget.update_all_data_combos()

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
        self.gui.data_selection_widget.update_all_data_combos()

    def undo(self):
        self.gui.current_dataset.exclude_session(self.session_id)
        self.gui.current_session = None
        self.gui.data_selection_widget.update_all_data_combos()

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
        self.gui.data_selection_widget.update_all_data_combos()

    def undo(self):
        self.gui.current_experiment.exclude_dataset(self.dataset_id)
        self.gui.current_dataset = None
        self.gui.data_selection_widget.update_all_data_combos()
        
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
        self.command_name = "Set Latency Windows"
        self.gui = gui
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

