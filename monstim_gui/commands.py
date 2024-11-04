import abc
from collections import deque
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMessageBox

if TYPE_CHECKING:
    from gui_main import EMGAnalysisGUI

class Command(abc.ABC):
    command_name : str = None

    @abc.abstractmethod
    def execute(self):
        pass

    @abc.abstractmethod
    def undo(self):
        pass

class CommandInvoker:
    def __init__(self, parent : 'EMGAnalysisGUI'):
        self.parent = parent # type: EMGAnalysisGUI
        self.history = deque() # type: deque[Command]
        self.redo_stack = deque() # type: deque[Command]

    def execute(self, command : Command):
        command.execute()
        self.parent.current_experiment.save_experiment()
        self.history.append(command)
        self.redo_stack.clear()
        self.parent.menu_bar.update_undo_redo_labels()

    def undo(self):
        if self.history:
            command = self.history.pop()
            command.undo()
            self.parent.current_experiment.save_experiment()
            self.redo_stack.append(command)
    
    def get_undo_command_name(self):
        if self.history:
            return self.history[-1].command_name
        return None
    
    def redo(self):
        if self.redo_stack:
            command = self.redo_stack.pop()
            command.execute()
            self.parent.current_experiment.save_experiment()
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
        self.gui : 'EMGAnalysisGUI' = gui
        self.recording_index = recording_index
    
    def execute(self):
        try:
            self.gui.current_session.exclude_recording(self.recording_index)
        except ValueError as e:
            QMessageBox.critical(self.gui, "Error", str(e))
    
    def undo(self):
        try:
            self.gui.current_session.restore_recording(self.recording_index)
        except ValueError as e:
            QMessageBox.critical(self.gui, "Error", str(e))

class RestoreRecordingCommand(Command):
    def __init__(self, gui, original_recording_index):
        self.command_name = "Restore Recording"
        self.gui : 'EMGAnalysisGUI' = gui
        self.recording_index = original_recording_index

    def execute(self):
        try:
            self.gui.current_session.restore_recording(self.recording_index)
        except ValueError as e:
            QMessageBox.critical(self.gui, "Error", str(e))

    def undo(self):
        try:
            self.gui.current_session.exclude_recording(self.recording_index)
        except ValueError as e:
            QMessageBox.critical(self.gui, "Error", str(e))

class RemoveSessionCommand(Command):
    def __init__(self, gui):
        self.command_name = "Remove Session"
        self.gui : 'EMGAnalysisGUI' = gui
        self.removed_session = None

    def execute(self):
        self.removed_session = self.gui.current_session
        self.gui.current_dataset.remove_session(self.gui.current_session.session_id)
        self.gui.current_session = None
        self.gui.data_selection_widget.update_session_combo()
        
    def undo(self):
        self.gui.current_dataset.add_session(self.removed_session)
        self.gui.current_session = self.removed_session
        self.gui.data_selection_widget.update_session_combo()

class RemoveDatasetCommand(Command):
    def __init__(self, gui):
        self.command_name = "Remove Dataset"
        self.gui : 'EMGAnalysisGUI' = gui
        self.removed_dataset = None
    
    def execute(self):
        self.removed_dataset = self.gui.current_dataset
        self.gui.current_experiment.remove_dataset(self.gui.current_dataset.dataset_id)
        self.gui.current_dataset = None
        self.gui.data_selection_widget.update_dataset_combo()

    def undo(self):
        self.gui.current_experiment.add_dataset(self.removed_dataset)
        self.gui.current_dataset = self.removed_dataset
        self.gui.data_selection_widget.update_dataset_combo()
        
class InvertChannelPolarityCommand(Command):
    def __init__(self, gui, level : str, channel_indexes_to_invert : list[int]):
        self.command_name = "Invert Channel Polarity"
        self.gui : 'EMGAnalysisGUI' = gui # type: EMGAnalysisGUI
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