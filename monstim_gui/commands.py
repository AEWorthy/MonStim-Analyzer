import abc
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gui_main import EMGAnalysisGUI

class Command(abc.ABC):
    @abc.abstractmethod
    def execute(self):
        pass

    @abc.abstractmethod
    def undo(self):
        pass

class CommandInvoker:
    def __init__(self):
        self.history = deque()
        self.redo_stack = deque()

    def execute(self, command):
        command.execute()
        self.history.append(command)
        self.redo_stack.clear()

    def undo(self):
        if self.history:
            command = self.history.pop()
            command.undo()
            self.redo_stack.append(command)

    def redo(self):
        if self.redo_stack:
            command = self.redo_stack.pop()
            command.execute()
            self.history.append(command)

# GUI command classes
class RemoveSessionCommand(Command):
    def __init__(self, gui):
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

