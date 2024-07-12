from PyQt6.QtWidgets import QMenuBar
from PyQt6.QtGui import QKeySequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gui_main import EMGAnalysisGUI

class MenuBar(QMenuBar):
        def __init__(self, parent : 'EMGAnalysisGUI'):
            super().__init__(parent)
            self.parent = parent
            self.create_file_menu()
            self.create_edit_menu()
        
        def create_file_menu(self):
            # File menu
            file_menu = self.addMenu("File")
            
            import_action = file_menu.addAction("Import New Data from CSV Files")
            import_action.triggered.connect(self.parent.import_csv_data)

            # save_action = file_menu.addAction("Save Data")
            # save_action.triggered.connect(self.parent.save_data)

            # load_action = file_menu.addAction("Load Data")
            # load_action.triggered.connect(self.parent.load_data)

            # refresh existing datasets button
            refresh_datasets_action = file_menu.addAction("Refresh Datasets/Sessions Lists")
            refresh_datasets_action.triggered.connect(self.parent.refresh_existing_datasets)
            refresh_datasets_action.setShortcut(QKeySequence.StandardKey.Refresh)

        def create_edit_menu(self):
            # Edit menu
            edit_menu = self.addMenu("Edit")

            # # Add undo and redo buttons to the menu bar
            self.undo_action = edit_menu.addAction("Undo")
            self.redo_action = edit_menu.addAction("Redo")
            self.undo_action.triggered.connect(self.parent.undo)
            self.redo_action.triggered.connect(self.parent.redo)
            self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
            self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)

            # Change channel names button
            change_names_action = edit_menu.addAction("Change Channel Names")
            change_names_action.triggered.connect(self.parent.change_channel_names)

            # Update window settings button
            update_window_action = edit_menu.addAction("Update Reflex Time Windows")
            update_window_action.triggered.connect(self.parent.update_reflex_settings)

            