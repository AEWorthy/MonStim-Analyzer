from PyQt6.QtWidgets import QMenuBar, QMessageBox
from PyQt6.QtGui import QKeySequence, QFont
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gui_main import EMGAnalysisGUI

class MenuBar(QMenuBar):
    def __init__(self, parent : 'EMGAnalysisGUI'):
        super().__init__(parent)
        self.parent = parent # type: EMGAnalysisGUI
        self.create_file_menu()
        self.create_edit_menu()
        self.create_help_menu()
    
    def create_file_menu(self):
        # File menu
        file_menu = self.addMenu("File")
        
        import_action = file_menu.addAction("Import an Experiment")
        import_action.triggered.connect(self.parent.import_expt_data)

        # save_action = file_menu.addAction("Save Data")
        # save_action.triggered.connect(self.parent.save_data)

        # load_action = file_menu.addAction("Load Data")
        # load_action.triggered.connect(self.parent.load_data)

        # refresh existing datasets button
        refresh_datasets_action = file_menu.addAction("Refresh Datasets/Sessions Lists")
        refresh_datasets_action.triggered.connect(self.parent.refresh_existing_experiments)
        refresh_datasets_action.setShortcut(QKeySequence.StandardKey.Refresh)

        # Preferences button
        preferences_action = file_menu.addAction("Preferences")
        preferences_action.triggered.connect(self.parent.show_preferences_window)

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

        # Reload current session button
        reload_action = edit_menu.addAction("Reload Current Session")
        reload_action.triggered.connect(self.confirm_reload)

        # Change channel names button
        change_names_action = edit_menu.addAction("Change Channel Names")
        change_names_action.triggered.connect(self.parent.change_channel_names)

        # Update window settings button
        update_window_action = edit_menu.addAction("Update Reflex Time Windows")
        update_window_action.triggered.connect(self.parent.update_reflex_settings)

        # Invert channel polarity button
        invert_polarity_action = edit_menu.addAction("Invert Channel Polarity")
        invert_polarity_action.triggered.connect(self.parent.invert_channel_polarity)
    
    def create_help_menu(self):
        # Help menu
        help_menu = self.addMenu("Help")

        # About button
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.parent.show_about_screen)

        # Show Help button
        help_action = help_menu.addAction("Show Help")
        help_action.triggered.connect(lambda: self.parent.show_help_dialog('help'))

        # Show EMG processing info button
        processing_info_action = help_menu.addAction("Show EMG Processing Info")
        processing_info_action.triggered.connect(lambda: self.parent.show_help_dialog('emg_processing'))

    def confirm_reload(self):
            reply = QMessageBox.question(self, 'Confirm Reload', 
                                 'Are you sure you want to reload the current session?\n\nNote: This will add back any recordings that were removed.',
                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                 QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.parent.reload_current_session()
    
    # Update functions
    def update_undo_redo_labels(self):
        undo_command_name = self.parent.command_invoker.get_undo_command_name()
        redo_command_name = self.parent.command_invoker.get_redo_command_name()

        undo_text = "Undo"
        redo_text = "Redo"

        # Add the command name as a hint if available
        if undo_command_name:
            undo_text += f" ({undo_command_name})"
        if redo_command_name:
            redo_text += f" ({redo_command_name})"

        # Set shadowed or disabled effect for the hint part
        hint_font = QFont()
        hint_font.setItalic(True)

        self.undo_action.setText(undo_text)
        self.redo_action.setText(redo_text)

        # Optionally apply the hint style to make the hint part different (e.g., shadowed)
        self.undo_action.setFont(hint_font)
        self.redo_action.setFont(hint_font)