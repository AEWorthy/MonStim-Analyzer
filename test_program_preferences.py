#!/usr/bin/env python3
"""
Test script for Program Preferences functionality
"""
import sys
import os
from PyQt6.QtWidgets import QApplication

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from monstim_gui.dialogs.program_preferences import ProgramPreferencesDialog
from monstim_gui.core.application_state import ApplicationState

def test_program_preferences():
    """Test the Program Preferences dialog."""
    app = QApplication(sys.argv)
    
    # Test ApplicationState preferences methods
    app_state = ApplicationState()
    
    print("Testing ApplicationState preference methods:")
    print(f"Initial session tracking: {app_state.should_track_session_restoration()}")
    print(f"Initial profile tracking: {app_state.should_track_analysis_profiles()}")
    print(f"Initial path tracking: {app_state.should_track_import_export_paths()}")
    print(f"Initial recent files tracking: {app_state.should_track_recent_files()}")
    
    # Test setting preferences
    app_state.set_preference("track_session_restoration", False)
    print(f"After setting session tracking to False: {app_state.should_track_session_restoration()}")
    
    # Reset for testing
    app_state.set_preference("track_session_restoration", True)
    
    # Test the dialog
    dialog = ProgramPreferencesDialog()
    
    print("\\nShowing Program Preferences dialog...")
    result = dialog.exec()
    
    if result == ProgramPreferencesDialog.DialogCode.Accepted:
        print("Dialog accepted - preferences saved")
    else:
        print("Dialog cancelled")
        
    print("\\nFinal preference states:")
    print(f"Session tracking: {app_state.should_track_session_restoration()}")
    print(f"Profile tracking: {app_state.should_track_analysis_profiles()}")
    print(f"Path tracking: {app_state.should_track_import_export_paths()}")
    print(f"Recent files tracking: {app_state.should_track_recent_files()}")
    
    # Keep app reference to avoid garbage collection
    return app

if __name__ == "__main__":
    app = test_program_preferences()
    sys.exit(0)
