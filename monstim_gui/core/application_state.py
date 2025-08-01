"""
Application state management using QSettings.
Handles UI state and user preferences that should persist across sessions.
"""
from PyQt6.QtCore import QSettings
from typing import Dict, List
import os
import logging


class ApplicationState:
    """Manage application state using QSettings (separate from analysis config)."""
    
    def __init__(self):
        self.settings = QSettings()
    
    # === IMPORT/EXPORT PATH MEMORY ===
    def save_last_import_path(self, path: str):
        """Save the last directory used for importing experiments."""
        if not self.should_track_import_export_paths():
            return
        if path and os.path.isdir(path):
            self.settings.setValue("LastPaths/import_directory", path)
            self.settings.sync()
        
    def get_last_import_path(self) -> str:
        """Get the last directory used for importing experiments."""
        if not self.should_track_import_export_paths():
            return ""
        return self.settings.value("LastPaths/import_directory", "", type=str)
    
    def save_last_export_path(self, path: str):
        """Save the last directory used for exporting reports."""
        if not self.should_track_import_export_paths():
            return
        if path and os.path.isdir(path):
            self.settings.setValue("LastPaths/export_directory", path)
            self.settings.sync()
        
    def get_last_export_path(self) -> str:
        """Get the last directory used for exporting reports."""
        if not self.should_track_import_export_paths():
            return ""
        return self.settings.value("LastPaths/export_directory", "", type=str)
    
    # === RECENT FILES & SESSION RESTORATION ===
    def save_recent_experiment(self, experiment_id: str):
        """Save recently opened experiment."""
        if not self.should_track_recent_files():
            return
        recent = self.get_recent_experiments()
        if experiment_id in recent:
            recent.remove(experiment_id)
        recent.insert(0, experiment_id)
        recent = recent[:10]  # Keep only last 10
        self.settings.setValue("RecentFiles/experiments", recent)
        self.settings.sync()
    
    def get_recent_experiments(self) -> List[str]:
        """Get list of recently opened experiments."""
        if not self.should_track_recent_files():
            return []
        return self.settings.value("RecentFiles/experiments", [], type=list)
    
    # === SESSION RESTORATION ===
    def save_current_session_state(self, experiment_id: str = None, dataset_id: str = None, 
                                  session_id: str = None, profile_name: str = None):
        """Save the current complete session state for restoration on startup."""
        if not self.should_track_session_restoration():
            return
            
        session_state = {}
        
        if experiment_id is not None:
            session_state['experiment'] = experiment_id
        if dataset_id is not None:
            session_state['dataset'] = dataset_id
        if session_id is not None:
            session_state['session'] = session_id
        if profile_name is not None:
            session_state['profile'] = profile_name
            
        # Only save if we have at least an experiment
        if session_state.get('experiment'):
            self.settings.setValue("SessionRestore/last_state", session_state)
            self.settings.sync()
    
    def get_last_session_state(self) -> Dict[str, str]:
        """Get the last saved session state."""
        return self.settings.value("SessionRestore/last_state", {}, type=dict)
    
    def clear_session_state(self):
        """Clear the saved session state (useful on manual session changes)."""
        self.settings.remove("SessionRestore/last_state")
        self.settings.sync()
    
    def should_restore_session(self) -> bool:
        """Check if there's a valid session state to restore and restoration is enabled."""
        if not self.should_track_session_restoration():
            return False
        state = self.get_last_session_state()
        return bool(state.get('experiment'))

    # Last selected items (for session restoration)
    def save_last_selection(self, experiment_id: str = None, dataset_id: str = None, session_id: str = None):
        """Save the last selected experiment/dataset/session."""
        if experiment_id is not None:
            self.settings.setValue("LastSelection/experiment", experiment_id)
        if dataset_id is not None:
            self.settings.setValue("LastSelection/dataset", dataset_id)
        if session_id is not None:
            self.settings.setValue("LastSelection/session", session_id)
        self.settings.sync()
    
    def get_last_selection(self) -> Dict[str, str]:
        """Get the last selected items."""
        return {
            'experiment': self.settings.value("LastSelection/experiment", "", type=str),
            'dataset': self.settings.value("LastSelection/dataset", "", type=str),
            'session': self.settings.value("LastSelection/session", "", type=str)
        }
    
    # === ANALYSIS PROFILES ===
    def save_recent_profile(self, profile_name: str):
        """Save recently used analysis profile."""
        recent = self.get_recent_profiles()
        if profile_name in recent:
            recent.remove(profile_name)
        recent.insert(0, profile_name)
        recent = recent[:5]  # Keep only last 5
        self.settings.setValue("RecentProfiles/names", recent)
        self.settings.sync()
    
    def get_recent_profiles(self) -> List[str]:
        """Get list of recently used analysis profiles."""
        return self.settings.value("RecentProfiles/names", [], type=list)
    
    def save_last_profile(self, profile_name: str):
        """Save the last selected analysis profile."""
        if not self.should_track_analysis_profiles():
            return
        self.settings.setValue("LastSelection/profile", profile_name)
        self.settings.sync()
    
    def get_last_profile(self) -> str:
        """Get the last selected analysis profile."""
        if not self.should_track_analysis_profiles():
            return "(default)"
        return self.settings.value("LastSelection/profile", "(default)", type=str)
    
    # === SESSION RESTORATION METHODS ===
    def restore_last_session(self, gui) -> bool:
        """
        Attempt to restore the last session state.
        Returns True if restoration was attempted, False if no valid state exists.
        """
        if not self.should_restore_session():
            return False
            
        state = self.get_last_session_state()
        experiment_id = state.get('experiment')
        dataset_id = state.get('dataset')
        session_id = state.get('session')  
        profile_name = state.get('profile')
        
        try:
            # First restore the profile if available
            if profile_name and hasattr(gui, 'profile_selector_combo'):
                profile_index = gui.profile_selector_combo.findText(profile_name)
                if profile_index >= 0:
                    gui.profile_selector_combo.setCurrentIndex(profile_index)
            
            # Check if experiment still exists
            if experiment_id not in gui.expts_dict_keys:
                logging.info(f"Cannot restore session: experiment '{experiment_id}' no longer exists")
                self.clear_session_state()
                return False
            
            # Restore experiment
            exp_index = gui.expts_dict_keys.index(experiment_id) + 1  # +1 for placeholder
            gui.data_selection_widget.experiment_combo.setCurrentIndex(exp_index)
            
            # Wait for experiment to load, then restore dataset/session
            if dataset_id or session_id:
                # Schedule dataset/session restoration for after experiment loads
                from PyQt6.QtCore import QTimer
                def restore_nested():
                    if gui.current_experiment and gui.current_experiment.id == experiment_id:
                        try:
                            # Restore dataset
                            if dataset_id:
                                dataset_names = [ds.id for ds in gui.current_experiment.datasets]
                                if dataset_id in dataset_names:
                                    ds_index = dataset_names.index(dataset_id)
                                    gui.data_selection_widget.dataset_combo.setCurrentIndex(ds_index)
                                    
                                    # Restore session
                                    if session_id and gui.current_dataset:
                                        session_names = [sess.id for sess in gui.current_dataset.sessions]
                                        if session_id in session_names:
                                            sess_index = session_names.index(session_id)
                                            gui.data_selection_widget.session_combo.setCurrentIndex(sess_index)
                        except Exception as e:
                            logging.error(f"Error restoring dataset/session: {e}")
                
                QTimer.singleShot(1000, restore_nested)  # Give experiment time to load
            
            logging.info(f"Session restoration initiated: {experiment_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error during session restoration: {e}")
            self.clear_session_state()
            return False
    
    # === PROGRAM PREFERENCES ===
    def get_preference(self, key: str, default_value=True) -> bool:
        """Get a program preference setting."""
        return self.settings.value(f"ProgramPreferences/{key}", default_value, type=bool)
    
    def set_preference(self, key: str, value: bool):
        """Set a program preference setting."""
        self.settings.setValue(f"ProgramPreferences/{key}", value)
        self.settings.sync()
    
    def should_track_session_restoration(self) -> bool:
        """Check if session restoration tracking is enabled."""
        return self.get_preference("track_session_restoration", True)
    
    def should_track_import_export_paths(self) -> bool:
        """Check if import/export path tracking is enabled."""
        return self.get_preference("track_import_export_paths", True)
    
    def should_track_recent_files(self) -> bool:
        """Check if recent files tracking is enabled."""
        return self.get_preference("track_recent_files", True)
    
    def should_track_analysis_profiles(self) -> bool:
        """Check if analysis profile tracking is enabled."""
        return self.get_preference("track_analysis_profiles", True)
    
    def clear_all_tracked_data(self):
        """Clear all tracked user data."""
        # Clear session restoration data
        self.clear_session_state()
        self.settings.remove("LastSelection")
        
        # Clear path memory
        self.settings.remove("LastPaths")
        
        # Clear recent files
        self.settings.remove("RecentFiles")
        
        # Clear profile data
        self.settings.remove("RecentProfiles")
        
        self.settings.sync()
        logging.info("All tracked user data cleared")


# Global application state instance
app_state = ApplicationState()
