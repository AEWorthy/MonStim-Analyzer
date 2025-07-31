"""
Example of how you could expand QSettings usage for UI state while keeping YAML for analysis configs.
"""
from PyQt6.QtCore import QSettings
from typing import Dict, List

class ApplicationState:
    """Manage application state using QSettings (separate from analysis config)."""
    
    def __init__(self):
        self.settings = QSettings()
    
    # === RECENT FILES & SESSION RESTORATION ===
    def save_recent_experiment(self, experiment_id: str):
        """Save recently opened experiment."""
        recent = self.get_recent_experiments()
        if experiment_id in recent:
            recent.remove(experiment_id)
        recent.insert(0, experiment_id)
        recent = recent[:10]  # Keep only last 10
        self.settings.setValue("RecentFiles/experiments", recent)
    
    def get_recent_experiments(self) -> List[str]:
        """Get list of recently opened experiments."""
        return self.settings.value("RecentFiles/experiments", [], type=list)
    
    # Last selected items (for session restoration)
    def save_last_selection(self, experiment_id: str = None, dataset_id: str = None, session_id: str = None):
        """Save the last selected experiment/dataset/session."""
        if experiment_id is not None:
            self.settings.setValue("LastSelection/experiment", experiment_id)
        if dataset_id is not None:
            self.settings.setValue("LastSelection/dataset", dataset_id)
        if session_id is not None:
            self.settings.setValue("LastSelection/session", session_id)
    
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
    
    def get_recent_profiles(self) -> List[str]:
        """Get list of recently used analysis profiles."""
        return self.settings.value("RecentProfiles/names", [], type=list)
    
    def save_last_profile(self, profile_name: str):
        """Save the last selected analysis profile."""
        self.settings.setValue("LastSelection/profile", profile_name)
    
    def get_last_profile(self) -> str:
        """Get the last selected analysis profile."""
        return self.settings.value("LastSelection/profile", "(default)", type=str)
    
    # === UI STATE & PREFERENCES ===
    def save_plot_preferences(self, show_grid: bool, show_legend: bool, plot_style: str):
        """Save plot display preferences (not analysis parameters)."""
        self.settings.setValue("PlotDisplay/show_grid", show_grid)
        self.settings.setValue("PlotDisplay/show_legend", show_legend)
        self.settings.setValue("PlotDisplay/style", plot_style)
    
    def get_plot_preferences(self) -> Dict[str, any]:
        """Get plot display preferences."""
        return {
            'show_grid': self.settings.value("PlotDisplay/show_grid", True, type=bool),
            'show_legend': self.settings.value("PlotDisplay/show_legend", True, type=bool),
            'style': self.settings.value("PlotDisplay/style", "default", type=str)
        }
    
    # === IMPORT/EXPORT PATHS ===
    def save_last_import_path(self, path: str):
        """Save the last directory used for importing experiments."""
        self.settings.setValue("LastPaths/import_directory", path)
        
    def get_last_import_path(self) -> str:
        """Get the last directory used for importing experiments."""
        return self.settings.value("LastPaths/import_directory", "", type=str)
    
    def save_last_export_path(self, path: str):
        """Save the last directory used for exporting reports."""
        self.settings.setValue("LastPaths/export_directory", path)
        
    def get_last_export_path(self) -> str:
        """Get the last directory used for exporting reports."""
        return self.settings.value("LastPaths/export_directory", "", type=str)
    
    # === DATA COMPLETION TRACKING ===
    def save_completion_filter_preferences(self, show_completed: bool, show_incomplete: bool):
        """Save preferences for filtering completed/incomplete items."""
        self.settings.setValue("DataFilters/show_completed", show_completed)
        self.settings.setValue("DataFilters/show_incomplete", show_incomplete)
    
    def get_completion_filter_preferences(self) -> Dict[str, bool]:
        """Get preferences for filtering completed/incomplete items."""
        return {
            'show_completed': self.settings.value("DataFilters/show_completed", True, type=bool),
            'show_incomplete': self.settings.value("DataFilters/show_incomplete", True, type=bool)
        }
    
    # === APPLICATION BEHAVIOR ===
    def save_confirmation_preferences(self, confirm_delete: bool, confirm_overwrite: bool, 
                                    confirm_exclude: bool, auto_save: bool):
        """Save user preferences for confirmation dialogs and auto-save."""
        self.settings.setValue("Confirmations/delete_experiment", confirm_delete)
        self.settings.setValue("Confirmations/overwrite_import", confirm_overwrite)
        self.settings.setValue("Confirmations/exclude_data", confirm_exclude)
        self.settings.setValue("Behavior/auto_save", auto_save)
    
    def get_confirmation_preferences(self) -> Dict[str, bool]:
        """Get user preferences for confirmation dialogs and auto-save."""
        return {
            'confirm_delete': self.settings.value("Confirmations/delete_experiment", True, type=bool),
            'confirm_overwrite': self.settings.value("Confirmations/overwrite_import", True, type=bool),
            'confirm_exclude': self.settings.value("Confirmations/exclude_data", True, type=bool),
            'auto_save': self.settings.value("Behavior/auto_save", False, type=bool)
        }
    
    # === PERFORMANCE PREFERENCES ===
    def save_performance_preferences(self, max_workers: int, cache_size_mb: int, 
                                   show_progress_dialogs: bool):
        """Save performance-related preferences."""
        self.settings.setValue("Performance/max_workers", max_workers)
        self.settings.setValue("Performance/cache_size_mb", cache_size_mb)
        self.settings.setValue("Performance/show_progress_dialogs", show_progress_dialogs)
    
    def get_performance_preferences(self) -> Dict[str, any]:
        """Get performance-related preferences."""
        import multiprocessing
        return {
            'max_workers': self.settings.value("Performance/max_workers", 
                                             max(1, multiprocessing.cpu_count() - 1), type=int),
            'cache_size_mb': self.settings.value("Performance/cache_size_mb", 100, type=int),
            'show_progress_dialogs': self.settings.value("Performance/show_progress_dialogs", True, type=bool)
        }
    
    # Workspace layout
    def save_workspace_layout(self, layout_name: str, layout_data: dict):
        """Save workspace layout configuration."""
        self.settings.setValue(f"WorkspaceLayout/{layout_name}", layout_data)
    
    def get_workspace_layout(self, layout_name: str) -> dict:
        """Get workspace layout configuration."""
        return self.settings.value(f"WorkspaceLayout/{layout_name}", {}, type=dict)
