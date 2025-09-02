"""
Application state management using QSettings.
Handles UI state and user preferences that should persist across sessions.
"""

import logging
import os
from typing import Dict, List

from PyQt6.QtCore import QSettings


class ApplicationState:
    """Manage application state using QSettings (separate from analysis config)."""

    def __init__(self):
        self._settings = None
        self._is_restoring_session = False  # Flag to suppress saves during restoration

        logging.debug(
            "Initializing ApplicationState"
            f" QSettings org={self.settings.organizationName()}, app={self.settings.applicationName()}"
        )

    @property
    def settings(self):
        """Lazy-loaded QSettings instance."""
        if self._settings is None:
            self._settings = QSettings()
        return self._settings

    def reinitialize_settings(self):
        """Reinitialize QSettings (call after QApplication org/app name is set)."""
        self._settings = None  # Force recreation on next access

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
    def save_current_session_state(
        self,
        experiment_id: str = None,
        dataset_id: str = None,
        session_id: str = None,
        profile_name: str = None,
    ):
        """Save the current complete session state for restoration on startup."""

        # Skip saving if we're currently restoring a session (to avoid redundant saves)
        if self._is_restoring_session:
            logging.debug(
                f"Skipping session state save during restoration: experiment={experiment_id}, dataset={dataset_id}, session={session_id}"
            )
            return

        # Debug: log existing persisted state before modification
        try:
            existing = {
                "experiment": self.settings.value("SessionRestore/experiment", "", type=str),
                "dataset": self.settings.value("SessionRestore/dataset", "", type=str),
                "session": self.settings.value("SessionRestore/session", "", type=str),
                "profile": self.settings.value("SessionRestore/profile", "", type=str),
            }
            logging.debug(f"save_current_session_state: existing SessionRestore={existing}")
        except Exception:
            logging.debug("save_current_session_state: could not read existing SessionRestore")

        # Always save profile information if provided (independent of session restoration setting)
        if profile_name is not None:
            self.save_last_profile(profile_name)

        if not self.should_track_session_restoration():
            logging.debug("save_current_session_state: Session restoration tracking is disabled")
            return

        # Only save session state if we have at least an experiment
        if experiment_id is not None:
            logging.info(
                f"save_current_session_state: Saving experiment={experiment_id}, dataset={dataset_id}, session={session_id}, profile={profile_name}"
                f" QSettings org={self.settings.organizationName()}, app={self.settings.applicationName()}"
            )

            # Determine whether the experiment changed compared to what's already saved
            previous_experiment = self.settings.value("SessionRestore/experiment", "", type=str)

            # If experiment differs from previously saved experiment, clear dataset/session
            # unless the caller explicitly provided dataset_id/session_id. This prevents
            # the situation where a dataset from a different experiment is preserved and
            # later restored incorrectly.
            if previous_experiment and previous_experiment != experiment_id:
                # Clear cross-experiment dataset/session unless explicitly provided
                if dataset_id is None:
                    try:
                        self.settings.remove("SessionRestore/dataset")
                    except Exception:
                        logging.debug("Could not remove stale SessionRestore/dataset key")
                if session_id is None:
                    try:
                        self.settings.remove("SessionRestore/session")
                    except Exception:
                        logging.debug("Could not remove stale SessionRestore/session key")

            # Save experiment id
            self.settings.setValue("SessionRestore/experiment", experiment_id)

            # Only update dataset/session/profile if explicitly provided (non-None)
            if dataset_id is not None:
                self.settings.setValue("SessionRestore/dataset", dataset_id)
            if session_id is not None:
                self.settings.setValue("SessionRestore/session", session_id)
            if profile_name is not None:
                self.settings.setValue("SessionRestore/profile", profile_name)

            self.settings.sync()
        else:
            logging.debug("save_current_session_state: No experiment_id provided, not saving session state")

    def get_last_session_state(self) -> Dict[str, str]:
        """Get the last saved session state."""
        return {
            "experiment": self.settings.value("SessionRestore/experiment", "", type=str),
            "dataset": self.settings.value("SessionRestore/dataset", "", type=str),
            "session": self.settings.value("SessionRestore/session", "", type=str),
            "profile": self.settings.value("SessionRestore/profile", "", type=str),
        }

    def clear_session_state(self):
        """Clear the saved session state (useful on manual session changes)."""
        self.settings.remove("SessionRestore")
        self.settings.sync()

    def should_restore_session(self) -> bool:
        """Check if there's a valid session state to restore and restoration is enabled."""
        if not self.should_track_session_restoration():
            return False
        state = self.get_last_session_state()
        return bool(state.get("experiment"))

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
            "experiment": self.settings.value("LastSelection/experiment", "", type=str),
            "dataset": self.settings.value("LastSelection/dataset", "", type=str),
            "session": self.settings.value("LastSelection/session", "", type=str),
        }

    # === ANALYSIS PROFILES ===
    def save_recent_profile(self, profile_name: str):
        """Save recently used analysis profile."""
        if not self.should_track_analysis_profiles():
            return
        recent = self.get_recent_profiles()
        if profile_name in recent:
            recent.remove(profile_name)
        recent.insert(0, profile_name)
        recent = recent[:5]  # Keep only last 5
        self.settings.setValue("RecentProfiles/names", recent)
        self.settings.sync()

    def get_recent_profiles(self) -> List[str]:
        """Get list of recently used analysis profiles."""
        if not self.should_track_analysis_profiles():
            return []
        return self.settings.value("RecentProfiles/names", [], type=list)

    def save_last_profile(self, profile_name: str):
        """Save the last selected analysis profile."""
        if not self.should_track_analysis_profiles():
            logging.debug(f"Profile tracking is disabled - not saving profile '{profile_name}'")
            return
        logging.debug(f"Saving last profile selection: '{profile_name}'")
        self.settings.setValue("LastSelection/profile", profile_name)
        self.settings.sync()

    def get_last_profile(self) -> str:
        """Get the last selected analysis profile."""
        if not self.should_track_analysis_profiles():
            logging.debug("Profile tracking is disabled - returning default profile")
            return "(default)"
        result = self.settings.value("LastSelection/profile", "(default)", type=str)
        logging.debug(f"Retrieved last profile selection: '{result}'")
        return result

    # === SESSION RESTORATION METHODS ===
    def restore_last_session(self, gui) -> bool:
        """
        Attempt to restore the last session state.
        Returns True if restoration was attempted, False if no valid state exists.
        """
        if not self.should_restore_session():
            return False

        state = self.get_last_session_state()
        experiment_id = state.get("experiment")
        dataset_id = state.get("dataset")
        session_id = state.get("session")
        profile_name = state.get("profile")

        try:
            # First restore the profile if available
            if profile_name and hasattr(gui, "profile_selector_combo"):
                profile_index = gui.profile_selector_combo.findText(profile_name)
                if profile_index >= 0:
                    gui.profile_selector_combo.setCurrentIndex(profile_index)

            # Check if experiment still exists
            if experiment_id not in gui.expts_dict_keys:
                logging.info(f"Cannot restore session: experiment '{experiment_id}' no longer exists")
                self.clear_session_state()
                return False

            # Set flag to suppress session state saves during restoration (including experiment loading)
            self._is_restoring_session = True

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
                                    logging.info(f"Session restoration: Restoring dataset '{dataset_id}' at index {ds_index}")
                                    gui.data_selection_widget.dataset_combo.setCurrentIndex(ds_index)

                                    # Restore session
                                    if session_id and gui.current_dataset:
                                        session_names = [sess.id for sess in gui.current_dataset.sessions]
                                        if session_id in session_names:
                                            sess_index = session_names.index(session_id)
                                            logging.info(
                                                f"Session restoration: Restoring session '{session_id}' at index {sess_index}"
                                            )
                                            gui.data_selection_widget.session_combo.setCurrentIndex(sess_index)
                                        else:
                                            logging.warning(
                                                f"Session restoration: session '{session_id}' not found in dataset '{dataset_id}'"
                                            )
                                    elif session_id:
                                        logging.warning(
                                            f"Session restoration: Cannot restore session '{session_id}' - no dataset loaded"
                                        )
                                else:
                                    logging.warning(
                                        f"Session restoration: dataset '{dataset_id}' not found in experiment '{experiment_id}'"
                                    )
                        except Exception as e:
                            logging.error(f"Error restoring dataset/session: {e}")
                            import traceback

                            logging.error(traceback.format_exc())
                        finally:
                            # Always clear the restoration flag when done
                            self._is_restoring_session = False
                            # Save the final restored state
                            self.save_current_session_state(
                                experiment_id=experiment_id,
                                dataset_id=dataset_id,
                                session_id=session_id,
                                profile_name=profile_name,
                            )
                    else:
                        # Experiment not ready yet, try again in 500ms (but only up to 10 attempts = 5 seconds total)
                        if not hasattr(restore_nested, "attempt_count"):
                            restore_nested.attempt_count = 0

                        restore_nested.attempt_count += 1
                        if restore_nested.attempt_count < 20:
                            logging.debug(
                                f"Session restoration: Waiting for experiment to load (attempt {restore_nested.attempt_count}/10)"
                            )
                            QTimer.singleShot(500, restore_nested)
                        else:
                            logging.warning("Session restoration: Gave up waiting for experiment to load after 5 seconds")
                            # Clear the restoration flag if we give up
                            self._is_restoring_session = False

                QTimer.singleShot(1000, restore_nested)  # Give experiment time to load

            logging.info(
                f"Session restoration in progress: Experiment={experiment_id}, Dataset={dataset_id}, Session={session_id}."
            )
            return True

        except Exception as e:
            logging.error(f"Error during session restoration: {e}")
            self.clear_session_state()
            # Make sure to clear the flag on error
            self._is_restoring_session = False
            return False

    # === PROGRAM PREFERENCES ===
    def get_preference(self, key: str, default_value=True) -> bool:
        logging.debug(
            f"Getting preference '{key}' with default={default_value}"
            f" QSettings org={self.settings.organizationName()}, app={self.settings.applicationName()}"
        )
        """Get a program preference setting."""
        return self.settings.value(f"ProgramPreferences/{key}", default_value, type=bool)

    def set_setting(self, key: str, value: bool):
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

    def should_use_opengl_acceleration(self) -> bool:
        """Check if OpenGL acceleration should be used."""
        return self.get_preference("use_opengl_acceleration", True)

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
