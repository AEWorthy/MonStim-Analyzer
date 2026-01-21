"""
Application state management using QSettings.
Handles UI state and user preferences that should persist across sessions.
"""

import logging
import os
from typing import TYPE_CHECKING, Dict, List

from PySide6.QtCore import QSettings

if TYPE_CHECKING:
    from monstim_gui.gui_main import MonstimGUI


class ApplicationState:
    """Manage application state using QSettings (separate from analysis config)."""

    def __init__(self):
        self._settings = None
        self._is_restoring_session = False  # Flag to suppress saves during restoration
        self._pending_dataset_id = None
        self._pending_session_id = None
        self._pending_profile_name = None
        self._pending_experiment_id = None

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
            logging.debug(
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
    def restore_last_session(self, gui: "MonstimGUI") -> bool:
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

            # Store restoration targets for after experiment loads
            self._pending_experiment_id = experiment_id
            self._pending_dataset_id = dataset_id
            self._pending_session_id = session_id
            self._pending_profile_name = profile_name

            # TODO: Robust restoration
            # - Prefer restoring by explicit IDs stored in combo UserRole instead of
            #   by index arithmetic (+1 placeholder). Where possible, always write
            #   and restore user-facing state by stable IDs to avoid fragile index
            #   based restoring when UI ordering or placeholders change.

            logging.info(
                f"Session restoration in progress: Experiment={experiment_id}, Dataset={dataset_id}, Session={session_id}."
            )

            # Restore experiment - this automatically triggers load_experiment() via signal
            # The restoration of dataset/session will happen in the experiment loaded callback
            exp_index = gui.expts_dict_keys.index(experiment_id) + 1  # +1 for placeholder
            gui.data_selection_widget.experiment_combo.setCurrentIndex(exp_index)
            return True

        except Exception as e:
            logging.error(f"Error during session restoration: {e}")
            self.clear_session_state()
            # Make sure to clear the flag on error
            self._is_restoring_session = False
            self._pending_dataset_id = None
            self._pending_session_id = None
            self._pending_profile_name = None
            self._pending_experiment_id = None
            return False

    def complete_session_restoration(self, gui: "MonstimGUI"):
        """
        Complete session restoration after experiment has loaded.
        Called by data_manager after experiment loading finishes.
        """
        if not self._is_restoring_session:
            return

        try:
            experiment_id = self._pending_experiment_id
            dataset_id = self._pending_dataset_id
            session_id = self._pending_session_id
            # Check if restoration was canceled (pending IDs would be None)
            if experiment_id is None:
                logging.debug("Session restoration was canceled - skipping completion")
                self._is_restoring_session = False
                return
            # Verify experiment loaded correctly
            if not gui.current_experiment or gui.current_experiment.id != experiment_id:
                logging.warning(
                    f"Session restoration: Experiment mismatch (expected '{experiment_id}', got '{gui.current_experiment.id if gui.current_experiment else 'None'}')"
                )
                return

            # Restore dataset
            if dataset_id:
                dataset_names = [ds.id for ds in gui.current_experiment.datasets]
                if dataset_id in dataset_names:
                    ds_index = dataset_names.index(dataset_id)
                    logging.info(f"Session restoration: Restoring dataset '{dataset_id}' at index {ds_index}")
                    gui.data_manager.load_dataset(ds_index)

                    # Restore session
                    if session_id and gui.current_dataset:
                        session_names = [sess.id for sess in gui.current_dataset.sessions]
                        if session_id in session_names:
                            sess_index = session_names.index(session_id)
                            logging.info(f"Session restoration: Restoring session '{session_id}' at index {sess_index}")
                            gui.data_manager.load_session(sess_index)
                        else:
                            logging.warning(f"Session restoration: session '{session_id}' not found in dataset '{dataset_id}'")
                    elif session_id:
                        logging.warning(f"Session restoration: Cannot restore session '{session_id}' - no dataset loaded")
                else:
                    logging.warning(f"Session restoration: dataset '{dataset_id}' not found in experiment '{experiment_id}'")

            # After loading, ensure visual combo selections are synced with actual state
            gui.data_selection_widget.update()

        except Exception as e:
            logging.error(f"Error completing session restoration: {e}")
            import traceback

            logging.error(traceback.format_exc())
        finally:
            # Always clear the restoration flag and pending state when done
            self._is_restoring_session = False
            # For empty experiments, ensure combos show correct state after restoration
            if gui.current_experiment and not gui.current_experiment.datasets:
                gui.data_selection_widget.update(levels=("dataset", "session"))

            # Save the final restored state
            profile_name = self._pending_profile_name
            experiment_id = self._pending_experiment_id
            dataset_id = self._pending_dataset_id
            session_id = self._pending_session_id

            # Clear pending state
            self._pending_dataset_id = None
            self._pending_session_id = None
            self._pending_profile_name = None
            self._pending_experiment_id = None

            # Save final state
            self.save_current_session_state(
                experiment_id=experiment_id,
                dataset_id=dataset_id,
                session_id=session_id,
                profile_name=profile_name,
            )

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
        return self.get_preference("use_opengl_acceleration", False)

    # === PERFORMANCE / LOADING PREFS ===
    def should_use_lazy_open_h5(self) -> bool:
        """Return whether HDF5 files should be opened lazily during experiment load.

        Default: True (faster initial load; raw data is re-opened lazily when required)
        """
        return self.get_preference("use_lazy_open_h5", True)

    def should_use_parallel_loading(self) -> bool:
        """Return whether parallel dataset loading should be enabled.

        Default: True (use multiple threads to load independent datasets)
        """
        pref = self.get_preference("enable_parallel_loading", True)
        # Parallel loading relies on lazy-opening HDF5 files. If lazy-open
        # is disabled we must not enable parallel loading because that would
        # risk opening many HDF5 handles concurrently (unsafe/slow).
        if pref and not self.should_use_lazy_open_h5():
            logging.debug("Parallel loading requested but disabled because lazy_open_h5 is False")
            return False
        return pref

    def get_parallel_load_workers(self) -> int:
        """Get the number of worker threads to use for parallel dataset loading.

        Behavior:
            - If the user has explicitly stored an integer under
              ProgramPreferences/parallel_load_workers, that value is used.
            - Otherwise, choose max(1, os.cpu_count() - 1) so we leave one
              core free for UI and general OS scheduling. If os.cpu_count()
              returns None, default to 1.
        """
        try:
            # If user explicitly set a value, use it
            val = self.settings.value("ProgramPreferences/parallel_load_workers", None, type=int)
            if isinstance(val, int) and val > 0:
                return val
        except Exception as e:
            # Failed to read parallel_load_workers from QSettings; falling back to default.
            logging.warning(f"Error reading ProgramPreferences/parallel_load_workers: {e}")

        # Default to CPU count - 1 (leave one core spare); if single-core, use 1
        try:
            import os

            count = os.cpu_count() or 1
            return max(1, count - 1)
        except Exception:
            return 1

    def should_build_index_on_load(self) -> bool:
        """Check whether experiment indexes should be (re)built during load."""
        return self.get_preference("build_index_on_load", True)

    def set_build_index_on_load(self, enabled: bool):
        """Set preference for building experiment indexes during load."""
        self.set_setting("build_index_on_load", bool(enabled))

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
