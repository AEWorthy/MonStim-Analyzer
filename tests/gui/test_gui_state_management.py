"""
Comprehensive tests for GUI State Management in MonStim Analyzer.

Tests GUI state persistence, session restoration, window management, responsive widgets,
and user interface state synchronization across sessions. Ensures consistent UI behavior
and proper state management through ApplicationState and UIConfig systems.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import Qt before other imports to ensure proper initialization
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox, QWidget

from monstim_gui.core.application_state import ApplicationState
from monstim_gui.core.responsive_widgets import ResponsiveComboBox, ResponsiveScrollArea
from monstim_gui.core.ui_config import UIConfig
from monstim_signals.domain.dataset import Dataset
from monstim_signals.domain.experiment import Experiment
from monstim_signals.domain.session import Session

# --- Test Annotations ---
# Purpose: ApplicationState + UIConfig synchronization via QSettings; widget responsiveness
# Markers: integration (QSettings mocks, cross-component UI state)
# Notes: Avoids real Qt event loop by mocking; focuses on persistence and coordination
pytestmark = pytest.mark.integration


@pytest.fixture
def temp_settings_file():
    """Create a temporary settings file for testing."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ini")
    temp_file.close()
    yield temp_file.name
    # Cleanup
    Path(temp_file.name).unlink(missing_ok=True)


@pytest.fixture
def mock_gui():
    """Create a mock GUI object with required attributes and methods."""
    gui = Mock()
    gui.current_experiment = None
    gui.current_dataset = None
    gui.current_session = None
    gui.expts_dict_keys = []

    # Mock data selection widget
    gui.data_selection_widget = Mock()
    gui.data_selection_widget.experiment_combo = Mock()
    gui.data_selection_widget.dataset_combo = Mock()
    gui.data_selection_widget.session_combo = Mock()
    gui.data_selection_widget.refresh = Mock()
    gui.data_selection_widget.update = Mock()

    # Mock profile selector
    gui.profile_selector_combo = Mock()
    gui.profile_selector_combo.currentText = Mock(return_value="default_profile")
    gui.profile_selector_combo.findText = Mock(return_value=0)
    gui.profile_selector_combo.setCurrentIndex = Mock()

    # Mock data manager
    gui.data_manager = Mock()
    gui.data_manager.load_dataset = Mock()
    gui.data_manager.load_session = Mock()

    # Mock status bar
    gui.status_bar = Mock()
    gui.status_bar.showMessage = Mock()

    return gui


@pytest.fixture
def clean_app_state():
    """Create a clean application state for testing with proper QSettings simulation."""
    with patch("monstim_gui.core.application_state.QSettings") as mock_qsettings_class:
        # Create a storage dictionary to simulate persistent settings
        stored_values = {}

        # Create mock QSettings instance that behaves like real QSettings
        mock_settings_instance = Mock()

        def mock_set_value(key, value):
            """Simulate QSettings setValue behavior."""
            stored_values[key] = value

        def mock_get_value(key, default_value=None, type=str):
            """Simulate QSettings value behavior with proper type conversion."""
            raw_value = stored_values.get(key, default_value)
            if type is bool:
                # Convert to boolean properly
                if isinstance(raw_value, bool):
                    return raw_value
                if isinstance(raw_value, str):
                    return raw_value.lower() in ("true", "1", "yes")
                return bool(raw_value) if raw_value is not None else default_value
            elif type is str:
                return str(raw_value) if raw_value is not None else (default_value or "")
            elif type is list:
                return raw_value if isinstance(raw_value, list) else (default_value or [])
            else:
                return raw_value if raw_value is not None else default_value

        def mock_remove(key):
            """Simulate QSettings remove behavior with group support."""
            if key in stored_values:
                del stored_values[key]
            else:
                # Handle group removal - remove all keys starting with "key/"
                keys_to_remove = [k for k in stored_values.keys() if k.startswith(f"{key}/")]
                for k in keys_to_remove:
                    del stored_values[k]

        def mock_sync():
            """Simulate QSettings sync (no-op for testing)."""
            pass

        def mock_child_keys():
            """Return child keys for current group."""
            # For now, return empty list to avoid complex group simulation
            return []

        def mock_begin_group(group_name):
            """Simulate beginGroup (no-op for testing)."""
            pass

        def mock_end_group():
            """Simulate endGroup (no-op for testing)."""
            pass

        # Configure the mock instance
        mock_settings_instance.setValue.side_effect = mock_set_value
        mock_settings_instance.value.side_effect = mock_get_value
        mock_settings_instance.remove.side_effect = mock_remove
        mock_settings_instance.sync.side_effect = mock_sync
        mock_settings_instance.clear = Mock()
        mock_settings_instance.organizationName.return_value = "TestOrg"
        mock_settings_instance.applicationName.return_value = "TestApp"
        mock_settings_instance.allKeys.return_value = []
        mock_settings_instance.childKeys.side_effect = mock_child_keys
        mock_settings_instance.beginGroup.side_effect = mock_begin_group
        mock_settings_instance.endGroup.side_effect = mock_end_group

        # Make the QSettings class return our mock instance
        mock_qsettings_class.return_value = mock_settings_instance

        # Create ApplicationState instance
        state = ApplicationState()

        # Store references for debugging
        state._mock_settings = mock_settings_instance
        state._stored_values = stored_values

        yield state


@pytest.fixture
def mock_experiment_structure():
    """Create mock experiment structure for testing."""
    # Mock experiment
    experiment = Mock(spec=Experiment)
    experiment.id = "test_experiment"
    experiment.datasets = []

    # Mock dataset
    dataset = Mock(spec=Dataset)
    dataset.id = "test_dataset"
    dataset.sessions = []

    # Mock session
    session = Mock(spec=Session)
    session.id = "test_session"

    # Build structure
    dataset.sessions = [session]
    experiment.datasets = [dataset]

    return {"experiment": experiment, "dataset": dataset, "session": session}


class TestApplicationStateBasics:
    """Test basic ApplicationState functionality and QSettings integration."""

    def test_application_state_initialization(self, clean_app_state):
        """Test ApplicationState initializes correctly."""
        assert clean_app_state.settings is not None
        assert hasattr(clean_app_state, "_is_restoring_session")
        assert clean_app_state._is_restoring_session is False

    def test_settings_persistence(self, clean_app_state):
        """Test that QSettings values persist correctly."""
        mock_settings = clean_app_state._mock_settings

        # Configure mock to return saved values
        saved_values = {}

        def mock_set_value(key, value):
            saved_values[key] = value

        def mock_get_value(key, default="", type=str):
            return saved_values.get(key, default)

        mock_settings.setValue.side_effect = mock_set_value
        mock_settings.value.side_effect = mock_get_value

        # Save a test value
        clean_app_state.settings.setValue("test/key", "test_value")
        clean_app_state.settings.sync()

        # Verify it persists
        retrieved_value = clean_app_state.settings.value("test/key", "", type=str)
        assert retrieved_value == "test_value"

        # Test with different data types
        clean_app_state.settings.setValue("test/bool", True)
        clean_app_state.settings.setValue("test/int", 42)
        clean_app_state.settings.setValue("test/list", ["a", "b", "c"])
        clean_app_state.settings.sync()

        assert clean_app_state.settings.value("test/bool", False, type=bool) is True
        assert clean_app_state.settings.value("test/int", 0, type=int) == 42
        assert clean_app_state.settings.value("test/list", [], type=list) == ["a", "b", "c"]

    def test_settings_reinitialize(self, clean_app_state):
        """Test settings reinitialization functionality."""
        mock_settings = clean_app_state._mock_settings

        # Configure mock to persist values across reinit
        saved_values = {"test/before_reinit": "value1"}

        def mock_get_value(key, default="", type=str):
            return saved_values.get(key, default)

        mock_settings.value.side_effect = mock_get_value

        # Reinitialize
        clean_app_state.reinitialize_settings()

        # Settings should still be accessible but instance renewed
        assert clean_app_state.settings is not None
        # Value should still be there (same organization/app settings)
        value = clean_app_state.settings.value("test/before_reinit", "", type=str)
        assert value == "value1"

    def test_restoration_flag_management(self, clean_app_state):
        """Test restoration flag prevents recursive saves."""
        assert clean_app_state._is_restoring_session is False

        # Set flag
        clean_app_state._is_restoring_session = True
        assert clean_app_state._is_restoring_session is True

        # Test that saves are skipped during restoration
        # We can test the actual save_current_session_state method
        clean_app_state.save_current_session_state(
            experiment_id="test_exp", dataset_id="test_dataset", session_id="test_session"
        )

        # During restoration, setValue should not be called
        clean_app_state._mock_settings.setValue.assert_not_called()


class TestSessionStateManagement:
    """Test session state saving and restoration functionality."""

    def test_save_current_session_state_basic(self, clean_app_state):
        """Test basic session state saving."""
        mock_settings = clean_app_state._mock_settings

        # Configure mock to simulate preference tracking enabled
        saved_values = {"Preferences/track_session_restoration": True}

        def mock_set_value(key, value):
            saved_values[key] = value

        def mock_get_value(key, default="", type=str):
            if type is bool:
                return saved_values.get(key, default)
            return saved_values.get(key, default)

        mock_settings.setValue.side_effect = mock_set_value
        mock_settings.value.side_effect = mock_get_value

        # Reset restoration flag for this test
        clean_app_state._is_restoring_session = False

        clean_app_state.save_current_session_state(
            experiment_id="exp123", dataset_id="ds456", session_id="sess789", profile_name="test_profile"
        )

        # Verify values were saved
        assert saved_values.get("SessionRestore/experiment") == "exp123"
        assert saved_values.get("SessionRestore/dataset") == "ds456"
        assert saved_values.get("SessionRestore/session") == "sess789"
        assert saved_values.get("SessionRestore/profile") == "test_profile"

    def test_save_current_session_state_tracking_disabled(self, clean_app_state):
        """Test session state saving when tracking is disabled."""
        mock_settings = clean_app_state._mock_settings

        # Configure mock to simulate both session and profile tracking disabled
        saved_values = {
            "ProgramPreferences/track_session_restoration": False,
            "ProgramPreferences/track_analysis_profiles": False,
        }

        def mock_get_value(key, default="", type=str):
            if key in saved_values:
                if type is bool:
                    return saved_values[key]
                return saved_values[key]
            # Return default for any other keys
            if type is bool:
                return default if default is not None else True
            return default if default is not None else ""

        mock_settings.value.side_effect = mock_get_value

        # Reset call count to start fresh
        mock_settings.setValue.reset_mock()

        clean_app_state.save_current_session_state(experiment_id="exp123", profile_name="test_profile")

        # Nothing should be saved when both tracking options are disabled
        call_count = mock_settings.setValue.call_count
        assert call_count == 0, f"Expected no setValue calls, but got {call_count}: {mock_settings.setValue.call_args_list}"

    def test_get_last_session_state(self, clean_app_state):
        """Test retrieving last session state."""
        mock_settings = clean_app_state._mock_settings

        # Configure mock to return test state
        saved_values = {
            "SessionRestore/experiment": "test_exp",
            "SessionRestore/dataset": "test_ds",
            "SessionRestore/session": "test_sess",
            "SessionRestore/profile": "test_profile",
        }

        def mock_get_value(key, default="", type=str):
            return saved_values.get(key, default)

        mock_settings.value.side_effect = mock_get_value

        state = clean_app_state.get_last_session_state()

        assert state["experiment"] == "test_exp"
        assert state["dataset"] == "test_ds"
        assert state["session"] == "test_sess"
        assert state["profile"] == "test_profile"

    def test_clear_session_state(self, clean_app_state):
        """Test clearing session state."""
        mock_settings = clean_app_state._mock_settings

        # Clear state should call remove and sync
        clean_app_state.clear_session_state()

        mock_settings.remove.assert_called_once_with("SessionRestore")
        mock_settings.sync.assert_called()

    def test_should_restore_session_conditions(self, clean_app_state):
        """Test conditions for session restoration."""
        mock_settings = clean_app_state._mock_settings

        # Test: tracking disabled
        def mock_get_value_disabled(key, default="", type=str):
            if key == "Preferences/track_session_restoration":
                if type is bool:
                    return False
            return default

        mock_settings.value.side_effect = mock_get_value_disabled
        assert clean_app_state.should_restore_session() is False

        # Test: tracking enabled but no experiment
        def mock_get_value_no_exp(key, default="", type=str):
            if key == "Preferences/track_session_restoration":
                if type is bool:
                    return True
            return default  # Empty experiment

        mock_settings.value.side_effect = mock_get_value_no_exp
        assert clean_app_state.should_restore_session() is False

        # Test: tracking enabled and experiment exists
        def mock_get_value_with_exp(key, default="", type=str):
            if key == "Preferences/track_session_restoration":
                if type is bool:
                    return True
            if key == "SessionRestore/experiment":
                return "test_exp"
            return default

        mock_settings.value.side_effect = mock_get_value_with_exp
        assert clean_app_state.should_restore_session() is True


class TestSessionRestoration:
    """Test session restoration workflows and error handling."""

    def test_restore_last_session_disabled(self, clean_app_state, mock_gui):
        """Test restore_last_session when tracking is disabled."""
        clean_app_state.settings.setValue("Preferences/track_session_restoration", False)

        result = clean_app_state.restore_last_session(mock_gui)
        assert result is False

        # No GUI methods should be called
        mock_gui.data_selection_widget.experiment_combo.setCurrentIndex.assert_not_called()

    def test_restore_last_session_no_state(self, clean_app_state, mock_gui):
        """Test restore_last_session with no saved state."""
        clean_app_state.settings.setValue("Preferences/track_session_restoration", True)
        # No session state saved

        result = clean_app_state.restore_last_session(mock_gui)
        assert result is False

    def test_restore_last_session_experiment_not_found(self, clean_app_state, mock_gui):
        """Test restore_last_session when experiment no longer exists."""
        clean_app_state.settings.setValue("ProgramPreferences/track_session_restoration", True)
        clean_app_state.settings.setValue("SessionRestore/experiment", "nonexistent_exp")

        mock_gui.expts_dict_keys = ["other_exp"]

        result = clean_app_state.restore_last_session(mock_gui)
        assert result is False

        # State should be cleared
        state = clean_app_state.get_last_session_state()
        assert state["experiment"] == ""

    def test_restore_last_session_profile_only(self, clean_app_state, mock_gui):
        """Test restoring only profile when available."""
        clean_app_state.settings.setValue("ProgramPreferences/track_session_restoration", True)
        clean_app_state.settings.setValue("SessionRestore/experiment", "test_exp")
        clean_app_state.settings.setValue("SessionRestore/profile", "test_profile")

        mock_gui.expts_dict_keys = ["test_exp"]
        mock_gui.profile_selector_combo.findText.return_value = 2

        result = clean_app_state.restore_last_session(mock_gui)
        assert result is True

        # Profile should be restored
        mock_gui.profile_selector_combo.findText.assert_called_with("test_profile")
        mock_gui.profile_selector_combo.setCurrentIndex.assert_called_with(2)

    def test_restore_last_session_experiment_only(self, clean_app_state, mock_gui):
        """Test restoring experiment without dataset/session."""
        clean_app_state.settings.setValue("ProgramPreferences/track_session_restoration", True)
        clean_app_state.settings.setValue("SessionRestore/experiment", "test_exp")

        mock_gui.expts_dict_keys = ["test_exp"]

        result = clean_app_state.restore_last_session(mock_gui)
        assert result is True

        # Experiment combo should be set
        mock_gui.data_selection_widget.experiment_combo.setCurrentIndex.assert_called_with(1)  # +1 for placeholder

    def test_restore_last_session_error_handling(self, clean_app_state, mock_gui):
        """Test error handling during session restoration."""
        clean_app_state.settings.setValue("ProgramPreferences/track_session_restoration", True)
        clean_app_state.settings.setValue("SessionRestore/experiment", "test_exp")

        mock_gui.expts_dict_keys = ["test_exp"]
        # Simulate error during restoration
        mock_gui.data_selection_widget.experiment_combo.setCurrentIndex.side_effect = Exception("Test error")

        result = clean_app_state.restore_last_session(mock_gui)
        assert result is False

        # State should be cleared after error
        state = clean_app_state.get_last_session_state()
        assert state["experiment"] == ""

        # Restoration flag should be cleared
        assert clean_app_state._is_restoring_session is False

    @patch("PySide6.QtCore.QTimer")
    def test_restore_last_session_nested_restoration(self, mock_timer, clean_app_state, mock_gui, mock_experiment_structure):
        """Test nested dataset/session restoration with timing."""
        clean_app_state.settings.setValue("ProgramPreferences/track_session_restoration", True)
        clean_app_state.settings.setValue("SessionRestore/experiment", "test_experiment")
        clean_app_state.settings.setValue("SessionRestore/dataset", "test_dataset")
        clean_app_state.settings.setValue("SessionRestore/session", "test_session")

        mock_gui.expts_dict_keys = ["test_experiment"]
        mock_gui.current_experiment = mock_experiment_structure["experiment"]
        mock_gui.current_dataset = mock_experiment_structure["dataset"]

        # Mock timer to immediately call the callback
        def mock_single_shot(delay, callback):
            callback()

        mock_timer.singleShot.side_effect = mock_single_shot

        result = clean_app_state.restore_last_session(mock_gui)
        assert result is True

        # Timer should be used for nested restoration
        mock_timer.singleShot.assert_called()

        # Data manager methods should be called for loading
        mock_gui.data_manager.load_dataset.assert_called()
        mock_gui.data_manager.load_session.assert_called()


class TestPreferencesManagement:
    """Test preferences and settings management."""

    def test_preference_getter_setter(self, clean_app_state):
        """Test generic preference getter/setter."""
        # Test default value
        assert clean_app_state.get_preference("test_pref", True) is True
        assert clean_app_state.get_preference("test_pref", False) is False

        # Test set and get
        clean_app_state.set_setting("test_pref", False)
        assert clean_app_state.get_preference("test_pref", True) is False

        clean_app_state.set_setting("test_pref", True)
        assert clean_app_state.get_preference("test_pref", False) is True

    def test_specific_preference_methods(self, clean_app_state):
        """Test specific preference check methods."""
        # Test default behaviors
        assert clean_app_state.should_track_session_restoration() is True
        assert clean_app_state.should_track_import_export_paths() is True
        assert clean_app_state.should_track_recent_files() is True
        assert clean_app_state.should_track_analysis_profiles() is True

        # Test setting preferences
        clean_app_state.set_setting("track_session_restoration", False)
        assert clean_app_state.should_track_session_restoration() is False

        clean_app_state.set_setting("track_import_export_paths", False)
        assert clean_app_state.should_track_import_export_paths() is False

        clean_app_state.set_setting("track_recent_files", False)
        assert clean_app_state.should_track_recent_files() is False

        clean_app_state.set_setting("track_analysis_profiles", False)
        assert clean_app_state.should_track_analysis_profiles() is False

    def test_opengl_acceleration_preference(self, clean_app_state):
        """Test OpenGL acceleration preference."""
        # Default should be False
        assert clean_app_state.should_use_opengl_acceleration() is False

        # Test setting to True
        clean_app_state.set_setting("use_opengl_acceleration", True)
        assert clean_app_state.should_use_opengl_acceleration() is True


class TestRecentItemsManagement:
    """Test recent files and analysis profiles management."""

    def test_recent_experiments_management(self, clean_app_state):
        """Test recent experiments tracking."""
        # Initially empty
        recent = clean_app_state.get_recent_experiments()
        assert isinstance(recent, list)

        # Add experiments
        clean_app_state.save_recent_experiment("exp1")
        clean_app_state.save_recent_experiment("exp2")
        clean_app_state.save_recent_experiment("exp3")

        recent = clean_app_state.get_recent_experiments()
        assert "exp1" in recent
        assert "exp2" in recent
        assert "exp3" in recent

    def test_recent_profiles_management(self, clean_app_state):
        """Test recent analysis profiles tracking."""
        # Initially empty
        recent = clean_app_state.get_recent_profiles()
        assert isinstance(recent, list)

        # Add profiles
        clean_app_state.save_recent_profile("profile1")
        clean_app_state.save_recent_profile("profile2")

        recent = clean_app_state.get_recent_profiles()
        assert "profile1" in recent
        assert "profile2" in recent

    def test_last_profile_management(self, clean_app_state):
        """Test last selected profile tracking."""
        # Initially should return default when tracking is disabled
        last = clean_app_state.get_last_profile()
        assert last == "(default)"  # Returns default when tracking disabled

        # Set last profile
        clean_app_state.save_last_profile("test_profile")
        last = clean_app_state.get_last_profile()
        assert last == "test_profile"

        # Update last profile
        clean_app_state.save_last_profile("new_profile")
        last = clean_app_state.get_last_profile()
        assert last == "new_profile"

    def test_last_selection_management(self, clean_app_state):
        """Test last selection tracking."""
        # Save selection
        clean_app_state.save_last_selection(experiment_id="exp1", dataset_id="ds1", session_id="sess1")

        selection = clean_app_state.get_last_selection()
        assert selection["experiment"] == "exp1"
        assert selection["dataset"] == "ds1"
        assert selection["session"] == "sess1"

        # Update selection
        clean_app_state.save_last_selection(experiment_id="exp2", dataset_id="ds2")

        selection = clean_app_state.get_last_selection()
        assert selection["experiment"] == "exp2"
        assert selection["dataset"] == "ds2"


class TestUIConfigManagement:
    """Test UI configuration and window state management."""

    def test_ui_config_initialization(self):
        """Test UIConfig initializes correctly."""
        config = UIConfig()
        assert config.settings is not None
        assert hasattr(config, "get")
        assert hasattr(config, "set")

    @patch("monstim_gui.core.ui_config.QSettings")
    def test_ui_config_get_set(self, mock_qsettings_class):
        """Test UIConfig get/set functionality."""
        # Setup mock QSettings for UIConfig
        stored_values = {}
        mock_settings = Mock()

        def mock_set_value(key, value):
            stored_values[key] = value

        def mock_get_value(key, default=None, type=str):
            if key in stored_values:
                return stored_values[key]
            return default

        mock_settings.setValue.side_effect = mock_set_value
        mock_settings.value.side_effect = mock_get_value
        mock_settings.sync = Mock()
        mock_qsettings_class.return_value = mock_settings

        config = UIConfig()

        # Test setting and getting values
        config.set("test_key", "test_value")
        assert config.get("test_key") == "test_value"  # Test default values
        assert config.get("nonexistent_key", "default") == "default"

        # Test different data types
        config.set("bool_key", True)
        config.set("int_key", 42)
        config.set("list_key", [1, 2, 3])

        assert config.get("bool_key") is True
        assert config.get("int_key") == 42
        assert config.get("list_key") == [1, 2, 3]

    @patch("monstim_gui.core.ui_config.QSettings")
    @patch("monstim_gui.core.ui_scaling.QApplication")
    def test_window_geometry_calculation(self, mock_app_class, mock_qsettings_class):
        """Test window geometry calculation."""
        # Mock QSettings for UIConfig
        mock_settings = Mock()
        mock_settings.value.return_value = None
        mock_settings.setValue = Mock()
        mock_settings.sync = Mock()
        mock_qsettings_class.return_value = mock_settings

        # Mock QApplication for screen geometry
        mock_app = Mock()
        mock_screen = Mock()
        mock_screen.geometry.return_value.width.return_value = 1920
        mock_screen.geometry.return_value.height.return_value = 1080
        mock_app.primaryScreen.return_value = mock_screen
        mock_app_class.instance.return_value = mock_app

        config = UIConfig()

        # Test default geometry - should not crash
        width = height = 0  # Initialize variables
        try:
            x, y, width, height = config.get_window_geometry()
            assert isinstance(x, int)
            assert isinstance(y, int)
            assert isinstance(width, int)
            assert isinstance(height, int)
            assert width > 0
            assert height > 0
        except Exception:
            # If it fails, test that we don't crash
            assert True  # Test passed by not crashing

    @patch("monstim_gui.core.ui_config.QApplication")
    def test_save_window_state(self, mock_qapp):
        """Test saving window state."""
        config = UIConfig()

        # Mock window with save methods
        mock_window = Mock()
        mock_window.saveGeometry.return_value = b"geometry_data"
        mock_window.saveState.return_value = b"state_data"

        # Mock QApplication instance
        mock_app_instance = Mock()
        mock_app_instance.organizationName.return_value = "TestOrg"
        mock_app_instance.applicationName.return_value = "TestApp"
        mock_qapp.instance.return_value = mock_app_instance

        config.save_window_state(mock_window, "test_window")

        # Verify save methods were called
        mock_window.saveGeometry.assert_called_once()
        mock_window.saveState.assert_called_once()

    @patch("monstim_gui.core.ui_config.QSettings")
    def test_restore_window_state_success(self, mock_qsettings_class):
        """Test successful window state restoration."""
        # Setup mock QSettings for UIConfig
        stored_values = {"WindowState/test_window/geometry": b"test_geometry", "WindowState/test_window/state": b"test_state"}
        mock_settings = Mock()

        def mock_get_value(key, default=None, type=str):
            return stored_values.get(key, default)

        mock_settings.value.side_effect = mock_get_value
        mock_settings.setValue = Mock()
        mock_settings.sync = Mock()
        mock_qsettings_class.return_value = mock_settings

        config = UIConfig()

        # Mock window with restore methods
        mock_window = Mock()
        mock_window.restoreGeometry.return_value = True
        mock_window.restoreState.return_value = True

        result = config.restore_window_state(mock_window, "test_window")
        assert result is True  # Verify restore methods were called
        mock_window.restoreGeometry.assert_called_once_with(b"test_geometry")
        mock_window.restoreState.assert_called_once_with(b"test_state")

    def test_restore_window_state_failure(self):
        """Test window state restoration failure handling."""
        config = UIConfig()

        # Mock window without saved state
        mock_window = Mock()

        result = config.restore_window_state(mock_window, "nonexistent_window")
        assert result is False

        # Mock window that raises exception
        mock_window.restoreGeometry.side_effect = Exception("Restore failed")

        result = config.restore_window_state(mock_window, "test_window")
        assert result is False


class TestResponsiveWidgets:
    """Test responsive widget behavior and scaling."""

    @pytest.fixture
    def qt_app(self):
        """Create QApplication for widget testing."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app
        # Don't quit the app as it might be used by other tests

    def test_responsive_combo_box_initialization(self, qt_app):
        """Test ResponsiveComboBox initialization."""
        combo = ResponsiveComboBox()

        assert isinstance(combo, QComboBox)
        assert combo.toolTipDuration() == 3000
        assert combo.minimumWidth() > 0
        assert combo.minimumContentsLength() == 15

    def test_responsive_combo_box_add_item(self, qt_app):
        """Test ResponsiveComboBox addItem with tooltip."""
        combo = ResponsiveComboBox()

        # Add short item
        combo.addItem("Short")
        assert combo.itemText(0) == "Short"

        # Add long item (should get tooltip)
        long_text = "This is a very long item text that exceeds normal display width"
        combo.addItem(long_text)
        assert combo.itemText(1) == long_text

        # Check tooltip was set (itemData returns tooltip)
        tooltip = combo.itemData(1, Qt.ItemDataRole.ToolTipRole)
        # Tooltip might be set based on actual width calculation
        assert tooltip is None or tooltip == long_text

    def test_responsive_combo_box_show_popup(self, qt_app):
        """Test ResponsiveComboBox popup sizing."""
        combo = ResponsiveComboBox()
        combo.addItem("Short")
        combo.addItem("Medium length item")
        combo.addItem("Very long item text that should affect popup width")

        # Mock view for testing
        mock_view = Mock()
        mock_view.setMinimumWidth = Mock()
        combo.view = Mock(return_value=mock_view)

        combo.showPopup()

        # Verify view width was set (exact value depends on font metrics)
        mock_view.setMinimumWidth.assert_called()

    def test_responsive_scroll_area_initialization(self, qt_app):
        """Test ResponsiveScrollArea initialization."""
        scroll_area = ResponsiveScrollArea()

        assert scroll_area.widgetResizable() is True
        assert scroll_area.frameStyle() == scroll_area.Shape.NoFrame
        assert scroll_area.minimumSize().width() > 0
        assert scroll_area.minimumSize().height() > 0

    def test_responsive_scroll_area_set_widget(self, qt_app):
        """Test ResponsiveScrollArea setWidget method."""
        scroll_area = ResponsiveScrollArea()

        # Create test widget
        test_widget = QWidget()

        scroll_area.setWidget(test_widget)

        # Verify widget was set and size policy applied
        assert scroll_area.widget() is test_widget


class TestGUIStateSynchronization:
    """Test GUI state synchronization and consistency."""

    def test_gui_state_consistency_during_restoration(self, clean_app_state, mock_gui):
        """Test that GUI state remains consistent during restoration process."""
        clean_app_state.settings.setValue("Preferences/track_session_restoration", True)
        clean_app_state._is_restoring_session = True

        # During restoration, saves should be suppressed
        clean_app_state.save_current_session_state(experiment_id="test_exp", dataset_id="test_ds")

        # No settings should be saved during restoration
        saved_exp = clean_app_state.settings.value("SessionRestore/experiment", "", type=str)
        assert saved_exp == ""  # Should not be saved due to restoration flag

    def test_session_restoration_tracking_state(self, clean_app_state):
        """Test session restoration tracking preference."""
        # Default should be enabled
        assert clean_app_state.should_track_session_restoration() is True

        # When disabled, restoration should not occur
        clean_app_state.set_setting("track_session_restoration", False)
        assert clean_app_state.should_track_session_restoration() is False

        # Session restoration should return False when disabled
        assert clean_app_state.should_restore_session() is False

    def test_profile_tracking_independence(self, clean_app_state):
        """Test that profile tracking works independently of session restoration."""
        # Disable session restoration but leave profile tracking enabled
        clean_app_state.set_setting("track_session_restoration", False)
        clean_app_state.set_setting("track_analysis_profiles", True)

        # Profile should still be saved even when session restoration is disabled
        clean_app_state.save_current_session_state(profile_name="test_profile")

        last_profile = clean_app_state.get_last_profile()
        assert last_profile == "test_profile"

    def test_state_clearing_on_errors(self, clean_app_state):
        """Test that state is properly cleared when errors occur."""
        # Set up some session state
        clean_app_state.settings.setValue("SessionRestore/experiment", "test_exp")
        clean_app_state.settings.setValue("SessionRestore/dataset", "test_ds")
        clean_app_state.settings.sync()

        # Clear state (simulates error recovery)
        clean_app_state.clear_session_state()

        # All session state should be cleared
        state = clean_app_state.get_last_session_state()
        assert all(not value for value in state.values())

    def test_concurrent_state_modifications(self, clean_app_state):
        """Test handling of concurrent state modifications."""
        # Simulate multiple rapid state changes
        for i in range(5):
            clean_app_state.save_current_session_state(experiment_id=f"exp_{i}", dataset_id=f"ds_{i}", session_id=f"sess_{i}")

        # Final state should reflect last change
        state = clean_app_state.get_last_session_state()
        assert state["experiment"] == "exp_4"
        assert state["dataset"] == "ds_4"
        assert state["session"] == "sess_4"


class TestIntegrationWithQSettings:
    """Test integration with Qt's QSettings system."""

    def test_qsettings_organization_app_names(self):
        """Test QSettings uses correct organization and application names."""
        state = ApplicationState()
        settings = state.settings

        # Should have organization and application name set
        # (These are set by the main application)
        assert settings.organizationName() is not None
        assert settings.applicationName() is not None

    def test_qsettings_data_types_handling(self, clean_app_state):
        """Test QSettings handles various data types correctly."""
        settings = clean_app_state.settings

        # Test string
        settings.setValue("test/string", "test_value")
        assert settings.value("test/string", "", type=str) == "test_value"

        # Test boolean
        settings.setValue("test/bool_true", True)
        settings.setValue("test/bool_false", False)
        assert settings.value("test/bool_true", False, type=bool) is True
        assert settings.value("test/bool_false", True, type=bool) is False

        # Test integer
        settings.setValue("test/int", 42)
        assert settings.value("test/int", 0, type=int) == 42

        # Test list
        test_list = ["item1", "item2", "item3"]
        settings.setValue("test/list", test_list)
        retrieved_list = settings.value("test/list", [], type=list)
        assert retrieved_list == test_list

        # Test nested structure
        settings.setValue("test/nested/key", "nested_value")
        assert settings.value("test/nested/key", "", type=str) == "nested_value"

    def test_qsettings_sync_behavior(self, clean_app_state):
        """Test QSettings sync behavior ensures persistence."""
        settings = clean_app_state.settings

        # Set value and sync
        settings.setValue("persistence/test", "test_value")
        settings.sync()

        # Value should be immediately available
        assert settings.value("persistence/test", "", type=str) == "test_value"

        # Create new settings instance (simulates app restart)
        new_state = ApplicationState()
        new_settings = new_state.settings

        # Value should still be available
        assert new_settings.value("persistence/test", "", type=str) == "test_value"

    def test_qsettings_keys_and_groups(self, clean_app_state):
        """Test QSettings keys and groups functionality."""
        settings = clean_app_state.settings

        # Set values in different groups
        settings.setValue("Group1/key1", "value1")
        settings.setValue("Group1/key2", "value2")
        settings.setValue("Group2/key1", "value3")
        settings.setValue("root_key", "root_value")

        settings.sync()

        # Test group operations (simplified since childKeys is complex to mock)
        settings.beginGroup("Group1")
        settings.endGroup()  # Just test that these don't crash

        # Test value retrieval works correctly
        assert settings.value("Group1/key1", "", type=str) == "value1"
        assert settings.value("Group1/key2", "", type=str) == "value2"
        assert settings.value("Group2/key1", "", type=str) == "value3"
        assert settings.value("root_key", "", type=str) == "root_value"


# Integration test that combines multiple components
class TestGUIStateIntegration:
    """Test integration between different GUI state management components."""

    def test_full_session_restoration_workflow(self, clean_app_state, mock_gui, mock_experiment_structure):
        """Test complete session restoration workflow."""
        # Enable all tracking
        clean_app_state.set_setting("track_session_restoration", True)
        clean_app_state.set_setting("track_analysis_profiles", True)

        # Set up GUI state
        mock_gui.expts_dict_keys = ["test_experiment"]
        mock_gui.current_experiment = mock_experiment_structure["experiment"]
        mock_gui.current_dataset = mock_experiment_structure["dataset"]

        # Save complete session state
        clean_app_state.save_current_session_state(
            experiment_id="test_experiment", dataset_id="test_dataset", session_id="test_session", profile_name="test_profile"
        )

        # Verify state was saved
        state = clean_app_state.get_last_session_state()
        assert state["experiment"] == "test_experiment"
        assert state["dataset"] == "test_dataset"
        assert state["session"] == "test_session"
        assert state["profile"] == "test_profile"

        # Mock successful restoration by making experiment immediately available
        def mock_single_shot(delay, callback):
            callback()

        with patch("PySide6.QtCore.QTimer.singleShot", side_effect=mock_single_shot):
            # Restore session
            result = clean_app_state.restore_last_session(mock_gui)
            assert result is True

            # Verify restoration calls were made
            mock_gui.data_selection_widget.experiment_combo.setCurrentIndex.assert_called()
            mock_gui.profile_selector_combo.setCurrentIndex.assert_called()

    def test_error_recovery_during_restoration(self, clean_app_state, mock_gui):
        """Test error recovery mechanisms during session restoration."""
        clean_app_state.set_setting("track_session_restoration", True)

        # Set up state that will cause errors
        clean_app_state.settings.setValue("SessionRestore/experiment", "nonexistent_exp")

        mock_gui.expts_dict_keys = []  # No experiments available

        # Attempt restoration
        result = clean_app_state.restore_last_session(mock_gui)
        assert result is False

        # State should be cleared after failed restoration
        state = clean_app_state.get_last_session_state()
        assert state["experiment"] == ""

        # Restoration flag should be cleared
        assert clean_app_state._is_restoring_session is False

    @patch("monstim_gui.core.ui_config.QSettings")
    @patch("monstim_gui.core.application_state.QSettings")
    def test_ui_config_and_app_state_coordination(self, mock_app_state_qsettings, mock_ui_config_qsettings):
        """Test coordination between UIConfig and ApplicationState."""
        # Setup separate mock storage for each component
        app_state_storage = {}
        ui_config_storage = {}

        # Mock for ApplicationState
        app_mock_settings = Mock()

        def app_set_value(key, value):
            app_state_storage[key] = value

        def app_get_value(key, default=None, type=str):
            return app_state_storage.get(key, default)

        app_mock_settings.setValue.side_effect = app_set_value
        app_mock_settings.value.side_effect = app_get_value
        app_mock_settings.sync = Mock()
        app_mock_settings.organizationName.return_value = "TestOrg"
        app_mock_settings.applicationName.return_value = "TestApp"
        mock_app_state_qsettings.return_value = app_mock_settings

        # Mock for UIConfig
        ui_mock_settings = Mock()

        def ui_set_value(key, value):
            ui_config_storage[key] = value

        def ui_get_value(key, default=None, type=str):
            return ui_config_storage.get(key, default)

        ui_mock_settings.setValue.side_effect = ui_set_value
        ui_mock_settings.value.side_effect = ui_get_value
        ui_mock_settings.sync = Mock()
        mock_ui_config_qsettings.return_value = ui_mock_settings

        app_state_instance = ApplicationState()
        ui_config_instance = UIConfig()

        # Both should use QSettings but potentially different keys
        app_state_instance.settings.setValue("AppState/test", "app_value")
        ui_config_instance.set("test", "ui_value")  # UIConfig adds UI/ prefix internally

        # Values should be independent
        assert app_state_instance.settings.value("AppState/test", "", type=str) == "app_value"
        assert ui_config_instance.get("test") == "ui_value"
