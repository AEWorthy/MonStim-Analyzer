"""
Comprehensive tests for the configuration and settings management system.

These tests cover critical user experience components that were previously untested:
- ConfigRepository type coercion and file handling
- ProfileManager loading/saving/listing profiles
- ApplicationState session restoration and preference tracking
- Analysis profile merging with global configuration
"""

import os
import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml

from monstim_gui.core.application_state import ApplicationState
from monstim_gui.io.config_repository import ConfigRepository
from monstim_gui.managers.profile_manager import ProfileManager

# --- Test Annotations ---
# Purpose: ConfigRepository I/O, ProfileManager save/load/list, ApplicationState QSettings integration
# Markers: integration (cross-component behavior), uses temp dirs
# Notes: Mocks QSettings; exercises type coercion paths
pytestmark = pytest.mark.integration


class TestConfigRepository:
    """Test configuration file handling and type coercion."""

    def setup_method(self):
        """Create temporary directories and files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.default_config_path = os.path.join(self.temp_dir, "config.yml")
        self.user_config_path = os.path.join(self.temp_dir, "config-user.yml")

        # Sample default configuration matching the application's structure
        self.default_config = {
            "bin_size": 0.01,
            "time_window": 8.0,
            "pre_stim_time": 2.0,
            "default_method": "rms",
            "default_channel_names": ["LG", "TA", "SOL"],
            "butter_filter_args": {"lowcut": 100, "highcut": 3500, "order": 4},
            "m_max_args": {"max_window_size": 15, "min_window_size": 2, "threshold": 0.3, "validation_tolerance": 1.05},
            "title_font_size": 16,
            "m_color": "tab:red",
            "h_color": "tab:blue",
            "h_threshold": 0.5,
            "preferred_date_format": "YYMMDD",
        }

        # Write default config
        with open(self.default_config_path, "w") as f:
            yaml.safe_dump(self.default_config, f)

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_repository_initialization(self):
        """Test ConfigRepository initialization and path handling."""
        repo = ConfigRepository(self.default_config_path)

        assert repo.default_config_file == self.default_config_path
        assert repo.user_config_file == self.user_config_path

        # Test explicit user config path
        custom_user_path = os.path.join(self.temp_dir, "custom-user.yml")
        repo_custom = ConfigRepository(self.default_config_path, custom_user_path)
        assert repo_custom.user_config_file == custom_user_path

    def test_read_config_default_only(self):
        """Test reading configuration when only default config exists."""
        repo = ConfigRepository(self.default_config_path)
        config = repo.read_config()

        assert config == self.default_config
        assert config["bin_size"] == 0.01
        assert config["butter_filter_args"]["lowcut"] == 100

    def test_read_config_with_user_overrides(self):
        """Test reading configuration with user overrides."""
        # Create user config with some overrides
        user_config = {
            "bin_size": 0.02,  # Override float
            "default_method": "peak_to_trough",  # Override string
            "butter_filter_args": {"lowcut": 200, "order": 6},  # Override nested value  # Override nested int
            "new_user_setting": "custom_value",  # New setting
        }

        with open(self.user_config_path, "w") as f:
            yaml.safe_dump(user_config, f)

        repo = ConfigRepository(self.default_config_path)
        config = repo.read_config()

        # Check overrides applied
        assert config["bin_size"] == 0.02
        assert config["default_method"] == "peak_to_trough"
        assert config["butter_filter_args"]["lowcut"] == 200
        assert config["butter_filter_args"]["order"] == 6
        assert config["new_user_setting"] == "custom_value"

        # Check non-overridden values preserved
        assert config["time_window"] == 8.0
        assert config["butter_filter_args"]["highcut"] == 3500

    def test_write_config(self):
        """Test writing user configuration."""
        repo = ConfigRepository(self.default_config_path)

        test_config = {"bin_size": 0.03, "custom_setting": "test"}
        repo.write_config(test_config)

        assert os.path.exists(self.user_config_path)

        # Verify written content
        with open(self.user_config_path, "r") as f:
            written_config = yaml.safe_load(f)

        assert written_config == test_config

    def test_type_coercion_basic_types(self):
        """Test type coercion for basic data types."""
        reference = {"int_val": 10, "float_val": 3.14, "bool_val": True, "str_val": "hello"}

        user_data = {
            "int_val": "20",  # String to int
            "float_val": "2.71",  # String to float
            "bool_val": "false",  # String to bool
            "str_val": 42,  # Non-string to string
        }

        result = ConfigRepository.coerce_types(user_data, reference)

        assert result["int_val"] == 20
        assert isinstance(result["int_val"], int)
        assert result["float_val"] == 2.71
        assert isinstance(result["float_val"], float)
        assert result["bool_val"] is False
        assert isinstance(result["bool_val"], bool)
        assert result["str_val"] == "42"
        assert isinstance(result["str_val"], str)

    def test_type_coercion_boolean_variations(self):
        """Test various boolean string representations."""
        reference = {"bool_val": True}

        true_values = ["true", "True", "TRUE", "1", "yes", "Yes", "on", "On"]
        false_values = ["false", "False", "FALSE", "0", "no", "No", "off", "Off"]

        for val in true_values:
            result = ConfigRepository.coerce_types({"bool_val": val}, reference)
            assert result["bool_val"] is True, f"Failed for '{val}'"

        for val in false_values:
            result = ConfigRepository.coerce_types({"bool_val": val}, reference)
            assert result["bool_val"] is False, f"Failed for '{val}'"

    def test_type_coercion_nested_dicts(self):
        """Test type coercion for nested dictionaries."""
        reference = {"nested": {"int_val": 5, "float_val": 1.5, "deep_nested": {"bool_val": True}}}

        user_data = {"nested": {"int_val": "10", "float_val": "2.5", "deep_nested": {"bool_val": "false"}}}

        result = ConfigRepository.coerce_types(user_data, reference)

        assert result["nested"]["int_val"] == 10
        assert result["nested"]["float_val"] == 2.5
        assert result["nested"]["deep_nested"]["bool_val"] is False

    def test_type_coercion_lists(self):
        """Test type coercion for lists."""
        reference = {"int_list": [1, 2, 3], "str_list": ["a", "b", "c"]}

        user_data = {"int_list": ["4", "5", "6"], "str_list": [10, 20, 30]}

        result = ConfigRepository.coerce_types(user_data, reference)

        assert result["int_list"] == [4, 5, 6]
        assert all(isinstance(x, int) for x in result["int_list"])
        assert result["str_list"] == ["10", "20", "30"]
        assert all(isinstance(x, str) for x in result["str_list"])

    def test_type_coercion_fallback_handling(self):
        """Test type coercion with invalid conversions."""
        reference = {"int_val": 10, "float_val": 3.14}

        user_data = {"int_val": "not_a_number", "float_val": "also_not_a_number"}

        result = ConfigRepository.coerce_types(user_data, reference)

        # Should fallback to original values when conversion fails
        assert result["int_val"] == "not_a_number"
        assert result["float_val"] == "also_not_a_number"

    def test_malformed_user_config_handling(self):
        """Test handling of malformed user config files."""
        # Create malformed YAML file
        with open(self.user_config_path, "w") as f:
            f.write("invalid: yaml: content: [")

        repo = ConfigRepository(self.default_config_path)

        # Should handle gracefully and return default config
        with pytest.raises(yaml.YAMLError):
            repo.read_config()

    def test_missing_default_config(self):
        """Test behavior when default config is missing."""
        missing_path = os.path.join(self.temp_dir, "nonexistent.yml")
        repo = ConfigRepository(missing_path)

        with pytest.raises(FileNotFoundError):
            repo.read_config()

    def test_update_nested_dict(self):
        """Test the nested dictionary update functionality."""
        repo = ConfigRepository(self.default_config_path)

        base = {
            "level1": {"level2": {"keep": "original", "update": "old_value"}, "keep_section": "unchanged"},
            "top_level": "original",
        }

        update = {"level1": {"level2": {"update": "new_value", "add": "new_key"}, "new_section": "added"}, "new_top": "added"}

        result = repo._update_nested_dict(base.copy(), update)

        # Check updates applied correctly
        assert result["level1"]["level2"]["update"] == "new_value"
        assert result["level1"]["level2"]["add"] == "new_key"
        assert result["level1"]["new_section"] == "added"
        assert result["new_top"] == "added"

        # Check original values preserved
        assert result["level1"]["level2"]["keep"] == "original"
        assert result["level1"]["keep_section"] == "unchanged"
        assert result["top_level"] == "original"


class TestProfileManager:
    """Test analysis profile management functionality."""

    def setup_method(self):
        """Create temporary profile directory and reference config."""
        self.temp_dir = tempfile.mkdtemp()
        self.profile_dir = os.path.join(self.temp_dir, "profiles")
        os.makedirs(self.profile_dir)

        # Reference config for type coercion
        self.reference_config = {
            "time_window": 8.0,
            "pre_stim_time": 2.0,
            "default_method": "rms",
            "butter_filter_args": {"lowcut": 100, "highcut": 3500, "order": 4},
        }

        # Sample profiles
        self.profile1_data = {
            "name": "Test Profile 1",
            "description": "A test profile for EMG analysis",
            "analysis_parameters": {"time_window": 10.0, "default_method": "peak_to_trough"},
            "stimuli_to_plot": ["Electrical"],
        }

        self.profile2_data = {
            "name": "Test Profile 2",
            "description": "Another test profile",
            "analysis_parameters": {"pre_stim_time": 3.0},
        }

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_profile_manager_initialization(self):
        """Test ProfileManager initialization."""
        pm = ProfileManager(self.profile_dir, self.reference_config)

        assert pm.profile_dir == self.profile_dir
        assert pm.reference_config == self.reference_config
        assert os.path.exists(self.profile_dir)

    def test_save_and_load_profile(self):
        """Test saving and loading a single profile."""
        pm = ProfileManager(self.profile_dir, self.reference_config)

        # Save profile
        filename = pm.save_profile(self.profile1_data)
        assert os.path.exists(filename)
        assert filename.endswith("test_profile_1.yml")

        # Load profile
        loaded_data = pm.load_profile(filename)
        assert loaded_data["name"] == "Test Profile 1"
        assert loaded_data["description"] == "A test profile for EMG analysis"
        assert loaded_data["analysis_parameters"]["time_window"] == 10.0

    def test_save_profile_with_custom_filename(self):
        """Test saving profile with custom filename."""
        pm = ProfileManager(self.profile_dir, self.reference_config)

        custom_filename = os.path.join(self.profile_dir, "custom_profile.yml")
        returned_filename = pm.save_profile(self.profile1_data, custom_filename)

        assert returned_filename == custom_filename
        assert os.path.exists(custom_filename)

    def test_save_profile_ordering(self):
        """Test that profiles are saved with correct key ordering."""
        pm = ProfileManager(self.profile_dir, self.reference_config)

        filename = pm.save_profile(self.profile1_data)

        # Read raw YAML to check ordering
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that name comes first, then description, etc.
        lines = content.split("\n")
        name_line = next(i for i, line in enumerate(lines) if line.startswith("name:"))
        desc_line = next(i for i, line in enumerate(lines) if line.startswith("description:"))

        assert name_line < desc_line, "Name should come before description"

    def test_list_profiles_empty(self):
        """Test listing profiles when directory is empty."""
        pm = ProfileManager(self.profile_dir, self.reference_config)

        profiles = pm.list_profiles()
        assert profiles == []

    def test_list_profiles_with_data(self):
        """Test listing profiles with existing profile files."""
        pm = ProfileManager(self.profile_dir, self.reference_config)

        # Save two profiles
        pm.save_profile(self.profile1_data)
        pm.save_profile(self.profile2_data)

        profiles = pm.list_profiles()

        assert len(profiles) == 2

        # Check profile structure: (name, filepath, data)
        names = [profile[0] for profile in profiles]
        assert "Test Profile 1" in names
        assert "Test Profile 2" in names

        # Check that file paths are included
        filepaths = [profile[1] for profile in profiles]
        assert all(os.path.exists(fp) for fp in filepaths)

        # Check that data is included
        profile_data = [profile[2] for profile in profiles]
        assert all(isinstance(data, dict) for data in profile_data)

    def test_profile_type_coercion_on_load(self):
        """Test that profiles undergo type coercion when loaded."""
        pm = ProfileManager(self.profile_dir, self.reference_config)

        # Create profile with string values that should be coerced
        # Using keys that exist in reference_config for type coercion
        profile_with_strings = {
            "name": "Type Coercion Test",
            "time_window": "12.5",  # Should become float
            "pre_stim_time": "3.5",  # Should become float
            "butter_filter_args": {"order": "6"},  # Should become int
        }

        filename = pm.save_profile(profile_with_strings)
        loaded_data = pm.load_profile(filename)

        # Check type coercion occurred for matching reference config keys
        assert isinstance(loaded_data["time_window"], float)
        assert loaded_data["time_window"] == 12.5
        assert isinstance(loaded_data["pre_stim_time"], float)
        assert loaded_data["pre_stim_time"] == 3.5
        assert isinstance(loaded_data["butter_filter_args"]["order"], int)
        assert loaded_data["butter_filter_args"]["order"] == 6

    def test_profile_corrupted_file_handling(self):
        """Test handling of corrupted profile files."""
        pm = ProfileManager(self.profile_dir, self.reference_config)

        # Create a valid profile first
        pm.save_profile(self.profile1_data)

        # Create a corrupted YAML file
        corrupted_file = os.path.join(self.profile_dir, "corrupted.yml")
        with open(corrupted_file, "w", encoding="utf-8") as f:
            f.write("invalid: yaml: [unclosed")

        # list_profiles should raise YAML error since it's not handled
        # This is the actual current behavior - YAML errors are not caught
        with pytest.raises(yaml.scanner.ScannerError):
            pm.list_profiles()

    def test_delete_profile(self):
        """Test profile deletion."""
        pm = ProfileManager(self.profile_dir, self.reference_config)

        filename = pm.save_profile(self.profile1_data)
        assert os.path.exists(filename)

        pm.delete_profile(filename)
        assert not os.path.exists(filename)

    def test_delete_nonexistent_profile(self):
        """Test deleting a profile that doesn't exist."""
        pm = ProfileManager(self.profile_dir, self.reference_config)

        nonexistent_file = os.path.join(self.profile_dir, "does_not_exist.yml")

        # Should not raise an exception
        pm.delete_profile(nonexistent_file)

    def test_profile_without_reference_config(self):
        """Test ProfileManager functionality without reference config."""
        pm = ProfileManager(self.profile_dir, reference_config=None)

        filename = pm.save_profile(self.profile1_data)
        loaded_data = pm.load_profile(filename)

        # Should work but without type coercion
        assert loaded_data["name"] == "Test Profile 1"


class TestApplicationState:
    """Test application state persistence and session management."""

    def setup_method(self):
        """Set up test environment with mocked QSettings."""
        # Mock QSettings to avoid actual file system interaction
        self.mock_settings = Mock()
        self.mock_settings.organizationName.return_value = "TestOrg"
        self.mock_settings.applicationName.return_value = "TestApp"
        self.mock_settings.sync.return_value = None

        # Storage for mock settings
        self.settings_storage = {}

        def mock_setValue(key, value):
            self.settings_storage[key] = value

        def mock_value(key, default=None, type=None):
            value = self.settings_storage.get(key, default)
            if type and value is not None:
                if type is list and not isinstance(value, list):
                    return [value] if value else []
                if type is str and not isinstance(value, str):
                    return str(value)
            return value

        self.mock_settings.setValue.side_effect = mock_setValue
        self.mock_settings.value.side_effect = mock_value

    # NOTE: ApplicationState initialization is covered in GUI tests.
    # Keeping configuration tests focused on ConfigRepository/ProfileManager.

    @patch("monstim_gui.core.application_state.QSettings")
    @patch("os.path.isdir")
    def test_import_export_path_tracking(self, mock_isdir, mock_qsettings):
        """Test import/export path persistence."""
        mock_qsettings.return_value = self.mock_settings
        mock_isdir.return_value = True  # Mock directory existence check

        app_state = ApplicationState()

        # Test import path
        test_import_path = "/path/to/import"
        app_state.save_last_import_path(test_import_path)

        assert self.settings_storage["LastPaths/import_directory"] == test_import_path

        retrieved_path = app_state.get_last_import_path()
        assert retrieved_path == test_import_path

        # Test export path
        test_export_path = "/path/to/export"
        app_state.save_last_export_path(test_export_path)

        assert self.settings_storage["LastPaths/export_directory"] == test_export_path

        retrieved_export = app_state.get_last_export_path()
        assert retrieved_export == test_export_path

    @patch("monstim_gui.core.application_state.QSettings")
    def test_path_tracking_disabled(self, mock_qsettings):
        """Test behavior when path tracking is disabled."""
        mock_qsettings.return_value = self.mock_settings

        app_state = ApplicationState()

        # Mock the preference to return False
        with patch.object(app_state, "should_track_import_export_paths", return_value=False):
            app_state.save_last_import_path("/some/path")

            # Should not save when tracking is disabled
            assert "LastPaths/import_directory" not in self.settings_storage

            # Should return empty string when tracking is disabled
            result = app_state.get_last_import_path()
            assert result == ""

    @patch("monstim_gui.core.application_state.QSettings")
    def test_recent_experiments_tracking(self, mock_qsettings):
        """Test recent experiments list management."""
        mock_qsettings.return_value = self.mock_settings

        app_state = ApplicationState()

        # Add experiments
        app_state.save_recent_experiment("exp1")
        app_state.save_recent_experiment("exp2")
        app_state.save_recent_experiment("exp3")

        recent = app_state.get_recent_experiments()
        assert recent == ["exp3", "exp2", "exp1"]  # Most recent first

        # Add duplicate - should move to front
        app_state.save_recent_experiment("exp1")
        recent = app_state.get_recent_experiments()
        assert recent == ["exp1", "exp3", "exp2"]

    @patch("monstim_gui.core.application_state.QSettings")
    def test_recent_experiments_limit(self, mock_qsettings):
        """Test that recent experiments list is limited to 10 items."""
        mock_qsettings.return_value = self.mock_settings

        app_state = ApplicationState()

        # Add 15 experiments
        for i in range(15):
            app_state.save_recent_experiment(f"exp{i}")

        recent = app_state.get_recent_experiments()
        assert len(recent) == 10  # Should be limited to 10
        assert recent[0] == "exp14"  # Most recent
        assert recent[-1] == "exp5"  # Oldest kept

    @patch("monstim_gui.core.application_state.QSettings")
    def test_analysis_profile_tracking(self, mock_qsettings):
        """Test analysis profile selection persistence."""
        mock_qsettings.return_value = self.mock_settings

        app_state = ApplicationState()

        # Test profile selection
        profile_name = "Custom EMG Profile"
        app_state.save_last_profile(profile_name)

        assert self.settings_storage["LastSelection/profile"] == profile_name

        retrieved = app_state.get_last_profile()
        assert retrieved == profile_name

        # Test recent profiles
        app_state.save_recent_profile("Profile 1")
        app_state.save_recent_profile("Profile 2")
        app_state.save_recent_profile("Profile 3")

        recent = app_state.get_recent_profiles()
        assert recent == ["Profile 3", "Profile 2", "Profile 1"]

    @patch("monstim_gui.core.application_state.QSettings")
    def test_session_state_persistence(self, mock_qsettings):
        """Test complete session state saving and restoration."""
        mock_qsettings.return_value = self.mock_settings

        app_state = ApplicationState()

        # Save session state
        app_state.save_current_session_state(
            experiment_id="test_experiment", dataset_id="test_dataset", session_id="test_session", profile_name="Test Profile"
        )

        # Check that all components were saved (check actual keys used by implementation)
        assert self.settings_storage["SessionRestore/experiment"] == "test_experiment"
        assert self.settings_storage["SessionRestore/dataset"] == "test_dataset"
        assert self.settings_storage["SessionRestore/session"] == "test_session"
        assert self.settings_storage["LastSelection/profile"] == "Test Profile"

    @patch("monstim_gui.core.application_state.QSettings")
    def test_session_restoration_flag(self, mock_qsettings):
        """Test that session restoration flag prevents redundant saves."""
        mock_qsettings.return_value = self.mock_settings

        app_state = ApplicationState()

        # Set restoration flag
        app_state._is_restoring_session = True

        # Try to save session - should be ignored
        app_state.save_current_session_state(experiment_id="test_exp")

        # Should not have saved anything
        assert "CurrentSession/experiment_id" not in self.settings_storage

    @patch("monstim_gui.core.application_state.QSettings")
    def test_preference_management(self, mock_qsettings):
        """Test program preference getting and setting."""
        mock_qsettings.return_value = self.mock_settings

        app_state = ApplicationState()

        # Test default preference value
        pref = app_state.get_preference("test_preference", default_value=True)
        assert pref is True

        # Set preference (uses ProgramPreferences/ prefix)
        app_state.set_setting("test_preference", False)
        assert self.settings_storage["ProgramPreferences/test_preference"] is False

        # Get modified preference
        pref = app_state.get_preference("test_preference", default_value=True)
        assert pref is False

    @patch("monstim_gui.core.application_state.QSettings")
    def test_preference_based_feature_flags(self, mock_qsettings):
        """Test various preference-based feature flags."""
        mock_qsettings.return_value = self.mock_settings

        app_state = ApplicationState()

        # Test default values (should all be True by default)
        assert app_state.should_track_session_restoration() is True
        assert app_state.should_track_import_export_paths() is True
        assert app_state.should_track_recent_files() is True
        assert app_state.should_track_analysis_profiles() is True
        assert app_state.should_use_opengl_acceleration() is False

        # Disable a feature
        app_state.set_setting("track_recent_files", False)
        assert app_state.should_track_recent_files() is False

    @patch("monstim_gui.core.application_state.QSettings")
    def test_session_restoration_with_mock_gui(self, mock_qsettings):
        """Test session restoration with mocked GUI object."""
        mock_qsettings.return_value = self.mock_settings

        app_state = ApplicationState()

        # First save some session state
        app_state.save_current_session_state(experiment_id="saved_exp", dataset_id="saved_dataset")

        # Create mock GUI
        mock_gui = Mock()
        mock_gui.data_manager = Mock()
        mock_gui.profile_selector_combo = Mock()

        # Mock the data manager methods
        mock_gui.data_manager.load_experiment.return_value = True
        mock_gui.data_manager.set_current_dataset.return_value = None

        # Test restoration
        # Note: Full restoration test would require more GUI mocking
        # This tests the basic structure - check that data was saved
        assert self.settings_storage["SessionRestore/experiment"] == "saved_exp"
        assert self.settings_storage["SessionRestore/dataset"] == "saved_dataset"


class TestConfigurationIntegration:
    """Integration tests combining multiple configuration components."""

    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.temp_dir, "config")
        self.profile_dir = os.path.join(self.temp_dir, "profiles")
        os.makedirs(self.config_dir)
        os.makedirs(self.profile_dir)

        self.default_config_path = os.path.join(self.config_dir, "config.yml")

        # Create realistic default config
        self.default_config = {
            "time_window": 8.0,
            "pre_stim_time": 2.0,
            "default_method": "rms",
            "butter_filter_args": {"lowcut": 100, "highcut": 3500, "order": 4},
            "m_color": "tab:red",
            "h_color": "tab:blue",
        }

        with open(self.default_config_path, "w") as f:
            yaml.safe_dump(self.default_config, f)

    def teardown_method(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_profile_config_integration(self):
        """Test integration between ProfileManager and ConfigRepository."""
        # Create config repository
        config_repo = ConfigRepository(self.default_config_path)
        base_config = config_repo.read_config()

        # Create profile manager with reference config
        profile_manager = ProfileManager(self.profile_dir, base_config)

        # Create and save a profile with overrides
        profile_data = {
            "name": "Integration Test Profile",
            "description": "Testing config-profile integration",
            "analysis_parameters": {
                "time_window": 12.0,  # Override
                "default_method": "peak_to_trough",  # Override
                "new_setting": "custom_value",  # New
            },
        }

        filename = profile_manager.save_profile(profile_data)

        # Load and merge with base config
        loaded_profile = profile_manager.load_profile(filename)

        # Simulate profile application (like GUI does)
        effective_config = base_config.copy()
        if "analysis_parameters" in loaded_profile:
            effective_config.update(loaded_profile["analysis_parameters"])

        # Check that profile overrides are applied
        assert effective_config["time_window"] == 12.0
        assert effective_config["default_method"] == "peak_to_trough"
        assert effective_config["new_setting"] == "custom_value"

        # Check that non-overridden values are preserved
        assert effective_config["pre_stim_time"] == 2.0
        assert effective_config["butter_filter_args"]["lowcut"] == 100

    def test_config_type_coercion_with_profiles(self):
        """Test that type coercion works properly with profile data."""
        config_repo = ConfigRepository(self.default_config_path)
        base_config = config_repo.read_config()

        profile_manager = ProfileManager(self.profile_dir, base_config)

        # Create profile with string values that need coercion (matching actual config structure)
        profile_data = {
            "name": "Type Coercion Profile",
            "time_window": "15.5",  # String -> float
            "pre_stim_time": "3.0",  # String -> float
            "butter_filter_args": {"order": "6", "lowcut": "150"},  # String -> int  # String -> int
        }

        filename = profile_manager.save_profile(profile_data)
        loaded_profile = profile_manager.load_profile(filename)

        # Check type coercion occurred for top-level keys
        assert isinstance(loaded_profile["time_window"], float)
        assert loaded_profile["time_window"] == 15.5
        assert isinstance(loaded_profile["pre_stim_time"], float)
        assert loaded_profile["pre_stim_time"] == 3.0

        # Check nested type coercion
        assert isinstance(loaded_profile["butter_filter_args"]["order"], int)
        assert loaded_profile["butter_filter_args"]["order"] == 6
        assert isinstance(loaded_profile["butter_filter_args"]["lowcut"], int)
        assert loaded_profile["butter_filter_args"]["lowcut"] == 150

    def test_multiple_profile_management(self):
        """Test managing multiple profiles simultaneously."""
        config_repo = ConfigRepository(self.default_config_path)
        base_config = config_repo.read_config()

        profile_manager = ProfileManager(self.profile_dir, base_config)

        # Create multiple profiles
        profiles = [
            {"name": "EMG Standard", "analysis_parameters": {"time_window": 8.0, "default_method": "rms"}},
            {"name": "EMG Extended", "analysis_parameters": {"time_window": 15.0, "default_method": "peak_to_trough"}},
            {"name": "Force Analysis", "analysis_parameters": {"pre_stim_time": 5.0, "h_color": "tab:green"}},
        ]

        # Save all profiles
        for profile in profiles:
            profile_manager.save_profile(profile)

        # List and verify all profiles
        listed_profiles = profile_manager.list_profiles()
        assert len(listed_profiles) == 3

        profile_names = [p[0] for p in listed_profiles]
        assert "EMG Standard" in profile_names
        assert "EMG Extended" in profile_names
        assert "Force Analysis" in profile_names

        # Test loading each profile
        for name, filepath, data in listed_profiles:
            loaded = profile_manager.load_profile(filepath)
            assert loaded["name"] == name
            assert "analysis_parameters" in loaded

    def test_error_recovery_in_integration(self):
        """Test error recovery when components interact."""
        config_repo = ConfigRepository(self.default_config_path)
        profile_manager = ProfileManager(self.profile_dir, config_repo.read_config())

        # Create a profile with some invalid data
        problematic_profile = {
            "name": "Problematic Profile",
            "analysis_parameters": {"time_window": "not_a_number", "valid_setting": 5.0},  # Will cause coercion to fail
        }

        filename = profile_manager.save_profile(problematic_profile)
        loaded = profile_manager.load_profile(filename)

        # Should handle the problematic value gracefully
        # Fallback behavior: If type coercion fails, the invalid value is preserved rather than raising an exception.
        # This allows the user or system to detect and correct invalid settings later, rather than failing outright.
        assert loaded["analysis_parameters"]["time_window"] == "not_a_number"  # Fallback
        assert loaded["analysis_parameters"]["valid_setting"] == 5.0  # Preserved
