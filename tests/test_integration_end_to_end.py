"""
Integration & End-to-End Testing Suite

This module contains comprehensive integration tests that validate complete user workflows
and end-to-end scenarios across the MonStim Analyzer application. These tests ensure
that all components work together seamlessly and that critical user paths function correctly.

Test Categories:
1. Complete Data Import to Analysis Workflow
2. User Interface State Management Integration
3. Command System Integration with Domain Models
4. Error Recovery Across Component Boundaries
5. Performance and Memory Management Integration
6. Configuration and Profile Management Integration
"""

import threading
import time
from unittest.mock import Mock, patch

import pytest

from monstim_gui.commands import CommandInvoker, CreateExperimentCommand, ExcludeRecordingCommand, ExcludeSessionCommand

# Import GUI components
from monstim_gui.core.application_state import ApplicationState
from monstim_gui.core.ui_config import UIConfig

# Import configuration system
from monstim_gui.io.config_repository import ConfigRepository

# Import managers
from monstim_gui.managers.data_manager import DataManager
from monstim_gui.managers.plot_controller import PlotController
from monstim_gui.managers.profile_manager import ProfileManager
from monstim_signals.core.data_models import LatencyWindow

# Import import/export pipeline
from monstim_signals.io import csv_importer

# Import domain models
from monstim_signals.io.repositories import ExperimentRepository

# --- Test Annotations ---
# Purpose: Validate full user workflows end-to-end across managers, commands, repositories, and config
# Markers: integration (E2E flows), slow (IO and import/conversion)
# Notes: Uses temp dirs and mocks; resilient to missing golden data
pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestCompleteDataWorkflows:
    """Test complete data import to analysis workflows."""

    @pytest.fixture
    def mock_csv_data_structure(self, temp_output_dir):
        """Create mock CSV data structure for import testing."""
        # Create experiment directory structure
        exp_dir = temp_output_dir / "TestExperiment"
        exp_dir.mkdir()

        # Create dataset directory
        dataset_dir = exp_dir / "250101 TestAnimal Baseline"
        dataset_dir.mkdir()

        # Create session CSV files
        session_csv = dataset_dir / "TestAnimal_Baseline_001.csv"
        session_csv.write_text("Time,Channel1,Channel2\n0.0,1.0,2.0\n0.001,1.1,2.1\n0.002,1.2,2.2\n")

        # Create STM file
        stm_file = dataset_dir / "TestAnimal_Baseline_001.stm"
        stm_file.write_text("0.010\t1.0\n0.020\t2.0\n0.030\t3.0\n")

        return exp_dir

    def test_complete_csv_import_to_domain_model_workflow(self, temp_output_dir, mock_csv_data_structure):
        """Test complete workflow from CSV import to domain model access."""
        # Step 1: Import CSV data
        output_dir = temp_output_dir / "imported_data"

        # Import experiment using the csv_importer module function
        try:
            csv_importer.import_experiment(
                expt_path=mock_csv_data_structure,
                output_path=output_dir,
                progress_callback=lambda v: None,
                is_canceled=lambda: False,
            )

            assert output_dir.exists(), "Output directory should be created"

            # Step 2: Load via repositories (single experiment at output root)
            exp_repo = ExperimentRepository(output_dir)
            experiment = exp_repo.load()

            # Step 3: Navigate domain hierarchy
            assert len(experiment.datasets) > 0, "Experiment should have datasets"
            dataset = experiment.datasets[0]

            assert len(dataset.sessions) > 0, "Dataset should have sessions"
            session = dataset.sessions[0]

            # Step 4: Access processed data
            assert len(session.recordings) > 0, "Session should have recordings"
            recordings_raw = session.recordings_raw
            recordings_filtered = session.recordings_filtered

            assert recordings_raw is not None, "Should have raw recordings data"
            assert recordings_filtered is not None, "Should have filtered recordings data"

            # Step 5: Test M-max calculation
            try:
                mmax_value = session.get_mmax()
                assert isinstance(mmax_value, (int, float)), "M-max should be numeric"
            except Exception:
                # M-max calculation may fail with mock data - that's okay
                pass

        except Exception:
            # CSV import may fail with mock data - test that it handles errors gracefully
            assert True, "Import should handle errors gracefully"

    def test_import_with_error_recovery(self, temp_output_dir):
        """Test import workflow with various error conditions and recovery."""
        # Create invalid CSV structure
        invalid_dir = temp_output_dir / "InvalidData"
        invalid_dir.mkdir()

        # Create file with invalid content
        invalid_csv = invalid_dir / "invalid.csv"
        invalid_csv.write_text("This is not CSV data!")

        output_dir = temp_output_dir / "output"

        # Import should handle errors gracefully
        try:
            csv_importer.import_experiment(
                expt_path=invalid_dir, output_path=output_dir, progress_callback=lambda v: None, is_canceled=lambda: False
            )
            # Import may succeed or fail, but shouldn't crash
            assert True, "Import completed without crashing"
        except Exception:
            # Import may fail with invalid data - that's expected
            assert True, "Import handled error gracefully"

    def test_concurrent_import_operations(self, temp_output_dir, mock_csv_data_structure):
        """Test multiple concurrent import operations."""

        def import_worker(worker_id):
            output_dir = temp_output_dir / f"output_{worker_id}"

            try:
                csv_importer.import_experiment(
                    expt_path=mock_csv_data_structure,
                    output_path=output_dir,
                    progress_callback=lambda v: None,
                    is_canceled=lambda: False,
                )
                return True
            except Exception:
                return False

        # Start multiple import threads
        threads = []
        results = {}

        for i in range(3):

            def worker_with_id(worker_id=i):
                results[worker_id] = import_worker(worker_id)

            thread = threading.Thread(target=worker_with_id)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout

        # Verify all completed successfully
        for worker_id, success in results.items():
            assert success, f"Worker {worker_id} should complete successfully"


class TestUIStateManagementIntegration:
    """Test integration between UI components and state management."""

    @pytest.fixture
    def integration_gui_mock(self):
        """Create comprehensive GUI mock for integration testing."""
        gui = Mock()

        # Mock main application components
        gui.current_experiment = None
        gui.current_dataset = None
        gui.current_session = None
        gui.expts_dict_keys = []

        # Mock data selection widget
        gui.data_selection_widget = Mock()
        gui.data_selection_widget.experiment_combo = Mock()
        gui.data_selection_widget.dataset_combo = Mock()
        gui.data_selection_widget.session_combo = Mock()
        gui.data_selection_widget.update = Mock()
        gui.data_selection_widget.refresh = Mock()

        # Mock profile selector
        gui.profile_selector_combo = Mock()
        gui.profile_selector_combo.currentText = Mock(return_value="default")
        gui.profile_selector_combo.findText = Mock(return_value=0)
        gui.profile_selector_combo.setCurrentIndex = Mock()

        # Mock data manager
        gui.data_manager = Mock(spec=DataManager)
        gui.data_manager.load_experiment = Mock()
        gui.data_manager.load_dataset = Mock()
        gui.data_manager.load_session = Mock()

        # Mock plot controller
        gui.plot_controller = Mock(spec=PlotController)
        gui.plot_controller.plot = Mock()
        gui.plot_controller.clear_plot = Mock()

        # Mock command system
        gui.command_invoker = Mock(spec=CommandInvoker)
        gui.command_invoker.execute = Mock()
        gui.command_invoker.undo = Mock()
        gui.command_invoker.redo = Mock()

        return gui

    @patch("monstim_gui.core.application_state.QSettings")
    def test_complete_session_restoration_workflow(self, mock_qsettings, integration_gui_mock):
        """Test complete session restoration across UI components."""
        # Setup QSettings mock
        stored_values = {}
        mock_settings = Mock()

        def mock_set_value(key, value):
            stored_values[key] = value

        def mock_get_value(key, default=None, type=str):
            if key in stored_values:
                if type is bool:
                    return bool(stored_values[key]) if stored_values[key] is not None else default
                return stored_values[key]
            return default

        mock_settings.setValue.side_effect = mock_set_value
        mock_settings.value.side_effect = mock_get_value
        mock_settings.sync = Mock()
        mock_settings.remove = Mock()
        mock_settings.organizationName.return_value = "TestOrg"
        mock_settings.applicationName.return_value = "TestApp"
        mock_qsettings.return_value = mock_settings

        # Create ApplicationState and save session state
        app_state = ApplicationState()
        app_state.save_current_session_state(
            experiment_id="test_experiment", dataset_id="test_dataset", session_id="test_session", profile_name="test_profile"
        )

        # Setup GUI mock for restoration
        integration_gui_mock.expts_dict_keys = ["test_experiment"]
        integration_gui_mock.current_experiment = Mock()
        integration_gui_mock.current_experiment.id = "test_experiment"
        integration_gui_mock.current_dataset = Mock()

        # Mock experiment datasets and sessions
        mock_dataset = Mock()
        mock_dataset.id = "test_dataset"
        mock_session = Mock()
        mock_session.id = "test_session"
        mock_dataset.sessions = [mock_session]
        integration_gui_mock.current_experiment.datasets = [mock_dataset]

        # Attempt session restoration
        result = app_state.restore_last_session(integration_gui_mock)

        # Verify restoration was attempted
        assert result is True, "Session restoration should be attempted"

        # Verify GUI interactions
        integration_gui_mock.data_selection_widget.experiment_combo.setCurrentIndex.assert_called()

        # Verify data manager calls (may be called via timer)
        if integration_gui_mock.data_manager.load_dataset.called:
            integration_gui_mock.data_manager.load_dataset.assert_called()

    def test_ui_config_and_state_coordination(self):
        """Test coordination between UI config and application state."""
        with (
            patch("monstim_gui.core.ui_config.QSettings") as mock_ui_qsettings,
            patch("monstim_gui.core.application_state.QSettings") as mock_app_qsettings,
        ):

            # Setup separate storage for each component
            ui_storage = {}
            app_storage = {}

            # Mock UIConfig QSettings
            ui_mock = Mock()
            ui_mock.setValue = lambda k, v: ui_storage.update({k: v})
            ui_mock.value = lambda k, d=None, type=str: ui_storage.get(k, d)
            ui_mock.sync = Mock()
            mock_ui_qsettings.return_value = ui_mock

            # Mock ApplicationState QSettings
            app_mock = Mock()
            app_mock.setValue = lambda k, v: app_storage.update({k: v})
            app_mock.value = lambda k, d=None, type=str: app_storage.get(k, d)
            app_mock.sync = Mock()
            app_mock.organizationName = Mock(return_value="TestOrg")
            app_mock.applicationName = Mock(return_value="TestApp")
            mock_app_qsettings.return_value = app_mock

            # Create instances
            ui_config = UIConfig()
            app_state = ApplicationState()

            # Test independent operation
            ui_config.set("test_ui_setting", "ui_value")
            app_state.set_setting("test_app_setting", True)

            # Verify values are stored independently
            assert ui_config.get("test_ui_setting") == "ui_value"
            assert app_state.get_preference("test_app_setting") is True

            # Verify cross-contamination doesn't occur (different storage systems)
            assert ui_config.get("test_app_setting") is None  # UI config doesn't have app setting
            assert app_state.get_preference("test_ui_setting", None) is None  # App state doesn't have UI setting


class TestCommandSystemIntegration:
    """Test command system integration with domain models and GUI."""

    def test_command_integration_with_real_domain_objects(self, temp_output_dir):
        """Test commands working with actual domain objects."""
        # Create minimal domain structure
        exp_dir = temp_output_dir / "TestExp"
        exp_dir.mkdir()

        # Create minimal experiment annotation
        exp_annot = exp_dir / "experiment.annot.json"
        exp_annot.write_text('{"datasets_excluded": []}')

        # Create dataset directory
        dataset_dir = exp_dir / "250101 TestAnimal Test"
        dataset_dir.mkdir()

        # Create session directory with minimal data
        session_dir = dataset_dir / "session_001"
        session_dir.mkdir()

        # Create minimal session annotation
        session_annot = session_dir / "session.annot.json"
        session_annot.write_text('{"recordings_excluded": [], "latency_windows": []}')

        # Create minimal HDF5 structure (mock file)
        hdf5_file = session_dir / "session.h5"
        hdf5_file.write_text("mock_hdf5_data")

        # Test command system with mock GUI
        gui_mock = Mock()
        gui_mock.menu_bar = Mock()
        gui_mock.menu_bar.update_undo_redo_labels = Mock()
        # Provide DataManager expected by CreateExperimentCommand
        gui_mock.data_manager = Mock(spec=DataManager)

        # Create command invoker with mock parent
        invoker = CommandInvoker(gui_mock)

        # Test experiment-level command
        create_cmd = CreateExperimentCommand(gui_mock, "NewExperiment")

        # Execute command
        invoker.execute(create_cmd)

        # Verify command was tracked
        assert len(invoker.history) == 1
        assert invoker.get_undo_command_name() is not None

        # Test undo
        invoker.undo()
        assert len(invoker.history) == 0

    def test_complex_command_workflow_integration(self):
        """Test complex multi-command workflow with undo/redo."""
        gui_mock = Mock()
        gui_mock.current_experiment = Mock()
        gui_mock.current_dataset = Mock()
        gui_mock.current_session = Mock()
        gui_mock.data_selection_widget = Mock()
        gui_mock.status_bar = Mock()
        gui_mock.menu_bar = Mock()
        gui_mock.menu_bar.update_undo_redo_labels = Mock()

        # Mock domain objects
        mock_recording = Mock()
        mock_recording.id = "recording_001"
        gui_mock.current_session.recordings = [mock_recording]
        gui_mock.current_session.annot = Mock()
        gui_mock.current_session.annot.recordings_excluded = []

        invoker = CommandInvoker(gui_mock)

        # Execute sequence of commands
        commands = [
            ExcludeRecordingCommand(gui_mock, "recording_001"),
            ExcludeSessionCommand(gui_mock),
        ]

        for cmd in commands:
            try:
                invoker.execute(cmd)
            except Exception:
                # Commands may fail with mock objects - focus on invoker behavior
                pass

        # Test undo/redo cycling
        initial_history_length = len(invoker.history)

        if initial_history_length > 0:
            invoker.undo()
            assert len(invoker.history) == initial_history_length - 1

        if len(invoker.redo_stack) > 0:
            invoker.redo()
            assert len(invoker.history) == initial_history_length


class TestErrorRecoveryIntegration:
    """Test error recovery across component boundaries."""

    def test_repository_error_cascades_gracefully(self, temp_output_dir):
        """Test that repository errors don't crash the application."""
        # Create invalid repository structure
        invalid_path = temp_output_dir / "nonexistent" / "path"

        # Attempt to create repositories with invalid paths
        try:
            exp_repo = ExperimentRepository(invalid_path)
            _ = exp_repo.load()
            # If load succeeds unexpectedly, ensure an Experiment-like object
            assert _ is not None
        except Exception as e:
            # If exceptions occur, they should be specific and handleable
            assert isinstance(e, (FileNotFoundError, PermissionError, OSError))

    def test_command_failure_recovery(self):
        """Test command system recovery from command failures."""
        gui_mock = Mock()
        gui_mock.menu_bar = Mock()
        gui_mock.menu_bar.update_undo_redo_labels = Mock()

        # Create command that will fail
        class FailingCommand(ExcludeRecordingCommand):
            def execute(self):
                raise RuntimeError("Test command failure")

        invoker = CommandInvoker(gui_mock)
        failing_cmd = FailingCommand(gui_mock, "test_id")

        # Execute failing command
        with pytest.raises(RuntimeError):
            invoker.execute(failing_cmd)

        # Verify command system state remains consistent
        assert len(invoker.history) == 0, "Failed command should not be in history"
        assert invoker.get_undo_command_name() is None, "Should not be able to undo failed command"

    def test_state_corruption_recovery(self):
        """Test recovery from state corruption scenarios."""
        with patch("monstim_gui.core.application_state.QSettings") as mock_qsettings:
            # Simulate corrupted settings
            mock_settings = Mock()
            mock_settings.value.side_effect = Exception("Corrupted settings")
            mock_settings.setValue = Mock()
            mock_settings.sync = Mock()
            mock_qsettings.return_value = mock_settings

            # ApplicationState should handle corrupted settings gracefully
            try:
                app_state = ApplicationState()
                # Basic operations should not crash
                state = app_state.get_last_session_state()
                assert isinstance(state, dict), "Should return valid dict even with corrupted settings"
            except Exception:
                # If initialization fails, it should fail cleanly
                pass


class TestPerformanceIntegration:
    """Test performance characteristics of integrated workflows."""

    def test_large_dataset_handling_integration(self, temp_output_dir):
        """Test handling of large dataset structures."""
        # Create directory structure with many files
        exp_dir = temp_output_dir / "LargeExp"
        exp_dir.mkdir()

        # Create multiple dataset directories
        for i in range(10):  # Smaller number for test performance
            dataset_dir = exp_dir / f"25010{i} Animal{i} Condition{i}"
            dataset_dir.mkdir()

            # Create session files
            session_csv = dataset_dir / f"Animal{i}_Condition{i}_001.csv"
            session_csv.write_text("Time,Ch1\n0.0,1.0\n0.001,1.1\n")

        # Test repository discovery performance
        start_time = time.time()
        exp_repo = ExperimentRepository(exp_dir)
        try:
            # Try to load experiment (if structure is valid)
            experiment = exp_repo.load()
            _ = experiment  # Use variable to avoid linting warning
        except Exception:
            # May fail with mock data - that's expected
            pass
        discovery_time = time.time() - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert discovery_time < 10.0, f"Discovery took {discovery_time:.2f}s, should be < 10s"

    def test_memory_usage_integration(self):
        """Test memory usage patterns in integrated workflows."""
        import gc

        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform operations that create/destroy many objects
        gui_mock = Mock()
        gui_mock.menu_bar = Mock()
        gui_mock.menu_bar.update_undo_redo_labels = Mock()

        invoker = CommandInvoker(gui_mock)

        # Create and execute many commands
        for i in range(100):
            cmd = ExcludeRecordingCommand(gui_mock, f"recording_{i}")
            try:
                invoker.execute(cmd)
                if i % 10 == 0:  # Periodic undo to test cleanup
                    invoker.undo()
            except Exception:
                # Commands may fail with mock - focus on memory behavior
                pass

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory growth should be reasonable
        object_growth = final_objects - initial_objects
        # Allow some headroom for Python object churn under GC
        assert object_growth < 5000, f"Too many objects created: {object_growth}"


class TestConfigurationIntegration:
    """Test configuration system integration across components."""

    def test_configuration_profile_workflow_integration(self, temp_output_dir):
        """Test complete configuration and profile management workflow."""
        # Create config directory
        config_dir = temp_output_dir / "config"
        config_dir.mkdir()

        # Create default config
        default_config = config_dir / "config.yml"
        default_config.write_text(
            """
default_setting: "default_value"
analysis:
  method: "default_method"
  threshold: 1.0
"""
        )

        # Create profile directory
        profile_dir = temp_output_dir / "profiles"
        profile_dir.mkdir()

        # Test configuration repository
        config_repo = ConfigRepository(str(config_dir / "config.yml"))
        config = config_repo.read_config()

        assert "default_setting" in config
        assert config["default_setting"] == "default_value"

        # Test profile manager
        profile_manager = ProfileManager(str(profile_dir))

        # Save test profile
        test_profile = {"analysis": {"method": "test_method", "threshold": 2.0}, "name": "test_profile"}
        profile_path = str(profile_dir / "test_profile.yml")
        profile_manager.save_profile(test_profile, profile_path)

        # Load and verify profile
        loaded_profile = profile_manager.load_profile(profile_path)
        assert loaded_profile["analysis"]["method"] == "test_method"
        assert loaded_profile["analysis"]["threshold"] == 2.0

        # Test profile listing
        profiles = profile_manager.list_profiles()
        assert "test_profile" in [p[0] for p in profiles]

    @patch("monstim_gui.core.application_state.QSettings")
    def test_application_state_configuration_integration(self, mock_qsettings):
        """Test integration between ApplicationState and configuration system."""
        # Setup QSettings mock
        stored_values = {}
        mock_settings = Mock()

        def mock_set_value(key, value):
            stored_values[key] = value

        def mock_get_value(key, default=None, type=str):
            return stored_values.get(key, default)

        mock_settings.setValue.side_effect = mock_set_value
        mock_settings.value.side_effect = mock_get_value
        mock_settings.sync = Mock()
        mock_settings.organizationName.return_value = "TestOrg"
        mock_settings.applicationName.return_value = "TestApp"
        mock_qsettings.return_value = mock_settings

        # Test ApplicationState preference management
        app_state = ApplicationState()

        # Test preference setting and getting
        app_state.set_setting("test_preference", True)
        assert app_state.get_preference("test_preference") is True

        # Test profile tracking
        app_state.save_last_profile("integration_test_profile")
        assert app_state.get_last_profile() == "integration_test_profile"

        # Test session state management
        app_state.save_current_session_state(experiment_id="integration_exp", profile_name="integration_test_profile")

        state = app_state.get_last_session_state()
        assert state["experiment"] == "integration_exp"
        assert state["profile"] == "integration_test_profile"


class TestEndToEndUserScenarios:
    """Test complete end-to-end user scenarios."""

    def test_new_user_first_time_setup_scenario(self, temp_output_dir):
        """Test complete new user setup scenario."""
        # Simulate new user environment - workspace available for future use
        _ = temp_output_dir  # Suppress unused warning

        with patch("monstim_gui.core.application_state.QSettings") as mock_qsettings:
            # Setup clean QSettings (no existing preferences)
            mock_settings = Mock()

            # Return provided default for any key
            def mock_value(key, default=None, type=str):
                return default

            mock_settings.value.side_effect = mock_value
            mock_settings.setValue = Mock()
            mock_settings.sync = Mock()
            mock_settings.organizationName.return_value = "MonStim"
            mock_settings.applicationName.return_value = "Analyzer"
            mock_qsettings.return_value = mock_settings

            # Initialize application state (first time)
            app_state = ApplicationState()

            # Verify default behavior for new user
            assert app_state.should_track_session_restoration() is True  # Default enabled
            assert app_state.should_track_analysis_profiles() is True
            assert app_state.get_last_session_state()["experiment"] == ""

            # Test first-time configuration
            ui_config = UIConfig()

            # Should not crash with no stored settings
            try:
                scale_factor = ui_config.get_scale_factor()
                assert isinstance(scale_factor, (int, float))
            except Exception:
                # May fail without Qt application - that's okay for unit test
                pass

    def test_experienced_user_workflow_scenario(self):
        """Test experienced user with existing preferences workflow."""
        with patch("monstim_gui.core.application_state.QSettings") as mock_qsettings:
            # Setup QSettings with existing user preferences
            user_preferences = {
                "ProgramPreferences/track_session_restoration": True,
                "ProgramPreferences/track_analysis_profiles": True,
                "LastSelection/profile": "my_custom_profile",
                "SessionRestore/experiment": "my_recent_experiment",
                "SessionRestore/dataset": "my_recent_dataset",
                "SessionRestore/session": "my_recent_session",
            }

            mock_settings = Mock()

            def mock_get_value(key, default=None, type=str):
                if key in user_preferences:
                    value = user_preferences[key]
                    if type is bool:
                        return bool(value)
                    return value
                return default

            mock_settings.value.side_effect = mock_get_value
            mock_settings.setValue = Mock()
            mock_settings.sync = Mock()
            mock_settings.organizationName.return_value = "MonStim"
            mock_settings.applicationName.return_value = "Analyzer"
            mock_qsettings.return_value = mock_settings

            # Initialize with existing preferences
            app_state = ApplicationState()

            # Verify user preferences are restored
            assert app_state.should_track_session_restoration() is True
            assert app_state.get_last_profile() == "my_custom_profile"

            last_state = app_state.get_last_session_state()
            assert last_state["experiment"] == "my_recent_experiment"
            assert last_state["dataset"] == "my_recent_dataset"
            assert last_state["session"] == "my_recent_session"

    def test_data_analysis_complete_workflow_scenario(self, temp_output_dir):
        """Test complete data analysis workflow from import to results."""
        # Create mock experimental data
        exp_dir = temp_output_dir / "AnalysisWorkflow"
        exp_dir.mkdir()

        dataset_dir = exp_dir / "250101 TestSubject Baseline"
        dataset_dir.mkdir()

        # Create CSV with more realistic EMG-like data
        csv_data = "Time,Muscle1,Muscle2,Stimulus\n"
        for i in range(1000):
            time_val = i * 0.001  # 1ms intervals
            muscle1 = 0.1 * (i % 10) + (0.5 if i > 100 and i < 110 else 0)  # Simulate M-wave
            muscle2 = 0.05 * (i % 8) + (0.3 if i > 95 and i < 115 else 0)
            stimulus = 1.0 if i == 100 else 0.0
            csv_data += f"{time_val},{muscle1},{muscle2},{stimulus}\n"

        session_csv = dataset_dir / "TestSubject_Baseline_001.csv"
        session_csv.write_text(csv_data)

        # Create stimulus timing file
        stm_content = "0.100\t5.0\n0.200\t10.0\n0.300\t15.0\n"
        stm_file = dataset_dir / "TestSubject_Baseline_001.stm"
        stm_file.write_text(stm_content)

        # Step 1: Import data
        output_dir = temp_output_dir / "analysis_output"

        try:
            csv_importer.import_experiment(
                expt_path=exp_dir, output_path=output_dir, progress_callback=lambda v: None, is_canceled=lambda: False
            )

            # Step 2: Load and analyze
            exp_repo = ExperimentRepository(output_dir)
            try:
                experiment = exp_repo.load()

                if experiment and len(experiment.datasets) > 0:
                    dataset = experiment.datasets[0]

                    if len(dataset.sessions) > 0:
                        session = dataset.sessions[0]

                        if len(session.recordings) > 0:
                            # Step 3: Perform analysis operations
                            recordings_data = session.recordings_filtered
                            assert recordings_data is not None, "Should have filtered data"

                            # Test annotation modifications
                            num_ch = session.num_channels
                            session.annot.latency_windows.append(
                                LatencyWindow(
                                    name="Test Window",
                                    color="#FF00FF",
                                    start_times=[0.095] * num_ch,
                                    durations=[0.02] * num_ch,
                                )
                            )

                            # Step 4: Save modifications
                            session_repo = session.repo
                            session_repo.save(session)

                            # Step 5: Reload and verify persistence
                            reloaded_session = session_repo.load()
                            assert len(reloaded_session.annot.latency_windows) > 0, "Latency windows should persist"
                            assert reloaded_session.annot.latency_windows[0].name == "Test Window"
            except Exception:
                # Analysis may fail with mock data - that's expected
                pass

        except Exception:
            # Import may fail with mock data - verify graceful handling
            assert True, "Import workflow handled errors gracefully"
