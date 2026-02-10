"""
Comprehensive test suite for the Command Pattern System.

Tests the CommandInvoker undo/redo mechanics, command execution,
data integrity, and all implemented command classes using real domain objects.
"""

from collections import deque
from unittest.mock import Mock

import pytest

from monstim_gui.commands import (
    BulkRecordingExclusionCommand,
    ChangeChannelNamesCommand,
    Command,
    CommandInvoker,
    CreateExperimentCommand,
    ExcludeDatasetCommand,
    ExcludeRecordingCommand,
    ExcludeSessionCommand,
    InvertChannelPolarityCommand,
    MoveDatasetCommand,
    RenameExperimentCommand,
    RestoreRecordingCommand,
    RestoreSessionCommand,
    SetLatencyWindowsCommand,
)
from monstim_signals.core.data_models import LatencyWindow
from monstim_signals.io.repositories import DatasetRepository, SessionRepository

# --- Test Annotations ---
# Purpose: Validate command pattern integration (undo/redo, behavior, and persistence triggers)
# Markers: integration (touches command system and filesystem via mocks/fixtures)
# Key fixtures: temp_output_dir (via helpers where relevant), fake_gui (from conftest)
# Notes: Uses Mock GUI components; aligns with current Command signatures in monstim_gui.commands
pytestmark = pytest.mark.integration


class TestCommandInvoker:
    """Test the core CommandInvoker undo/redo mechanics."""

    def test_command_invoker_initialization(self):
        """Test that CommandInvoker initializes correctly."""
        mock_gui = Mock()
        invoker = CommandInvoker(mock_gui)

        assert invoker.parent is mock_gui
        assert isinstance(invoker.history, deque)
        assert isinstance(invoker.redo_stack, deque)
        assert len(invoker.history) == 0
        assert len(invoker.redo_stack) == 0

    def test_execute_command_basic(self):
        """Test basic command execution."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        invoker = CommandInvoker(mock_gui)

        # Create a mock command
        mock_command = Mock(spec=Command)
        mock_command.command_name = "Test Command"

        # Execute command
        invoker.execute(mock_command)

        # Verify execution
        mock_command.execute.assert_called_once()
        assert len(invoker.history) == 1
        assert invoker.history[0] is mock_command
        assert len(invoker.redo_stack) == 0
        mock_gui.menu_bar.update_undo_redo_labels.assert_called_once()

    def test_execute_clears_redo_stack(self):
        """Test that executing a new command clears the redo stack."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        invoker = CommandInvoker(mock_gui)

        # Set up initial state with items in redo stack
        old_command = Mock(spec=Command)
        old_command.command_name = "Old Command"
        invoker.redo_stack.append(old_command)

        # Execute new command
        new_command = Mock(spec=Command)
        new_command.command_name = "New Command"
        invoker.execute(new_command)

        # Verify redo stack is cleared
        assert len(invoker.redo_stack) == 0
        assert len(invoker.history) == 1

    def test_undo_command_basic(self):
        """Test basic command undo functionality."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        invoker = CommandInvoker(mock_gui)

        # Execute a command first
        mock_command = Mock(spec=Command)
        mock_command.command_name = "Test Command"
        invoker.execute(mock_command)

        # Undo the command
        invoker.undo()

        # Verify undo
        mock_command.undo.assert_called_once()
        assert len(invoker.history) == 0
        assert len(invoker.redo_stack) == 1
        assert invoker.redo_stack[0] is mock_command
        assert mock_gui.menu_bar.update_undo_redo_labels.call_count == 2

    def test_undo_empty_history(self):
        """Test undo with empty history does nothing."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        invoker = CommandInvoker(mock_gui)

        # Try to undo with empty history
        invoker.undo()

        # Verify nothing happens
        assert len(invoker.history) == 0
        assert len(invoker.redo_stack) == 0

    def test_redo_command_basic(self):
        """Test basic command redo functionality."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        invoker = CommandInvoker(mock_gui)

        # Execute and undo a command
        mock_command = Mock(spec=Command)
        mock_command.command_name = "Test Command"
        invoker.execute(mock_command)
        invoker.undo()

        # Reset mock to check redo execution
        mock_command.reset_mock()

        # Redo the command
        invoker.redo()

        # Verify redo
        mock_command.execute.assert_called_once()
        assert len(invoker.history) == 1
        assert len(invoker.redo_stack) == 0
        assert invoker.history[0] is mock_command

    def test_redo_empty_stack(self):
        """Test redo with empty redo stack does nothing."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        invoker = CommandInvoker(mock_gui)

        # Try to redo with empty redo stack
        invoker.redo()

        # Verify nothing happens
        assert len(invoker.history) == 0
        assert len(invoker.redo_stack) == 0

    def test_multiple_commands_undo_redo_sequence(self):
        """Test complex sequence of multiple commands with undo/redo."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        invoker = CommandInvoker(mock_gui)

        # Create multiple commands
        cmd1 = Mock(spec=Command)
        cmd1.command_name = "Command 1"
        cmd2 = Mock(spec=Command)
        cmd2.command_name = "Command 2"
        cmd3 = Mock(spec=Command)
        cmd3.command_name = "Command 3"

        # Execute commands
        invoker.execute(cmd1)
        invoker.execute(cmd2)
        invoker.execute(cmd3)

        assert len(invoker.history) == 3
        assert len(invoker.redo_stack) == 0

        # Undo two commands
        invoker.undo()  # Undo cmd3
        invoker.undo()  # Undo cmd2

        assert len(invoker.history) == 1
        assert len(invoker.redo_stack) == 2
        assert invoker.history[0] is cmd1

        # Redo one command
        invoker.redo()  # Redo cmd2

        assert len(invoker.history) == 2
        assert len(invoker.redo_stack) == 1
        assert invoker.history[-1] is cmd2

    def test_get_command_names(self):
        """Test getting command names for undo/redo."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        invoker = CommandInvoker(mock_gui)

        # Initially no commands
        assert invoker.get_undo_command_name() is None
        assert invoker.get_redo_command_name() is None

        # Execute command
        mock_command = Mock(spec=Command)
        mock_command.command_name = "Test Command"
        invoker.execute(mock_command)

        assert invoker.get_undo_command_name() == "Test Command"
        assert invoker.get_redo_command_name() is None

        # Undo command
        invoker.undo()

        assert invoker.get_undo_command_name() is None
        assert invoker.get_redo_command_name() == "Test Command"

    def test_remove_command_by_name(self):
        """Test removing commands by name from history and redo stack."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        invoker = CommandInvoker(mock_gui)

        # Create commands with same and different names
        cmd1 = Mock(spec=Command)
        cmd1.command_name = "Remove Me"
        cmd2 = Mock(spec=Command)
        cmd2.command_name = "Keep Me"
        cmd3 = Mock(spec=Command)
        cmd3.command_name = "Remove Me"

        # Add to history and redo_stack
        invoker.history.extend([cmd1, cmd2, cmd3])
        invoker.redo_stack.extend([cmd1, cmd2])

        # Remove commands named "Remove Me"
        invoker.remove_command_by_name("Remove Me")

        # Verify removal
        assert len(invoker.history) == 1
        assert invoker.history[0] is cmd2
        assert len(invoker.redo_stack) == 1
        assert invoker.redo_stack[0] is cmd2


class TestCommandExecutionIntegrity:
    """Test command execution maintains data integrity."""

    def test_command_execute_and_undo_maintain_state(self):
        """Test that execute and undo operations maintain consistent state."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        invoker = CommandInvoker(mock_gui)

        # Mock object with state to modify
        mock_obj = Mock()
        mock_obj.value = "initial"

        # Create command that modifies state
        class TestCommand(Command):
            command_name = "Test State Command"

            def __init__(self, obj):
                self.obj = obj
                self.old_value = None
                self.new_value = "modified"

            def execute(self):
                self.old_value = self.obj.value
                self.obj.value = self.new_value

            def undo(self):
                self.obj.value = self.old_value

        cmd = TestCommand(mock_obj)

        # Test execution
        invoker.execute(cmd)
        assert mock_obj.value == "modified"

        # Test undo
        invoker.undo()
        assert mock_obj.value == "initial"

        # Test redo
        invoker.redo()
        assert mock_obj.value == "modified"

    def test_command_failure_during_execute(self):
        """Test handling of command failures during execution."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        invoker = CommandInvoker(mock_gui)

        # Create command that fails during execution
        failing_command = Mock(spec=Command)
        failing_command.command_name = "Failing Command"
        failing_command.execute.side_effect = Exception("Execute failed")

        # Execute should propagate the exception
        with pytest.raises(Exception, match="Execute failed"):
            invoker.execute(failing_command)

        # Verify command was not added to history
        assert len(invoker.history) == 0
        assert len(invoker.redo_stack) == 0

    def test_command_failure_during_undo(self):
        """Test handling of command failures during undo."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        invoker = CommandInvoker(mock_gui)

        # Create command that fails during undo
        failing_command = Mock(spec=Command)
        failing_command.command_name = "Failing Undo Command"
        failing_command.undo.side_effect = Exception("Undo failed")

        # Execute command successfully first
        invoker.execute(failing_command)
        assert len(invoker.history) == 1

        # Undo should propagate the exception
        with pytest.raises(Exception, match="Undo failed"):
            invoker.undo()

        # Verify state is still consistent (command removed from history even if undo failed)
        # Current implementation doesn't push to redo_stack if undo raises
        assert len(invoker.history) == 0
        assert len(invoker.redo_stack) == 0


class TestRecordingCommands:
    """Test recording-level commands."""

    def test_exclude_recording_command(self):
        """Test ExcludeRecordingCommand execution and undo."""
        mock_gui = Mock()
        mock_gui.current_session = Mock()
        mock_gui.data_selection_widget = Mock()

        cmd = ExcludeRecordingCommand(mock_gui, "REC001")

        # Test command name
        assert cmd.command_name == "Exclude Recording"

        # Test execute
        cmd.execute()
        mock_gui.current_session.exclude_recording.assert_called_once_with("REC001")
        mock_gui.data_selection_widget.sync_combo_selections.assert_called_once()

        # Test undo
        cmd.undo()
        mock_gui.current_session.restore_recording.assert_called_once_with("REC001")

    def test_restore_recording_command(self):
        """Test RestoreRecordingCommand execution and undo."""
        mock_gui = Mock()
        mock_gui.current_session = Mock()
        mock_gui.data_selection_widget = Mock()

        cmd = RestoreRecordingCommand(mock_gui, "REC001")

        # Test command name
        assert cmd.command_name == "Restore Recording"

        # Test execute
        cmd.execute()
        mock_gui.current_session.restore_recording.assert_called_once_with("REC001")
        mock_gui.data_selection_widget.sync_combo_selections.assert_called_once()

        # Test undo
        cmd.undo()
        mock_gui.current_session.exclude_recording.assert_called_once_with("REC001")

    def test_bulk_recording_exclusion_command(self):
        """Test BulkRecordingExclusionCommand execution and undo."""
        mock_gui = Mock()
        mock_gui.data_selection_widget = Mock()
        session = Mock()
        # Build changes: two exclusions and one inclusion
        changes = [
            {
                "session": session,
                "changes": [
                    {"recording_id": "REC001", "exclude": True},
                    {"recording_id": "REC002", "exclude": False},
                    {"recording_id": "REC003", "exclude": True},
                ],
            }
        ]

        cmd = BulkRecordingExclusionCommand(mock_gui, changes)

        # Test command name
        assert cmd.command_name == "Bulk Recording Exclusion"

        # Test execute
        cmd.execute()
        session.exclude_recording.assert_any_call("REC001")
        session.restore_recording.assert_any_call("REC002")
        session.exclude_recording.assert_any_call("REC003")
        mock_gui.data_selection_widget.sync_combo_selections.assert_called_once()

        # Test undo (reverse operations)
        cmd.undo()
        session.restore_recording.assert_any_call("REC001")
        session.exclude_recording.assert_any_call("REC002")
        session.restore_recording.assert_any_call("REC003")
        assert mock_gui.data_selection_widget.sync_combo_selections.call_count == 2


class TestSessionCommands:
    """Test session-level commands."""

    def test_exclude_session_command(self):
        """Test ExcludeSessionCommand execution and undo."""
        mock_gui = Mock()
        mock_gui.data_selection_widget = Mock()
        mock_session = Mock()
        mock_session.id = "SES001"
        mock_dataset = Mock()
        mock_dataset.sessions = [mock_session]
        mock_gui.current_dataset = mock_dataset
        mock_gui.current_session = mock_session

        cmd = ExcludeSessionCommand(mock_gui)

        # Test command name
        assert cmd.command_name == "Exclude Session"

        # Test execute
        cmd.execute()
        mock_dataset.exclude_session.assert_called_once_with("SES001")

        # Test undo
        cmd.undo()
        mock_dataset.restore_session.assert_called_once_with("SES001")

    def test_exclude_session_command_already_excluded(self):
        """Test ExcludeSessionCommand gracefully handles already-excluded session."""
        mock_gui = Mock()
        mock_gui.data_selection_widget = Mock()
        mock_session = Mock()
        mock_session.id = "SES001"
        mock_dataset = Mock()
        # Session is NOT in the sessions list (already excluded)
        mock_dataset.sessions = []
        mock_gui.current_dataset = mock_dataset
        mock_gui.current_session = mock_session

        cmd = ExcludeSessionCommand(mock_gui)

        # Test execute - should NOT crash and should NOT call exclude_session
        cmd.execute()
        # exclude_session should not be called since session is not in list
        mock_dataset.exclude_session.assert_not_called()

    def test_exclude_session_command_no_selection(self):
        """Test ExcludeSessionCommand gracefully handles no current session/dataset."""
        mock_gui = Mock()
        mock_gui.data_selection_widget = Mock()
        mock_gui.current_dataset = None
        mock_gui.current_session = None

        cmd = ExcludeSessionCommand(mock_gui)

        # Test execute - should NOT crash
        cmd.execute()
        # No methods should be called since there's no valid selection

    def test_restore_session_command(self):
        """Test RestoreSessionCommand execution and undo."""
        mock_gui = Mock()
        mock_gui.data_selection_widget = Mock()
        mock_session = Mock()
        mock_session.id = "SES001"
        mock_dataset = Mock()
        mock_dataset.get_all_sessions.return_value = [mock_session]
        mock_gui.current_dataset = mock_dataset

        cmd = RestoreSessionCommand(mock_gui, "SES001")

        # Test command name
        assert cmd.command_name == "Restore Session"

        # Test execute
        cmd.execute()
        mock_dataset.restore_session.assert_called_once_with("SES001")

        # Test undo
        cmd.undo()
        mock_dataset.exclude_session.assert_called_once_with("SES001")


class TestDatasetCommands:
    """Test dataset-level commands."""

    def test_exclude_dataset_command(self):
        """Test ExcludeDatasetCommand execution and undo."""
        mock_gui = Mock()
        mock_gui.data_selection_widget = Mock()
        mock_dataset = Mock()
        mock_dataset.id = "DS001"
        mock_experiment = Mock()
        mock_experiment.datasets = [mock_dataset]
        mock_gui.current_experiment = mock_experiment
        mock_gui.current_dataset = mock_dataset

        cmd = ExcludeDatasetCommand(mock_gui)

        # Test command name
        assert cmd.command_name == "Exclude Dataset"

        # Test execute
        cmd.execute()
        mock_experiment.exclude_dataset.assert_called_once_with("DS001")

        # Test undo
        cmd.undo()
        mock_experiment.restore_dataset.assert_called_once_with("DS001")

    def test_exclude_dataset_command_already_excluded(self):
        """Test ExcludeDatasetCommand gracefully handles already-excluded dataset."""
        mock_gui = Mock()
        mock_gui.data_selection_widget = Mock()
        mock_dataset = Mock()
        mock_dataset.id = "DS001"
        mock_experiment = Mock()
        # Dataset is NOT in the datasets list (already excluded)
        mock_experiment.datasets = []
        mock_gui.current_experiment = mock_experiment
        mock_gui.current_dataset = mock_dataset

        cmd = ExcludeDatasetCommand(mock_gui)

        # Test execute - should NOT crash and should NOT call exclude_dataset
        cmd.execute()
        # exclude_dataset should not be called since dataset is not in list
        mock_experiment.exclude_dataset.assert_not_called()

    def test_exclude_dataset_command_no_selection(self):
        """Test ExcludeDatasetCommand gracefully handles no current dataset/experiment."""
        mock_gui = Mock()
        mock_gui.data_selection_widget = Mock()
        mock_gui.current_experiment = None
        mock_gui.current_dataset = None

        cmd = ExcludeDatasetCommand(mock_gui)

        # Test execute - should NOT crash
        cmd.execute()
        # No methods should be called since there's no valid selection


class TestChannelCommands:
    """Test channel-related commands."""

    def test_invert_channel_polarity_command(self):
        """Test InvertChannelPolarityCommand execution and undo."""
        mock_gui = Mock()
        mock_session = Mock()
        mock_gui.current_session = mock_session
        channel_index = 0

        cmd = InvertChannelPolarityCommand(mock_gui, "session", [channel_index])

        # Test command name
        assert cmd.command_name == "Invert Channel Polarity"

        # Test execute
        cmd.execute()
        mock_session.invert_channel_polarity.assert_called_once_with(channel_index)

        # Test undo (invert again)
        cmd.undo()
        assert mock_session.invert_channel_polarity.call_count == 2

    def test_change_channel_names_command(self):
        """Test ChangeChannelNamesCommand execution and undo."""
        mock_gui = Mock()
        mock_session = Mock()
        mock_gui.current_session = mock_session
        mapping = {"Ch0": "EMG1", "Ch1": "EMG2"}

        cmd = ChangeChannelNamesCommand(mock_gui, "session", mapping)

        # Test command name
        assert cmd.command_name == "Change Channel Names"

        # Test execute
        cmd.execute()
        mock_session.rename_channels.assert_called_once_with(mapping)

        # Test undo (reverse mapping)
        cmd.undo()
        mock_session.rename_channels.assert_called_with({"EMG1": "Ch0", "EMG2": "Ch1"})


class TestExperimentCommands:
    """Test experiment-level commands."""

    def test_create_experiment_command(self):
        """Test CreateExperimentCommand execution and undo."""
        mock_gui = Mock()
        mock_data_manager = Mock()
        mock_gui.data_manager = mock_data_manager
        experiment_name = "Test Experiment"

        cmd = CreateExperimentCommand(mock_gui, experiment_name)

        # Test command name
        assert cmd.command_name == "Create Experiment 'Test Experiment'"

        # Test execute
        cmd.execute()
        mock_data_manager.create_experiment.assert_called_once_with(experiment_name)

        # Test undo
        cmd.undo()
        mock_data_manager.delete_experiment_by_id.assert_called_once_with(experiment_name)

    def test_rename_experiment_command(self):
        """Test RenameExperimentCommand execution and undo."""
        mock_gui = Mock()
        mock_data_manager = Mock()
        mock_gui.data_manager = mock_data_manager
        old_name = "Old Name"
        new_name = "New Name"

        cmd = RenameExperimentCommand(mock_gui, old_name, new_name)

        # Test command name
        assert cmd.command_name == "Rename Experiment 'Old Name' to 'New Name'"

        # Test execute
        cmd.execute()
        mock_data_manager.rename_experiment_by_id.assert_called_once_with(old_name, new_name)

        # Test undo
        cmd.undo()
        mock_data_manager.rename_experiment_by_id.assert_called_with(new_name, old_name)


class TestDatasetManagementCommands:
    """Test dataset management commands."""

    def test_move_dataset_command(self):
        """Test MoveDatasetCommand execution and undo."""
        mock_gui = Mock()
        mock_data_manager = Mock()
        mock_gui.data_manager = mock_data_manager

        cmd = MoveDatasetCommand(mock_gui, dataset_id="DS001", dataset_name="DS001", from_exp="ExpA", to_exp="ExpB")

        # Test command name
        assert cmd.command_name == "Move 'DS001' from 'ExpA' to 'ExpB'"

        # Test execute
        cmd.execute()
        mock_data_manager.move_dataset.assert_called_once_with("DS001", "DS001", "ExpA", "ExpB")

        # Test undo
        cmd.undo()
        mock_data_manager.move_dataset.assert_called_with("DS001", "DS001", "ExpB", "ExpA")


class TestCommandIntegrationScenarios:
    """Test complex command integration scenarios."""

    def test_complex_workflow_scenario(self):
        """Test a complex workflow with multiple command types."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        invoker = CommandInvoker(mock_gui)

        # Mock objects
        mock_session = Mock()
        mock_gui.current_session = mock_session

        # Create sequence of commands
        cmd1 = ExcludeRecordingCommand(mock_gui, "REC001")
        cmd2 = ExcludeRecordingCommand(mock_gui, "REC002")
        cmd3 = InvertChannelPolarityCommand(mock_gui, "session", [0])

        # Execute commands in sequence
        invoker.execute(cmd1)
        invoker.execute(cmd2)
        invoker.execute(cmd3)

        # Verify all executed
        assert len(invoker.history) == 3
        mock_session.exclude_recording.assert_any_call("REC001")
        mock_session.exclude_recording.assert_any_call("REC002")
        mock_session.invert_channel_polarity.assert_called_with(0)

        # Undo all commands in reverse order
        invoker.undo()  # Undo channel polarity
        invoker.undo()  # Undo recording2 exclusion
        invoker.undo()  # Undo recording1 exclusion

        # Verify all undone
        assert len(invoker.history) == 0
        assert len(invoker.redo_stack) == 3
        mock_session.restore_recording.assert_any_call("REC001")
        mock_session.restore_recording.assert_any_call("REC002")
        assert mock_session.invert_channel_polarity.call_count == 2  # Execute + undo

    def test_command_memory_management(self):
        """Test that commands are properly managed in memory."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        invoker = CommandInvoker(mock_gui)

        # Create many commands to test memory management
        commands = []
        for i in range(100):
            cmd = Mock(spec=Command)
            cmd.command_name = f"Command {i}"
            commands.append(cmd)
            invoker.execute(cmd)

        # Verify all commands are in history
        assert len(invoker.history) == 100

        # Undo half the commands
        for _ in range(50):
            invoker.undo()

        assert len(invoker.history) == 50
        assert len(invoker.redo_stack) == 50

        # Execute a new command (should clear redo stack)
        new_cmd = Mock(spec=Command)
        new_cmd.command_name = "New Command"
        invoker.execute(new_cmd)

        assert len(invoker.history) == 51
        assert len(invoker.redo_stack) == 0

    def test_command_error_recovery(self):
        """Test command system handles errors gracefully."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        invoker = CommandInvoker(mock_gui)

        # Execute a successful command
        good_cmd = Mock(spec=Command)
        good_cmd.command_name = "Good Command"
        invoker.execute(good_cmd)

        # Try to execute a failing command
        bad_cmd = Mock(spec=Command)
        bad_cmd.command_name = "Bad Command"
        bad_cmd.execute.side_effect = RuntimeError("Command failed")

        with pytest.raises(RuntimeError):
            invoker.execute(bad_cmd)

        # Verify system state is still consistent
        assert len(invoker.history) == 1  # Only good command
        assert invoker.history[0] is good_cmd

        # Can still undo the good command
        invoker.undo()
        good_cmd.undo.assert_called_once()
        assert len(invoker.history) == 0

    def test_command_description_consistency(self):
        """Test that all command classes have consistent descriptions."""
        mock_gui = Mock()
        mock_session = Mock()
        # Test various command descriptions
        mock_gui.current_session = mock_session
        cmd = ExcludeRecordingCommand(mock_gui, "REC001")
        assert cmd.get_description() == "Excluded recording 'REC001'"

        cmd = InvertChannelPolarityCommand(mock_gui, "session", [0])
        assert cmd.get_description() == "Invert Channel Polarity"

        # Test fallback to class name
        class NoNameCommand(Command):
            def execute(self):
                pass

            def undo(self):
                pass

        cmd = NoNameCommand()
        # Base class returns command_name which defaults to None
        assert cmd.get_description() is None


# Test fixtures using real test data
@pytest.fixture
def golden_session(tmp_path):
    """Import golden CSVs to a temp dir and load a real session."""
    from monstim_signals.io.csv_importer import import_experiment
    from tests.helpers import get_golden_root

    out_expt = tmp_path / "GoldenExp"
    import_experiment(get_golden_root(), out_expt, overwrite=True, max_workers=1)

    # Pick first dataset/session
    ds_dirs = [p for p in out_expt.iterdir() if p.is_dir()]
    assert ds_dirs, "No datasets were imported from golden fixtures"
    sess_dirs = [p for p in ds_dirs[0].iterdir() if p.is_dir()]
    assert sess_dirs, f"No sessions found in imported dataset {ds_dirs[0]}"
    session_path = sess_dirs[0]

    # Add M-response latency window for testing
    session = SessionRepository(session_path).load()
    m_window = LatencyWindow(
        name="M-response", color="blue", start_times=[5.0] * session.num_channels, durations=[10.0] * session.num_channels
    )
    session.annot.latency_windows = [m_window]
    session.update_latency_window_parameters()

    return session


@pytest.fixture
def golden_dataset(tmp_path):
    """Import golden CSVs to a temp dir and load a real dataset."""
    from monstim_signals.io.csv_importer import import_experiment
    from tests.helpers import get_golden_root

    out_expt = tmp_path / "GoldenExp"
    import_experiment(get_golden_root(), out_expt, overwrite=True, max_workers=1)

    ds_dirs = [p for p in out_expt.iterdir() if p.is_dir()]
    assert ds_dirs, "No datasets were imported from golden fixtures"
    dataset_path = ds_dirs[0]

    ds = DatasetRepository(dataset_path).load()
    # Ensure no sessions are pre-excluded
    if ds.annot.excluded_sessions:
        ds.annot.excluded_sessions = []
        if ds.repo is not None:
            ds.repo.save(ds)
    return ds


@pytest.fixture
def mock_gui():
    """Create a mock GUI for command testing."""
    gui = Mock()
    gui.data_manager = Mock()
    gui.plot_controller = Mock()
    gui.status_bar = Mock()
    return gui


class TestCommandsWithRealData:
    """Test commands using real domain objects from golden test data."""

    def test_exclude_recording_command_with_real_session(self, golden_session, mock_gui):
        """Test recording exclusion with a real session object."""
        # Get a recording to exclude (use Recording object, not numpy array)
        recording_to_exclude = golden_session.get_all_recordings(include_excluded=True)[0]
        recording_id = recording_to_exclude.id
        initial_count = len(golden_session.recordings_filtered)

        # Set up mock GUI with current session
        mock_gui.current_session = golden_session
        mock_gui.data_selection_widget = Mock()

        # Create and execute exclude command
        cmd = ExcludeRecordingCommand(mock_gui, recording_id)
        cmd.execute()

        # Verify recording is excluded
        assert len(golden_session.recordings_filtered) == initial_count - 1
        assert recording_to_exclude not in golden_session.get_all_recordings(include_excluded=False)

        # Test undo
        cmd.undo()
        assert len(golden_session.recordings_filtered) == initial_count
        assert recording_to_exclude in golden_session.get_all_recordings(include_excluded=False)

    def test_session_exclusion_with_real_dataset(self, golden_dataset, mock_gui):
        """Test session exclusion using real dataset."""
        # Get a session to exclude
        session_to_exclude = golden_dataset.sessions[0]
        initial_count = len(golden_dataset.sessions)

        # Set up mock GUI with current dataset and session
        mock_gui.current_dataset = golden_dataset
        mock_gui.current_session = session_to_exclude
        mock_gui.data_selection_widget = Mock()

        # Create and execute exclude command (takes only gui parameter)
        cmd = ExcludeSessionCommand(mock_gui)
        cmd.execute()

        # Verify session is excluded
        assert len(golden_dataset.sessions) == initial_count - 1
        assert session_to_exclude not in golden_dataset.sessions

        # Test undo
        cmd.undo()
        assert len(golden_dataset.sessions) == initial_count
        assert session_to_exclude in golden_dataset.sessions

    def test_channel_name_modification_with_real_session(self, golden_session, mock_gui):
        """Test channel name modification with real session."""
        channel_index = 0
        original_name = golden_session.annot.channels[channel_index].name
        new_name = "Test Channel Name"

        # Set up mock GUI with current session
        mock_gui.current_session = golden_session

        # Create and execute command (maps old name to new name, not index to new name)
        cmd = ChangeChannelNamesCommand(mock_gui, "session", {original_name: new_name})
        cmd.execute()

        # Verify name changed
        assert golden_session.annot.channels[channel_index].name == new_name

        # Test undo
        cmd.undo()
        assert golden_session.annot.channels[channel_index].name == original_name

    def test_latency_window_commands_with_real_session(self, golden_session, mock_gui):
        """Test latency window commands with real session."""
        # Create a test latency window
        window = LatencyWindow(
            name="Test Window",
            color="red",
            start_times=[5.0] * golden_session.num_channels,
            durations=[10.0] * golden_session.num_channels,
        )

        initial_window_count = len(golden_session.annot.latency_windows)

        # Set up mock GUI with current session
        mock_gui.current_session = golden_session

        # Test adding latency window (level should be string, not object)
        add_cmd = SetLatencyWindowsCommand(mock_gui, "session", [window])
        add_cmd.execute()

        # Verify window was added (it replaces existing windows)
        assert len(golden_session.annot.latency_windows) == 1
        assert golden_session.annot.latency_windows[0].name == "Test Window"

        # Test undo
        add_cmd.undo()
        assert len(golden_session.annot.latency_windows) == initial_window_count

    def test_polarity_inversion_with_real_session(self, golden_session, mock_gui):
        """Test channel polarity inversion with real session."""
        channel_index = 0
        original_polarity = golden_session.annot.channels[channel_index].invert

        # Set up mock GUI with current session
        mock_gui.current_session = golden_session

        # Create and execute command (needs level and channel list parameters)
        cmd = InvertChannelPolarityCommand(mock_gui, "session", [channel_index])
        cmd.execute()

        # Verify polarity changed
        assert golden_session.annot.channels[channel_index].invert != original_polarity

        # Test undo
        cmd.undo()
        assert golden_session.annot.channels[channel_index].invert == original_polarity

    def test_change_channel_names_duplicate_guard_command(self, golden_session, mock_gui):
        """Command should raise ValueError when mapping leads to duplicate names; no change should be applied."""
        mock_gui.current_session = golden_session
        names = list(golden_session.channel_names)
        if len(names) < 2:
            pytest.skip("Need at least 2 channels to test duplicate rename guard")

        # Map two different existing names to the same new name
        mapping = {names[0]: "dup", names[1]: "dup"}
        cmd = ChangeChannelNamesCommand(mock_gui, "session", mapping)

        with pytest.raises(ValueError):
            cmd.execute()

        # No change should be applied
        assert golden_session.channel_names == names
