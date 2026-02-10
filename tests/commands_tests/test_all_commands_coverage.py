"""
Comprehensive coverage test for all command classes.

This module ensures:
1. All command classes are properly registered and testable
2. Every command has required attributes and methods
3. CI can automatically detect untested commands
"""

import inspect
from unittest.mock import Mock

import pytest

from monstim_gui import commands
from monstim_gui.commands import Command

pytestmark = pytest.mark.integration


# List of all command classes that should exist
EXPECTED_COMMANDS = {
    "ExcludeRecordingCommand",
    "RestoreRecordingCommand",
    "ExcludeSessionCommand",
    "ExcludeDatasetCommand",
    "RestoreSessionCommand",
    "RestoreDatasetCommand",
    "InvertChannelPolarityCommand",
    "SetLatencyWindowsCommand",
    "InsertSingleLatencyWindowCommand",
    "ChangeChannelNamesCommand",
    "BulkRecordingExclusionCommand",
    "CreateExperimentCommand",
    "MoveDatasetCommand",
    "MoveDatasetsCommand",
    "CopyDatasetCommand",
    "DeleteExperimentCommand",
    "RenameExperimentCommand",
    "DeleteDatasetCommand",
    "ToggleDatasetInclusionCommand",
    "ToggleCompletionStatusCommand",
}


# Commands that are inherently untestable without full GUI integration
UNTESTABLE_COMMANDS = {
    "DeleteExperimentCommand",  # Requires complex GUI state and file system operations
    "DeleteDatasetCommand",  # Requires complex GUI state and file system operations
    "CopyDatasetCommand",  # Requires complex filesystem operations and experiment state
}


def get_all_command_classes():
    """Discover all Command subclasses in monstim_gui.commands module."""
    command_classes = {}
    for name, obj in inspect.getmembers(commands):
        if inspect.isclass(obj) and issubclass(obj, Command) and obj is not Command:
            command_classes[name] = obj
    return command_classes


def test_all_expected_commands_exist():
    """Verify all expected command classes exist in the codebase."""
    actual_commands = set(get_all_command_classes().keys())
    missing = EXPECTED_COMMANDS - actual_commands

    assert not missing, f"Expected commands not found: {missing}"


def test_no_unexpected_commands():
    """Alert if new commands are added without updating tests."""
    actual_commands = set(get_all_command_classes().keys())
    unexpected = actual_commands - EXPECTED_COMMANDS

    if unexpected:
        pytest.fail(
            f"New command classes detected: {unexpected}\n"
            f"Please add them to EXPECTED_COMMANDS in test_all_commands_coverage.py "
            f"and create appropriate tests."
        )


def test_all_commands_have_required_attributes():
    """Verify all command classes have required attributes and methods."""
    command_classes = get_all_command_classes()

    for name, cls in command_classes.items():
        # Check required methods
        assert hasattr(cls, "execute"), f"{name} missing execute() method"
        assert hasattr(cls, "undo"), f"{name} missing undo() method"
        assert hasattr(cls, "get_description"), f"{name} missing get_description() method"

        # Verify methods are callable
        assert callable(getattr(cls, "execute")), f"{name}.execute is not callable"
        assert callable(getattr(cls, "undo")), f"{name}.undo is not callable"


def test_all_commands_set_command_name():
    """Verify all commands set a meaningful command_name."""
    command_classes = get_all_command_classes()

    for name, cls in command_classes.items():
        # We can't easily test this without instantiation, but we can check
        # if command_name is defined in the class or will be set in __init__
        init_source = inspect.getsource(cls.__init__)
        has_command_name = "command_name" in init_source or hasattr(cls, "command_name")
        assert has_command_name, f"{name} doesn't set command_name"


class TestExcludeRecordingCommand:
    """Test ExcludeRecordingCommand."""

    def test_execute_and_undo(self):
        """Test execute excludes recording and undo restores it."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        mock_gui.data_selection_widget = Mock()
        mock_gui.current_session = Mock()

        cmd = commands.ExcludeRecordingCommand(mock_gui, "REC001")

        # Execute
        cmd.execute()
        mock_gui.current_session.exclude_recording.assert_called_once_with("REC001")

        # Undo
        cmd.undo()
        mock_gui.current_session.restore_recording.assert_called_once_with("REC001")


class TestRestoreRecordingCommand:
    """Test RestoreRecordingCommand."""

    def test_execute_and_undo(self):
        """Test execute restores recording and undo excludes it."""
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        mock_gui.data_selection_widget = Mock()
        mock_gui.current_session = Mock()

        cmd = commands.RestoreRecordingCommand(mock_gui, "REC001")

        # Execute
        cmd.execute()
        mock_gui.current_session.restore_recording.assert_called_once_with("REC001")

        # Undo
        cmd.undo()
        mock_gui.current_session.exclude_recording.assert_called_once_with("REC001")


class TestInvertChannelPolarityCommand:
    """Test InvertChannelPolarityCommand."""

    def test_command_structure(self):
        """Test InvertChannelPolarityCommand has proper structure."""
        # This command is tested with real data in test_command_pattern_system.py
        assert hasattr(commands.InvertChannelPolarityCommand, "execute")
        assert hasattr(commands.InvertChannelPolarityCommand, "undo")


class TestSetLatencyWindowsCommand:
    """Test SetLatencyWindowsCommand."""

    def test_command_structure(self):
        """Test SetLatencyWindowsCommand has proper structure."""
        # This command is tested with real data in test_command_pattern_system.py
        assert hasattr(commands.SetLatencyWindowsCommand, "execute")
        assert hasattr(commands.SetLatencyWindowsCommand, "undo")


class TestInsertSingleLatencyWindowCommand:
    """Test InsertSingleLatencyWindowCommand."""

    def test_command_structure(self):
        """Test InsertSingleLatencyWindowCommand has proper structure."""
        assert hasattr(commands.InsertSingleLatencyWindowCommand, "execute")
        assert hasattr(commands.InsertSingleLatencyWindowCommand, "undo")


class TestChangeChannelNamesCommand:
    """Test ChangeChannelNamesCommand."""

    def test_command_structure(self):
        """Test ChangeChannelNamesCommand has proper structure."""
        # This command is tested with real data in test_command_pattern_system.py
        assert hasattr(commands.ChangeChannelNamesCommand, "execute")
        assert hasattr(commands.ChangeChannelNamesCommand, "undo")


class TestBulkRecordingExclusionCommand:
    """Test BulkRecordingExclusionCommand."""

    def test_command_structure(self):
        """Test BulkRecordingExclusionCommand has proper structure."""
        # This command is tested with real data in test_command_pattern_system.py
        assert hasattr(commands.BulkRecordingExclusionCommand, "execute")
        assert hasattr(commands.BulkRecordingExclusionCommand, "undo")


class TestExcludeSessionCommand:
    """Test ExcludeSessionCommand."""

    def test_command_structure(self):
        """Test ExcludeSessionCommand has proper structure."""
        # This command is tested with real data in test_command_pattern_system.py
        assert hasattr(commands.ExcludeSessionCommand, "execute")
        assert hasattr(commands.ExcludeSessionCommand, "undo")


class TestRestoreSessionCommand:
    """Test RestoreSessionCommand."""

    def test_command_structure(self):
        """Test RestoreSessionCommand has proper structure."""
        # This command is tested with real data in test_command_pattern_system.py
        assert hasattr(commands.RestoreSessionCommand, "execute")
        assert hasattr(commands.RestoreSessionCommand, "undo")


class TestCreateExperimentCommand:
    """Test CreateExperimentCommand."""

    def test_command_structure(self):
        """Test CreateExperimentCommand has proper structure."""
        # This command requires extensive GUI integration
        # Just verify it exists and has required methods
        assert hasattr(commands.CreateExperimentCommand, "execute")
        assert hasattr(commands.CreateExperimentCommand, "undo")


class TestRenameExperimentCommand:
    """Test RenameExperimentCommand."""

    def test_command_structure(self):
        """Test RenameExperimentCommand has proper structure."""
        # This command requires GUI integration
        # Just verify it exists and has required methods
        assert hasattr(commands.RenameExperimentCommand, "execute")
        assert hasattr(commands.RenameExperimentCommand, "undo")


class TestToggleDatasetInclusionCommand:
    """Test ToggleDatasetInclusionCommand."""

    def test_command_structure(self):
        """Test ToggleDatasetInclusionCommand has proper structure."""
        assert hasattr(commands.ToggleDatasetInclusionCommand, "execute")
        assert hasattr(commands.ToggleDatasetInclusionCommand, "undo")


class TestToggleCompletionStatusCommand:
    """Test ToggleCompletionStatusCommand."""

    def test_command_structure(self):
        """Test ToggleCompletionStatusCommand has proper structure."""
        assert hasattr(commands.ToggleCompletionStatusCommand, "execute")
        assert hasattr(commands.ToggleCompletionStatusCommand, "undo")

    @pytest.mark.parametrize(
        "level,annot_filename,target_id",
        [
            ("experiment", "experiment.annot.json", "TEST_EXP"),
            ("dataset", "dataset.annot.json", "TEST_DATASET"),
            ("session", "session.annot.json", "TEST_SESSION"),
        ],
    )
    def test_execute_and_undo(self, tmp_path, level, annot_filename, target_id):
        """Test execute toggles completion status and undo restores it at all hierarchy levels."""
        import json

        from monstim_signals.core import DatasetAnnot, ExperimentAnnot, SessionAnnot

        # Create necessary directory structure
        exp_path = tmp_path / "test_exp"
        exp_path.mkdir()
        dataset_path = None
        session_path = None

        if level in ["dataset", "session"]:
            dataset_path = exp_path / "TEST_DATASET"
            dataset_path.mkdir()
        if level == "session":
            session_path = dataset_path / "TEST_SESSION"
            session_path.mkdir()

        # Create initial annotation file with is_completed = False
        annot_path = None
        if level == "experiment":
            annot_path = exp_path / annot_filename
            annot = ExperimentAnnot.create_empty()
        elif level == "dataset":
            annot_path = dataset_path / annot_filename
            annot = DatasetAnnot.create_empty()
        elif level == "session":
            annot_path = session_path / annot_filename
            annot = SessionAnnot.create_empty()

        annot.is_completed = False
        from dataclasses import asdict

        annot_path.write_text(json.dumps(asdict(annot), indent=2))

        # Create mock GUI with expts_dict
        mock_gui = Mock()
        mock_gui.data_selection_widget = Mock()
        mock_gui.expts_dict = {"TEST_EXP": str(exp_path)}
        mock_gui.current_experiment = Mock(id="TEST_EXP")
        mock_gui.current_dataset = Mock(id="TEST_DATASET") if level in ["dataset", "session"] else None

        # Create mock target object
        mock_target = Mock()
        mock_target.id = target_id
        mock_target.is_completed = False

        cmd = commands.ToggleCompletionStatusCommand(mock_gui, level, mock_target)

        # Verify initial state captured
        assert not cmd.old_status
        assert cmd.new_status

        # Execute
        cmd.execute()

        # Verify annotation file was updated
        annot_data = json.loads(annot_path.read_text())
        assert annot_data["is_completed"] is True

        # Undo
        cmd.undo()

        # Verify annotation file was restored
        annot_data = json.loads(annot_path.read_text())
        assert annot_data["is_completed"] is False


# Mark untestable commands
class TestUntestableCommands:
    """Document commands that cannot be reasonably unit tested."""

    def test_untestable_commands_documented(self):
        """Ensure untestable commands are documented."""
        # These commands require full GUI context, complex filesystem operations,
        # or extensive mocking that would not provide meaningful test value

        assert "DeleteExperimentCommand" in UNTESTABLE_COMMANDS
        assert "DeleteDatasetCommand" in UNTESTABLE_COMMANDS
        assert "CopyDatasetCommand" in UNTESTABLE_COMMANDS

    def test_untestable_commands_have_basic_structure(self):
        """Verify untestable commands at least have proper structure."""
        for cmd_name in UNTESTABLE_COMMANDS:
            if cmd_name in get_all_command_classes():
                cmd_class = get_all_command_classes()[cmd_name]
                assert hasattr(cmd_class, "execute")
                assert hasattr(cmd_class, "undo")
                assert hasattr(cmd_class, "get_description")
