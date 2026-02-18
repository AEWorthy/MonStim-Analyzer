"""
Tests for EditDatasetMetadataCommand.

These tests verify that:
1. Metadata-only changes work correctly
2. Metadata + folder rename works correctly
3. Undo properly reverts both metadata and folder name
4. UI refresh is triggered after execute and undo
5. Error handling works correctly
"""

from pathlib import Path
from unittest.mock import Mock, PropertyMock

import pytest

from monstim_gui.commands import EditDatasetMetadataCommand


class TestEditDatasetMetadataCommand:
    """Test suite for EditDatasetMetadataCommand."""

    def test_metadata_only_no_folder_rename(self):
        """Test editing metadata without renaming the folder."""
        # Setup mocks
        mock_gui = Mock()
        mock_gui.data_selection_widget = Mock()

        mock_dataset = Mock()
        mock_dataset.annot = Mock()
        mock_dataset.annot.date = "2024-01-15"
        mock_dataset.annot.animal_id = "C123"
        mock_dataset.annot.condition = "control"
        mock_dataset.repo = Mock()

        # Create command to change metadata only (no folder rename)
        cmd = EditDatasetMetadataCommand(
            gui=mock_gui,
            dataset=mock_dataset,
            old_date="2024-01-15",
            new_date="2024-01-16",
            old_animal_id="C123",
            new_animal_id="C123",
            old_condition="control",
            new_condition="test",
            old_folder_name=None,
            new_folder_name=None,
        )

        # Execute
        cmd.execute()

        # Verify metadata was updated
        assert mock_dataset.annot.date == "2024-01-16"
        assert mock_dataset.annot.animal_id == "C123"
        assert mock_dataset.annot.condition == "test"

        # Verify repo.save was called
        mock_dataset.repo.save.assert_called_once_with(mock_dataset)

        # Verify repo.rename was NOT called (no folder rename)
        mock_dataset.repo.rename.assert_not_called()

        # Verify UI was refreshed
        mock_gui.data_selection_widget.update.assert_called_once_with(levels=("dataset", "session"))

        # Reset mocks for undo
        mock_dataset.repo.save.reset_mock()
        mock_gui.data_selection_widget.update.reset_mock()

        # Undo
        cmd.undo()

        # Verify metadata was reverted
        assert mock_dataset.annot.date == "2024-01-15"
        assert mock_dataset.annot.animal_id == "C123"
        assert mock_dataset.annot.condition == "control"

        # Verify repo.save was called again
        mock_dataset.repo.save.assert_called_once_with(mock_dataset)

        # Verify UI was refreshed again
        mock_gui.data_selection_widget.update.assert_called_once_with(levels=("dataset", "session"))

    def test_metadata_with_folder_rename(self):
        """Test editing metadata that requires folder rename."""
        # Setup mocks
        mock_gui = Mock()
        mock_gui.data_selection_widget = Mock()

        mock_dataset = Mock()
        mock_dataset.annot = Mock()
        mock_dataset.annot.date = "2024-01-15"
        mock_dataset.annot.animal_id = "C123"
        mock_dataset.annot.condition = "control"
        mock_dataset.id = "240115 C123 control"
        mock_dataset.repo = Mock()

        # Mock the folder path - need to properly mock Path behavior
        old_folder = Mock(spec=Path)
        old_folder.name = "240115 C123 control"
        parent_folder = Mock(spec=Path)

        # Mock the / operator for parent / folder_name
        def parent_div(self_arg, folder_name):
            new_path = Mock(spec=Path)
            new_path.name = folder_name
            new_path.exists.return_value = False
            return new_path

        parent_folder.__truediv__ = parent_div
        old_folder.parent = parent_folder

        type(mock_dataset.repo).folder = PropertyMock(return_value=old_folder)

        # Create command with folder rename
        cmd = EditDatasetMetadataCommand(
            gui=mock_gui,
            dataset=mock_dataset,
            old_date="2024-01-15",
            new_date="2024-01-16",
            old_animal_id="C123",
            new_animal_id="C456",
            old_condition="control",
            new_condition="test",
            old_folder_name="240115 C123 control",
            new_folder_name="240116 C456 test",
        )

        # Execute
        cmd.execute()

        # Verify metadata was updated
        assert mock_dataset.annot.date == "2024-01-16"
        assert mock_dataset.annot.animal_id == "C456"
        assert mock_dataset.annot.condition == "test"

        # Verify repo.rename was called with new folder name
        mock_dataset.repo.rename.assert_called_once()
        call_args = mock_dataset.repo.rename.call_args
        # First argument should be the new folder path
        assert call_args[0][0].name == "240116 C456 test"
        # dataset parameter should be passed
        assert call_args[1]["dataset"] == mock_dataset

        # Verify repo.save was called
        mock_dataset.repo.save.assert_called_once_with(mock_dataset)

        # Verify UI was refreshed
        mock_gui.data_selection_widget.update.assert_called_once_with(levels=("dataset", "session"))

    def test_folder_already_exists_error(self):
        """Test that FileExistsError is raised if target folder already exists."""
        # Setup mocks
        mock_gui = Mock()
        mock_gui.data_selection_widget = Mock()

        mock_dataset = Mock()
        mock_dataset.annot = Mock()
        mock_dataset.annot.date = "2024-01-15"
        mock_dataset.annot.animal_id = "C123"
        mock_dataset.annot.condition = "control"
        mock_dataset.repo = Mock()

        # Mock the folder path
        old_folder = Mock(spec=Path)
        old_folder.name = "240115 C123 control"
        parent_folder = Mock(spec=Path)

        # Mock the / operator to return a path that exists
        def parent_div(self_arg, folder_name):
            new_path = Mock(spec=Path)
            new_path.name = folder_name
            new_path.exists.return_value = True  # Target already exists
            return new_path

        parent_folder.__truediv__ = parent_div
        old_folder.parent = parent_folder

        type(mock_dataset.repo).folder = PropertyMock(return_value=old_folder)

        # Create command with folder rename
        cmd = EditDatasetMetadataCommand(
            gui=mock_gui,
            dataset=mock_dataset,
            old_date="2024-01-15",
            new_date="2024-01-16",
            old_animal_id="C123",
            new_animal_id="C456",
            old_condition="control",
            new_condition="test",
            old_folder_name="240115 C123 control",
            new_folder_name="240116 C456 test",
        )

        # Execute should raise FileExistsError
        with pytest.raises(FileExistsError, match="Target folder already exists"):
            cmd.execute()

    def test_undo_with_folder_rename(self):
        """Test that undo properly reverts folder rename."""
        # Setup mocks
        mock_gui = Mock()
        mock_gui.data_selection_widget = Mock()

        mock_dataset = Mock()
        mock_dataset.annot = Mock()
        mock_dataset.id = "240116 C456 test"
        mock_dataset.repo = Mock()

        # After execute, folder has been renamed to new name
        new_folder = Mock(spec=Path)
        new_folder.name = "240116 C456 test"
        parent_folder = Mock(spec=Path)

        # Mock the / operator for parent / folder_name
        def parent_div(self_arg, folder_name):
            path = Mock(spec=Path)
            path.name = folder_name
            path.exists.return_value = False
            return path

        parent_folder.__truediv__ = parent_div
        new_folder.parent = parent_folder

        type(mock_dataset.repo).folder = PropertyMock(return_value=new_folder)

        # Create command
        cmd = EditDatasetMetadataCommand(
            gui=mock_gui,
            dataset=mock_dataset,
            old_date="2024-01-15",
            new_date="2024-01-16",
            old_animal_id="C123",
            new_animal_id="C456",
            old_condition="control",
            new_condition="test",
            old_folder_name="240115 C123 control",
            new_folder_name="240116 C456 test",
        )

        # Simulate state after execute (metadata already changed)
        mock_dataset.annot.date = "2024-01-16"
        mock_dataset.annot.animal_id = "C456"
        mock_dataset.annot.condition = "test"

        # Undo
        cmd.undo()

        # Verify metadata was reverted
        assert mock_dataset.annot.date == "2024-01-15"
        assert mock_dataset.annot.animal_id == "C123"
        assert mock_dataset.annot.condition == "control"

        # Verify repo.rename was called to rename back
        mock_dataset.repo.rename.assert_called_once()
        call_args = mock_dataset.repo.rename.call_args
        # Should rename back to old folder name
        assert call_args[0][0].name == "240115 C123 control"
        assert call_args[1]["dataset"] == mock_dataset

        # Verify repo.save was called
        mock_dataset.repo.save.assert_called_once_with(mock_dataset)

        # Verify UI was refreshed
        mock_gui.data_selection_widget.update.assert_called_once_with(levels=("dataset", "session"))

    def test_no_repo_metadata_only(self):
        """Test that command works when dataset has no repository (metadata-only mode)."""
        # Setup mocks
        mock_gui = Mock()
        mock_gui.data_selection_widget = Mock()

        mock_dataset = Mock()
        mock_dataset.annot = Mock()
        mock_dataset.annot.date = "2024-01-15"
        mock_dataset.annot.animal_id = "C123"
        mock_dataset.annot.condition = "control"
        mock_dataset.repo = None  # No repository

        # Create command (no folder rename when repo is None)
        cmd = EditDatasetMetadataCommand(
            gui=mock_gui,
            dataset=mock_dataset,
            old_date="2024-01-15",
            new_date="2024-01-16",
            old_animal_id="C123",
            new_animal_id="C456",
            old_condition="control",
            new_condition="test",
            old_folder_name=None,
            new_folder_name=None,
        )

        # Execute should work without errors
        cmd.execute()

        # Verify metadata was updated
        assert mock_dataset.annot.date == "2024-01-16"
        assert mock_dataset.annot.animal_id == "C456"
        assert mock_dataset.annot.condition == "test"

        # UI should still be refreshed
        mock_gui.data_selection_widget.update.assert_called_once_with(levels=("dataset", "session"))

    def test_command_name_generation(self):
        """Test that command name is generated correctly."""
        mock_gui = Mock()
        mock_dataset = Mock()

        # Test with all fields changed
        cmd = EditDatasetMetadataCommand(
            gui=mock_gui,
            dataset=mock_dataset,
            old_date="2024-01-15",
            new_date="2024-01-16",
            old_animal_id="C123",
            new_animal_id="C456",
            old_condition="control",
            new_condition="test",
        )
        assert "date: 2024-01-15 → 2024-01-16" in cmd.command_name
        assert "ID: C123 → C456" in cmd.command_name
        assert "condition: control → test" in cmd.command_name

        # Test with only date changed
        cmd = EditDatasetMetadataCommand(
            gui=mock_gui,
            dataset=mock_dataset,
            old_date="2024-01-15",
            new_date="2024-01-16",
            old_animal_id="C123",
            new_animal_id="C123",
            old_condition="test",
            new_condition="test",
        )
        assert "date: 2024-01-15 → 2024-01-16" in cmd.command_name
        assert "ID:" not in cmd.command_name
        assert "condition:" not in cmd.command_name

    def test_get_description(self):
        """Test that get_description returns a meaningful description."""
        mock_gui = Mock()
        mock_dataset = Mock()

        cmd = EditDatasetMetadataCommand(
            gui=mock_gui,
            dataset=mock_dataset,
            old_date="2024-01-15",
            new_date="2024-01-16",
            old_animal_id="C123",
            new_animal_id="C456",
            old_condition="control",
            new_condition="test",
        )

        description = cmd.get_description()
        assert "date to 2024-01-16" in description
        assert "ID to C456" in description
        assert "condition to test" in description

    def test_access_denied_error_handling(self):
        """Test that OSError with EACCES is properly handled."""
        import errno

        # Setup mocks
        mock_gui = Mock()
        mock_gui.data_selection_widget = Mock()

        mock_dataset = Mock()
        mock_dataset.annot = Mock()
        mock_dataset.repo = Mock()

        # Mock the folder path
        old_folder = Mock(spec=Path)
        old_folder.name = "240115 C123 control"
        parent_folder = Mock(spec=Path)

        # Mock the / operator
        def parent_div(self_arg, folder_name):
            new_path = Mock(spec=Path)
            new_path.name = folder_name
            new_path.exists.return_value = False
            return new_path

        parent_folder.__truediv__ = parent_div
        old_folder.parent = parent_folder

        type(mock_dataset.repo).folder = PropertyMock(return_value=old_folder)

        # Make repo.rename raise access denied error
        access_error = OSError("Access denied")
        access_error.errno = errno.EACCES
        mock_dataset.repo.rename.side_effect = access_error

        # Create command with folder rename
        cmd = EditDatasetMetadataCommand(
            gui=mock_gui,
            dataset=mock_dataset,
            old_date="2024-01-15",
            new_date="2024-01-16",
            old_animal_id="C123",
            new_animal_id="C456",
            old_condition="control",
            new_condition="test",
            old_folder_name="240115 C123 control",
            new_folder_name="240116 C456 test",
        )

        # Execute should raise OSError with helpful message
        with pytest.raises(OSError, match="Cannot rename folder - it is in use"):
            cmd.execute()

    def test_ui_refresh_without_data_selection_widget(self):
        """Test that command doesn't fail if GUI has no data_selection_widget."""
        # Setup mocks - GUI without data_selection_widget
        mock_gui = Mock(spec=[])  # Empty spec means no attributes

        mock_dataset = Mock()
        mock_dataset.annot = Mock()
        mock_dataset.annot.date = "2024-01-15"
        mock_dataset.annot.animal_id = "C123"
        mock_dataset.annot.condition = "control"
        mock_dataset.repo = Mock()

        # Create command
        cmd = EditDatasetMetadataCommand(
            gui=mock_gui,
            dataset=mock_dataset,
            old_date="2024-01-15",
            new_date="2024-01-16",
            old_animal_id="C123",
            new_animal_id="C123",
            old_condition="control",
            new_condition="test",
        )

        # Execute should not raise error even without data_selection_widget
        cmd.execute()

        # Verify metadata was still updated
        assert mock_dataset.annot.date == "2024-01-16"
        mock_dataset.repo.save.assert_called_once_with(mock_dataset)
