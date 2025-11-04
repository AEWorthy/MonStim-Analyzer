import pytest
from unittest.mock import Mock

from monstim_gui.commands import RestoreDatasetCommand


def test_restore_dataset_command_execute_and_undo():
    # Prepare mock GUI and experiment with a restored dataset present
    mock_gui = Mock()
    mock_gui.data_selection_widget = Mock()
    mock_gui.plot_widget = Mock()

    # Create a fake dataset object that will be present after restore
    restored_ds = Mock()
    restored_ds.id = "DS1"
    restored_ds.sessions = [Mock()]

    # current_experiment should have restore_dataset and datasets list
    mock_gui.current_experiment = Mock()
    mock_gui.current_experiment.datasets = [restored_ds]
    mock_gui.current_experiment.restore_dataset = Mock()
    mock_gui.current_experiment.exclude_dataset = Mock()

    # Ensure current_dataset is initially None
    mock_gui.current_dataset = None

    cmd = RestoreDatasetCommand(mock_gui, "DS1")

    # Execute should call restore_dataset and set current_dataset
    cmd.execute()
    mock_gui.current_experiment.restore_dataset.assert_called_once_with("DS1")
    assert mock_gui.current_dataset is restored_ds
    # Data selection widget updated for dataset list
    mock_gui.data_selection_widget.update.assert_called()

    # Undo should exclude the dataset and clear current selection
    cmd.undo()
    mock_gui.current_experiment.exclude_dataset.assert_called_once_with("DS1")
    assert mock_gui.current_dataset is None
    # Undo should call update for dataset and session levels
    mock_gui.data_selection_widget.update.assert_called_with(levels=("dataset", "session"))


if __name__ == "__main__":
    pytest.main([__file__])
