import pytest
from unittest.mock import Mock

from monstim_gui.commands import MoveDatasetCommand, MoveDatasetsCommand


def test_move_dataset_command_execute_and_undo():
    mock_gui = Mock()
    mock_gui.data_manager = Mock()
    # Provide a mock Data Curation Manager so the command will call load_data()
    mock_gui._data_curation_manager = Mock()

    cmd = MoveDatasetCommand(mock_gui, "ds1", "Dataset One", "expA", "expB")

    # Execute should call data_manager.move_dataset with forward args
    cmd.execute()
    mock_gui.data_manager.move_dataset.assert_called_once_with("ds1", "Dataset One", "expA", "expB")
    mock_gui._data_curation_manager.load_data.assert_called_once()

    # Undo should move back (to_exp -> from_exp)
    cmd.undo()
    # Two calls total now: one for execute, one for undo
    assert mock_gui.data_manager.move_dataset.call_count == 2
    mock_gui.data_manager.move_dataset.assert_called_with("ds1", "Dataset One", "expB", "expA")
    # load_data should also have been called again
    assert mock_gui._data_curation_manager.load_data.call_count == 2


def test_move_datasets_command_execute_and_undo_all_succeed():
    mock_gui = Mock()
    mock_gui.data_manager = Mock()
    mock_gui._data_curation_manager = Mock()

    moves = [
        ("ds1", "Dataset One", "expA", "expB"),
        ("ds2", "Dataset Two", "expC", "expB"),
    ]

    cmd = MoveDatasetsCommand(mock_gui, moves)

    # Execute should call move_dataset for each item in the same order
    cmd.execute()
    calls = mock_gui.data_manager.move_dataset.call_args_list
    assert len(calls) == 2
    assert calls[0].args == ("ds1", "Dataset One", "expA", "expB")
    assert calls[1].args == ("ds2", "Dataset Two", "expC", "expB")

    # Batched command should record succeeded moves
    assert hasattr(cmd, "_succeeded")
    assert len(cmd._succeeded) == 2

    # Data curation manager refreshed once after execute
    mock_gui._data_curation_manager.load_data.assert_called_once()

    # Reset call history then undo
    mock_gui.data_manager.move_dataset.reset_mock()
    cmd.undo()

    # Undo should move back succeeded items in reverse order
    undo_calls = mock_gui.data_manager.move_dataset.call_args_list
    assert len(undo_calls) == 2
    # first undo call should correspond to reversing the second move
    assert undo_calls[0].args == ("ds2", "Dataset Two", "expB", "expC")
    assert undo_calls[1].args == ("ds1", "Dataset One", "expB", "expA")

    # Data curation manager refreshed once after undo
    mock_gui._data_curation_manager.load_data.assert_called()


def test_move_datasets_command_partial_failure():
    mock_gui = Mock()
    # Create a data_manager.move_dataset that fails on second call
    def side_effect(ds_id, ds_name, from_exp, to_exp):
        if ds_id == "ds2":
            raise Exception("disk error")
        return None

    mock_gui.data_manager = Mock()
    mock_gui.data_manager.move_dataset.side_effect = side_effect
    mock_gui._data_curation_manager = Mock()

    moves = [
        ("ds1", "Dataset One", "expA", "expB"),
        ("ds2", "Dataset Two", "expC", "expB"),
        ("ds3", "Dataset Three", "expD", "expB"),
    ]

    cmd = MoveDatasetsCommand(mock_gui, moves)

    # execute should complete without raising, and record only successful moves
    cmd.execute()
    assert len(cmd._succeeded) == 2  # ds1 and ds3 should succeed

    # Undo should only attempt to move back the succeeded ones (in reverse order)
    mock_gui.data_manager.move_dataset.reset_mock()
    cmd.undo()
    undo_calls = mock_gui.data_manager.move_dataset.call_args_list
    # Should have two undo calls for ds3 then ds1
    assert len(undo_calls) == 2
    assert undo_calls[0].args == ("ds3", "Dataset Three", "expB", "expD")
    assert undo_calls[1].args == ("ds1", "Dataset One", "expB", "expA")


if __name__ == "__main__":
    pytest.main([__file__])
