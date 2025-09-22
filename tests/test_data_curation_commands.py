from pathlib import Path

import pytest

from monstim_gui.commands import (
    CommandInvoker,
    CopyDatasetCommand,
    CreateExperimentCommand,
    DeleteExperimentCommand,
    MoveDatasetCommand,
    RenameExperimentCommand,
)


def make_experiment_with_dataset(root: Path, exp_name: str, ds_name: str, sessions: list[str] = None):
    """Create a minimal experiment folder with a dataset and empty session folders and annotations."""
    sessions = sessions or ["S01", "S02"]
    exp_path = root / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)
    # experiment annotation
    (exp_path / "experiment.annot.json").write_text("{}")

    ds_path = exp_path / ds_name
    ds_path.mkdir(parents=True, exist_ok=True)
    # dataset annotation (minimal)
    (ds_path / "dataset.annot.json").write_text("{}")
    for s in sessions:
        (ds_path / s).mkdir(parents=True, exist_ok=True)
        # session annot optional for metadata scan
        # (ds_path / s / "session.annot.json").write_text("{}")
    return exp_path, ds_path


@pytest.mark.usefixtures("temp_output_dir")
class TestDataCurationCommands:
    def test_create_and_undo_experiment(self, fake_gui, temp_output_dir: Path):
        inv = CommandInvoker(fake_gui)
        cmd = CreateExperimentCommand(fake_gui, "ExpA")
        inv.execute(cmd)

        assert "ExpA" in fake_gui.expts_dict
        assert (temp_output_dir / "ExpA" / "experiment.annot.json").exists()

        inv.undo()
        assert "ExpA" not in fake_gui.expts_dict
        assert not (temp_output_dir / "ExpA").exists()

    def test_rename_experiment_and_undo(self, fake_gui, temp_output_dir: Path):
        # seed one experiment
        make_experiment_with_dataset(temp_output_dir, "OldExp", "240101 A1 cond")
        fake_gui.data_manager.unpack_existing_experiments()
        fake_gui.data_selection_widget.refresh()
        inv = CommandInvoker(fake_gui)

        cmd = RenameExperimentCommand(fake_gui, "OldExp", "NewExp")
        inv.execute(cmd)
        assert "NewExp" in fake_gui.expts_dict and "OldExp" not in fake_gui.expts_dict
        assert (temp_output_dir / "NewExp").exists()

        inv.undo()
        assert "OldExp" in fake_gui.expts_dict and "NewExp" not in fake_gui.expts_dict
        assert (temp_output_dir / "OldExp").exists()

    def test_move_dataset_and_undo(self, fake_gui, temp_output_dir: Path):
        # Create two experiments and one dataset in source
        make_experiment_with_dataset(temp_output_dir, "Exp1", "240101 A1 cond")
        make_experiment_with_dataset(temp_output_dir, "Exp2", "240102 A2 cond")
        fake_gui.data_manager.unpack_existing_experiments()
        fake_gui.data_selection_widget.refresh()

        inv = CommandInvoker(fake_gui)
        cmd = MoveDatasetCommand(
            fake_gui, dataset_id="240101 A1 cond", dataset_name="240101 A1 cond", from_exp="Exp1", to_exp="Exp2"
        )
        inv.execute(cmd)
        assert not (temp_output_dir / "Exp1" / "240101 A1 cond").exists()
        assert (temp_output_dir / "Exp2" / "240101 A1 cond").exists()

        inv.undo()
        assert (temp_output_dir / "Exp1" / "240101 A1 cond").exists()
        assert not (temp_output_dir / "Exp2" / "240101 A1 cond").exists()

    def test_copy_dataset_and_undo(self, fake_gui, temp_output_dir: Path):
        make_experiment_with_dataset(temp_output_dir, "Exp1", "240101 A1 cond")
        make_experiment_with_dataset(temp_output_dir, "Exp2", "240102 A2 cond")
        fake_gui.data_manager.unpack_existing_experiments()
        fake_gui.data_selection_widget.refresh()

        inv = CommandInvoker(fake_gui)
        cmd = CopyDatasetCommand(
            fake_gui, dataset_id="240101 A1 cond", dataset_name="240101 A1 cond", from_exp="Exp1", to_exp="Exp2"
        )
        inv.execute(cmd)

        # A new folder should appear in Exp2; could be same name or suffixed with _copyN
        exp2_folders = [p.name for p in (temp_output_dir / "Exp2").iterdir() if p.is_dir()]
        assert any("240101 A1 cond" in name for name in exp2_folders)

        # Undo should remove the copied folder
        created = cmd.copied_folder_name
        assert created is not None
        assert (temp_output_dir / "Exp2" / created).exists()
        inv.undo()
        assert not (temp_output_dir / "Exp2" / created).exists()

    def test_delete_experiment_is_irreversible(self, fake_gui, temp_output_dir: Path, monkeypatch):
        # Setup experiment then delete via command
        make_experiment_with_dataset(temp_output_dir, "ExpDel", "240101 A1 cond")
        fake_gui.data_manager.unpack_existing_experiments()
        fake_gui.data_selection_widget.refresh()
        inv = CommandInvoker(fake_gui)

        # Monkeypatch QMessageBox.warning to avoid GUI interaction during tests
        import monstim_gui.commands as cmds

        messages = {}

        def fake_warning(gui, title, text):
            messages["title"] = title
            messages["text"] = text
            return None

        monkeypatch.setattr(cmds.QMessageBox, "warning", fake_warning)

        cmd = DeleteExperimentCommand(fake_gui, "ExpDel")
        inv.execute(cmd)
        assert "ExpDel" not in fake_gui.expts_dict
        assert not (temp_output_dir / "ExpDel").exists()

        # Undo should not restore, but should display a warning instead (captured above)
        inv.undo()
        assert messages.get("title") == "Cannot Undo Deletion"
        assert "cannot be restored" in messages.get("text", "")
