import json
from pathlib import Path

import pytest

from monstim_gui.commands import (
    CommandInvoker,
    DeleteDatasetCommand,
    ToggleDatasetInclusionCommand,
)


def make_experiment_with_dataset(root: Path, exp_name: str, ds_name: str):
    exp_path = root / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)
    (exp_path / "experiment.annot.json").write_text(json.dumps({}))
    ds_path = exp_path / ds_name
    ds_path.mkdir(parents=True, exist_ok=True)
    (ds_path / "dataset.annot.json").write_text(json.dumps({}))
    return exp_path, ds_path


@pytest.mark.usefixtures("temp_output_dir")
class TestDatasetInclusionAndDelete:
    def test_toggle_dataset_inclusion_and_undo(self, fake_gui, temp_output_dir: Path):
        make_experiment_with_dataset(temp_output_dir, "ExpX", "DS1")
        fake_gui.data_manager.unpack_existing_experiments()
        fake_gui.data_selection_widget.refresh()
        inv = CommandInvoker(fake_gui)

        cmd = ToggleDatasetInclusionCommand(fake_gui, "ExpX", "DS1", exclude=True)
        inv.execute(cmd)

        # Verify annot updated
        annot = json.loads((temp_output_dir / "ExpX" / "experiment.annot.json").read_text())
        assert "DS1" in annot.get("excluded_datasets", [])

        # Undo inclusion
        inv.undo()
        annot = json.loads((temp_output_dir / "ExpX" / "experiment.annot.json").read_text())
        assert "excluded_datasets" not in annot or "DS1" not in annot.get("excluded_datasets", [])

    def test_delete_dataset_command_irreversible(self, fake_gui, temp_output_dir: Path, monkeypatch):
        make_experiment_with_dataset(temp_output_dir, "ExpY", "DS2")
        fake_gui.data_manager.unpack_existing_experiments()
        fake_gui.data_selection_widget.refresh()
        inv = CommandInvoker(fake_gui)

        messages = {}
        import monstim_gui.commands as cmds

        def fake_warning(gui, title, text):
            messages["title"] = title
            messages["text"] = text
            return None

        monkeypatch.setattr(cmds.QMessageBox, "warning", fake_warning)

        cmd = DeleteDatasetCommand(fake_gui, "DS2", "DS2", "ExpY")
        inv.execute(cmd)
        assert not (temp_output_dir / "ExpY" / "DS2").exists()

        inv.undo()
        assert messages.get("title") == "Cannot Undo Deletion"
        assert "cannot be restored" in messages.get("text", "")
