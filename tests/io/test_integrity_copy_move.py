from __future__ import annotations

import json
from pathlib import Path

import pytest

from monstim_gui.commands import CommandInvoker, CopyDatasetCommand, MoveDatasetCommand
from tests.helpers import list_files_with_hashes, make_experiment_and_dataset_with_nested_content

"""
Test Annotations
- Purpose: Ensure copy/move commands preserve nested file integrity and metadata with undo support
- Markers: integration (filesystem operations via commands)
- Notes: Uses helpers for deterministic content and hash validation
"""
pytestmark = pytest.mark.integration


@pytest.mark.usefixtures("temp_output_dir")
class TestIntegrityCopyMove:
    def test_copy_preserves_all_content_and_metadata(self, fake_gui, temp_output_dir: Path):
        # Prepare source with nested content
        make_experiment_and_dataset_with_nested_content(temp_output_dir, "Exp1", "DS-Alpha")
        # Destination experiment exists
        (temp_output_dir / "Exp2" / "experiment.annot.json").parent.mkdir(parents=True, exist_ok=True)
        (temp_output_dir / "Exp2" / "experiment.annot.json").write_text(json.dumps({}))
        fake_gui.data_manager.unpack_existing_experiments()
        fake_gui.data_selection_widget.refresh()

        src_ds = temp_output_dir / "Exp1" / "DS-Alpha"
        before_hashes = list_files_with_hashes(src_ds)

        inv = CommandInvoker(fake_gui)
        cmd = CopyDatasetCommand(fake_gui, dataset_id="DS-Alpha", dataset_name="DS-Alpha", from_exp="Exp1", to_exp="Exp2")
        inv.execute(cmd)

        created = cmd.copied_folder_name
        assert created is not None
        dst_ds = temp_output_dir / "Exp2" / created
        assert dst_ds.exists()

        # 1) Verify metadata fields preserved in dataset.annot.json
        dst_meta = json.loads((dst_ds / "dataset.annot.json").read_text())
        assert dst_meta.get("is_completed") is True
        assert dst_meta.get("data_version") == "2025.09-test"
        assert dst_meta.get("animal_id") == "A1"
        assert dst_meta.get("date") == "250101"
        assert dst_meta.get("condition") == "cond-X"
        # excluded_sessions should carry over
        assert isinstance(dst_meta.get("excluded_sessions", []), list)

        # 2) Verify nested file integrity (same set and same size+hashes)
        after_hashes = list_files_with_hashes(dst_ds)
        assert set(before_hashes.keys()) == set(after_hashes.keys())
        for rel, (sz, md5) in before_hashes.items():
            assert rel in after_hashes
            assert after_hashes[rel][0] == sz
            assert after_hashes[rel][1] == md5

        # Undo should remove the copied folder
        inv.undo()
        assert not dst_ds.exists()

    def test_move_preserves_all_content_and_metadata(self, fake_gui, temp_output_dir: Path):
        # Prepare two experiments and one dataset with nested content under Exp1
        make_experiment_and_dataset_with_nested_content(temp_output_dir, "Exp1", "DS-Bravo")
        (temp_output_dir / "Exp2" / "experiment.annot.json").parent.mkdir(parents=True, exist_ok=True)
        (temp_output_dir / "Exp2" / "experiment.annot.json").write_text(json.dumps({}))
        fake_gui.data_manager.unpack_existing_experiments()
        fake_gui.data_selection_widget.refresh()

        src_ds = temp_output_dir / "Exp1" / "DS-Bravo"
        before_hashes = list_files_with_hashes(src_ds)

        inv = CommandInvoker(fake_gui)
        cmd = MoveDatasetCommand(fake_gui, dataset_id="DS-Bravo", dataset_name="DS-Bravo", from_exp="Exp1", to_exp="Exp2")
        inv.execute(cmd)

        # After move, source gone and dest exists with identical content
        assert not src_ds.exists()
        dst_ds = temp_output_dir / "Exp2" / "DS-Bravo"
        assert dst_ds.exists()

        dst_meta = json.loads((dst_ds / "dataset.annot.json").read_text())
        assert dst_meta.get("data_version") == "2025.09-test"
        after_hashes = list_files_with_hashes(dst_ds)
        assert set(before_hashes.keys()) == set(after_hashes.keys())
        for rel, (sz, md5) in before_hashes.items():
            assert after_hashes[rel] == (sz, md5)

        # Undo should move it back intact
        inv.undo()
        assert (temp_output_dir / "Exp1" / "DS-Bravo").exists()
        moved_back_hashes = list_files_with_hashes(temp_output_dir / "Exp1" / "DS-Bravo")
        assert moved_back_hashes == before_hashes
