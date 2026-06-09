from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QGroupBox, QSpinBox, QWidget

pytestmark = pytest.mark.unit


@pytest.fixture
def qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_experiment_group_displays_status_and_exports_dataset_ids(qt_app):
    from monstim_gui.dialogs.bulk_export_dialog import _DatasetStatus, _ExperimentGroup

    group = _ExperimentGroup(
        "Experiment A",
        False,
        [
            _DatasetStatus(
                dataset_id="DS_FOLDER_ID",
                display_name="2024-08-16 C309.6 post-dec mcurve_long-",
                is_completed=True,
            )
        ],
    )

    group.set_dataset_mode(True)
    group._dataset_cbs[0].setChecked(True)

    assert group._expt_status_lbl.text() == "Incomplete"
    assert group._dataset_summary_lbl.text() == "1 complete, 0 incomplete"
    assert group.selected_dataset_ids == ["DS_FOLDER_ID"]


def test_completed_only_filter_limits_experiment_checkbox_selection(qt_app):
    from monstim_gui.dialogs.bulk_export_dialog import _DatasetStatus, _ExperimentGroup

    group = _ExperimentGroup(
        "Experiment A",
        True,
        [
            _DatasetStatus(dataset_id="DS_COMPLETE", display_name="Complete dataset", is_completed=True),
            _DatasetStatus(dataset_id="DS_INCOMPLETE", display_name="Incomplete dataset", is_completed=False),
            _DatasetStatus(dataset_id="DS_UNKNOWN", display_name="Unknown dataset", is_completed=None),
        ],
    )

    group._dataset_cbs[1].setChecked(True)
    group.set_completed_only(True)
    group._expt_cb.setChecked(True)

    assert group._dataset_row_by_cb[group._dataset_cbs[0]].isHidden() is False
    assert group._dataset_row_by_cb[group._dataset_cbs[1]].isHidden() is True
    assert group._dataset_row_by_cb[group._dataset_cbs[2]].isHidden() is True
    assert group._dataset_cbs[1].isChecked() is False
    assert group.selected_dataset_ids == ["DS_COMPLETE"]


def test_discover_experiment_status_reads_completion_metadata(monkeypatch, tmp_path):
    import monstim_signals.io.repositories as repos_mod
    from monstim_gui.dialogs.bulk_export_dialog import BulkExportDialog

    class FakeExperimentRepository:
        def __init__(self, folder):
            self.folder = folder

        def get_metadata(self):
            return {
                "is_completed": True,
                "excluded_datasets": ["DS2"],
                "datasets": [
                    {
                        "id": "DS1",
                        "formatted_name": "Dataset One",
                        "is_completed": True,
                    },
                    {
                        "id": "DS2",
                        "formatted_name": "Dataset Two",
                        "is_completed": False,
                    },
                ],
            }

    monkeypatch.setattr(repos_mod, "ExperimentRepository", FakeExperimentRepository)

    status = BulkExportDialog._discover_experiment_status(str(tmp_path))

    assert status.is_completed is True
    assert [ds.dataset_id for ds in status.datasets] == ["DS1", "DS2"]
    assert [ds.is_completed for ds in status.datasets] == [True, False]
    assert status.datasets[1].is_excluded is True


def test_bulk_export_dialog_does_not_expose_worker_count(qt_app):
    from monstim_gui.dialogs.bulk_export_dialog import BulkExportDialog

    gui = QWidget()
    gui.expts_dict = {}
    gui.export_path = ""
    gui.channel_names = ["Ch0"]
    gui.current_session = None
    gui.current_dataset = None
    gui.current_experiment = None

    dialog = BulkExportDialog(gui)

    group_titles = {group.title() for group in dialog.findChildren(QGroupBox)}
    assert "Export Options" in group_titles
    assert "Plot Options" not in group_titles
    assert not hasattr(dialog, "_sb_workers")
    assert dialog.findChildren(QSpinBox) == []
