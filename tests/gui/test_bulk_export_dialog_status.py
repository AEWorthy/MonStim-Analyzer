from __future__ import annotations

import pytest
from PySide6.QtWidgets import QGroupBox, QSpinBox, QWidget

pytestmark = pytest.mark.unit


def test_experiment_group_displays_status_and_exports_dataset_ids():
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


def test_completed_only_filter_limits_experiment_checkbox_selection():
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


def test_bulk_export_dialog_does_not_expose_worker_count():
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


def test_recording_exclusion_editor_does_not_crash_when_stimulus_filter_disabled():
    from monstim_gui.dialogs.recording_exclusion_editor import RecordingExclusionEditor

    class DummyRecording:
        def __init__(self):
            self.id = "rec-1"
            self.stim_amplitude = 7.5
            self.num_channels = 1
            self.channel_types = ["emg"]

    class DummySession:
        def __init__(self):
            self.id = "sess-1"
            self.excluded_recordings = set()

        def get_all_recordings(self, include_excluded=True):
            return [DummyRecording()]

    parent_widget = QWidget()
    parent_widget.current_session = DummySession()
    parent_widget.current_dataset = None
    parent_widget.current_experiment = None
    parent_widget.status_bar = None

    dialog = RecordingExclusionEditor(parent_widget)
    recording = DummyRecording()

    dialog.stimulus_group.setChecked(False)

    evaluation = dialog._evaluation_for_recording(recording, parent_widget.current_session, {}, set())
    assert evaluation["flagged"] is False


def test_recording_exclusion_preview_is_explicit_and_skips_quality_work_for_stimulus_rules(monkeypatch):
    from monstim_gui.dialogs.recording_exclusion_editor import RecordingExclusionEditor

    class DummyRecording:
        def __init__(self):
            self.id = "rec-1"
            self.stim_amplitude = 7.5
            self.num_channels = 1
            self.channel_types = ["emg"]

    class DummySession:
        def __init__(self):
            self.id = "sess-1"
            self.excluded_recordings = set()

        def get_all_recordings(self, include_excluded=True):
            return [DummyRecording()]

    parent_widget = QWidget()
    parent_widget.current_session = DummySession()
    parent_widget.current_dataset = None
    parent_widget.current_experiment = None
    parent_widget.status_bar = None

    dialog = RecordingExclusionEditor(parent_widget)
    quality_calls = 0

    assert dialog.threshold2_spinbox.isHidden()
    assert dialog.threshold2_label.isHidden()
    dialog.threshold_type_combo.setCurrentIndex(dialog.threshold_type_combo.findData("outside"))
    assert not dialog.threshold2_spinbox.isHidden()
    assert not dialog.threshold2_label.isHidden()
    dialog.threshold_type_combo.setCurrentIndex(dialog.threshold_type_combo.findData("above"))

    def unexpected_quality_work(*_args, **_kwargs):
        nonlocal quality_calls
        quality_calls += 1
        raise AssertionError("stimulus-only preview must not calculate quality metrics")

    monkeypatch.setattr(dialog, "compute_quality_metrics", unexpected_quality_work)
    dialog.stimulus_group.setChecked(True)

    assert dialog._preview_is_stale
    assert not dialog.apply_button.isEnabled()
    assert not dialog.threshold_spinbox.keyboardTracking()

    dialog.update_preview()

    assert quality_calls == 0
    assert not dialog._preview_is_stale

    dialog.recordings_table.selectRow(0)
    assert dialog.recordings_table.selectionModel().selectedRows()
    dialog._clear_table_selection()
    assert not dialog.recordings_table.selectionModel().selectedRows()

    dialog.threshold_spinbox.editingFinished.emit()

    assert not dialog._preview_is_stale
    assert not dialog.apply_button.isEnabled()


def test_recording_exclusion_apply_sets_busy_before_focus_commit_without_repreview(monkeypatch, qapplication):
    from unittest.mock import Mock

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    import monstim_gui.dialogs.recording_exclusion_editor as editor_module

    class DummyRecording:
        def __init__(self):
            self.id = "rec-1"
            self.stim_amplitude = 7.5
            self.num_channels = 1
            self.channel_types = ["emg"]

    class DummySession:
        def __init__(self):
            self.id = "sess-1"
            self.excluded_recordings = set()

        def get_all_recordings(self, include_excluded=True):
            return [DummyRecording()]

    preview_calls = 0
    original_update_preview = editor_module.RecordingExclusionEditor.update_preview

    def tracked_update_preview(self):
        nonlocal preview_calls
        preview_calls += 1
        return original_update_preview(self)

    monkeypatch.setattr(editor_module.RecordingExclusionEditor, "update_preview", tracked_update_preview)
    feedback_cursors = []

    def decline_confirmation(*_args, **_kwargs):
        feedback_cursors.append(QApplication.overrideCursor())
        return editor_module.QMessageBox.StandardButton.No

    monkeypatch.setattr(editor_module.QMessageBox, "question", decline_confirmation)

    parent_widget = QWidget()
    parent_widget.current_session = DummySession()
    parent_widget.current_dataset = None
    parent_widget.current_experiment = None
    parent_widget.command_invoker = Mock()
    parent_widget.status_bar = Mock()

    dialog = editor_module.RecordingExclusionEditor(parent_widget)
    dialog.show()
    dialog.stimulus_group.setChecked(True)
    dialog.update_preview()
    baseline_preview_calls = preview_calls
    cursor_during_focus_commit = []
    dialog.threshold_spinbox.editingFinished.connect(
        lambda: cursor_during_focus_commit.append(QApplication.overrideCursor().shape() if QApplication.overrideCursor() else None)
    )
    dialog.threshold_spinbox.setFocus()
    qapplication.processEvents()

    QTest.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)

    assert cursor_during_focus_commit == [Qt.CursorShape.WaitCursor]
    assert preview_calls == baseline_preview_calls
    assert feedback_cursors == [None]
    assert QApplication.overrideCursor() is None

    dialog.threshold_spinbox.setValue(100.0)
    dialog.threshold_spinbox.editingFinished.emit()

    assert preview_calls == baseline_preview_calls + 1
