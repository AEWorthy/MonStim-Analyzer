from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QWidget

from monstim_gui.dialogs.recording_exclusion_editor import RecordingExclusionEditor


class DummyRecording:
    def __init__(self, recording_id):
        self.id = recording_id
        self.stim_amplitude = 7.5
        self.num_channels = 1
        self.channel_types = ["emg"]


class DummySession:
    id = "session-1"

    def __init__(self):
        self.recordings = [DummyRecording("rec-1"), DummyRecording("rec-2")]
        self.excluded_recordings = set()

    def get_all_recordings(self, include_excluded=True):
        return self.recordings


def make_editor():
    parent = QWidget()
    parent.current_session = DummySession()
    parent.current_dataset = None
    parent.current_experiment = None
    parent.status_bar = None
    return RecordingExclusionEditor(parent)


def test_manual_include_and_exclude_preserve_selected_recordings(qapplication):
    editor = make_editor()
    editor.recordings_table.selectAll()
    assert len(editor._selected_entries()) == 2

    QTest.mouseClick(editor.toggle_exclusion_button, Qt.MouseButton.LeftButton)
    qapplication.processEvents()

    assert set(editor.manual_decisions.values()) == {True}
    assert len(editor.manual_decisions) == 2

    editor.recordings_table.selectAll()
    QTest.mouseClick(editor.include_button, Qt.MouseButton.LeftButton)
    qapplication.processEvents()

    assert set(editor.manual_decisions.values()) == {False}
    assert len(editor.manual_decisions) == 2


def test_automatic_preview_preserves_exclusion_added_after_dialog_open(qapplication):
    editor = make_editor()
    editor.current_session.excluded_recordings.add("rec-1")

    editor.update_preview()

    states = {entry["recording"].id: entry for entry in editor._last_recordings_data}
    assert states["rec-1"]["currently_excluded"] is True
    assert states["rec-1"]["will_exclude"] is True
    assert states["rec-1"]["status"] == "Existing exclusion"
