from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QWidget

import monstim_gui.plotting.plotting_cycler as plotting_cycler
from monstim_gui.plotting.plotting_cycler import RecordingCyclerWidget


@pytest.mark.parametrize(
    ("excluded_index", "expected_text"),
    [(0, "Include"), (None, "Exclude")],
)
def test_recording_cycler_initializes_exclusion_state(monkeypatch, qapplication, excluded_index, expected_text):
    recordings = [SimpleNamespace(id="recording-0"), SimpleNamespace(id="recording-1")]
    session = SimpleNamespace(
        all_recordings=recordings,
        num_all_recordings=len(recordings),
        excluded_recordings=[recordings[excluded_index].id] if excluded_index is not None else [],
    )
    gui = SimpleNamespace(current_session=session, plot_controller=None)
    monkeypatch.setattr(plotting_cycler, "get_main_window", lambda: gui)

    parent = QWidget()
    cycler = RecordingCyclerWidget(parent)

    assert cycler.exclude_button.text() == expected_text
