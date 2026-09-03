from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QSizePolicy, QToolButton, QWidget

import monstim_gui.plotting.plotting_cycler as plotting_cycler
from monstim_gui.plotting.plot_options import OptionToggleButton
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


def test_recording_cycler_matches_option_control_height_and_theme(monkeypatch, qapplication):
    session = SimpleNamespace(all_recordings=[SimpleNamespace(id="recording-0")], num_all_recordings=1, excluded_recordings=[])
    gui = SimpleNamespace(current_session=session, plot_controller=None)
    monkeypatch.setattr(plotting_cycler, "get_main_window", lambda: gui)

    parent = QWidget()
    cycler = RecordingCyclerWidget(parent)

    reference_button = OptionToggleButton("Show Flags", "Show flags.")
    expected_height = reference_button.sizeHint().height()
    assert all(
        control.height() == expected_height
        for control in (
            cycler.prev_button,
            cycler.next_button,
            cycler.exclude_button,
            cycler.recording_spinbox,
            cycler.step_size,
        )
    )
    assert isinstance(cycler.prev_button, QToolButton)
    assert isinstance(cycler.next_button, QToolButton)
    assert not cycler.prev_button.icon().isNull()
    assert not cycler.next_button.icon().isNull()
    assert cycler.prev_button.iconSize() == QSize(16, 16)
    assert cycler.next_button.iconSize() == QSize(16, 16)
    assert cycler.prev_button.sizePolicy().horizontalPolicy() is QSizePolicy.Policy.Expanding
    assert cycler.next_button.sizePolicy().horizontalPolicy() is QSizePolicy.Policy.Expanding
    assert cycler.layout.contentsMargins().top() == 0
