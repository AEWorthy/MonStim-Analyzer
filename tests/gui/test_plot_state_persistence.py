"""Regression coverage for persisted plot-control state."""

from types import SimpleNamespace

from PySide6.QtWidgets import QWidget

from monstim_gui.plotting.plotting_widget import PlotWidget


class _Gui(QWidget):
    def __init__(self):
        super().__init__()
        self.current_session = SimpleNamespace(num_channels=3)
        self.current_dataset = SimpleNamespace(num_channels=3)
        self.current_experiment = SimpleNamespace(num_channels=3)
        self.plot_controller = SimpleNamespace(plot_data=lambda: None, get_raw_data=lambda: None)


def test_plot_widget_restores_valid_state_and_clamps_channels(qapplication, monkeypatch):
    saved_state = {
        "version": 1,
        "view": "session",
        "last_plot_type": {"session": "Single EMG Recordings"},
        "last_options": {
            "session": {
                "Single EMG Recordings": {
                    "data_type": "filtered",
                    "channel_indices": [1, 4],
                    "fixed_y_axis": False,
                }
            }
        },
        "channel_selection": [1, 4],
    }
    monkeypatch.setattr("monstim_gui.plotting.plotting_widget.app_state.get_plot_state", lambda: saved_state)
    persisted = []
    monkeypatch.setattr("monstim_gui.plotting.plotting_widget.app_state.save_plot_state", persisted.append)

    gui = _Gui()
    widget = PlotWidget(gui)
    gui.plot_widget = widget
    widget.initialize_plot_widget()
    assert widget.current_option_widget.data_type_combo.currentText() == "filtered"
    qapplication.processEvents()

    assert widget.view == "session"
    assert widget.plot_type_combo.currentText() == "Single EMG Recordings"
    assert widget.current_option_widget.data_type_combo.currentText() == "filtered"
    assert widget.current_option_widget.channel_selector.get_selected_channels() == [1]

    widget.current_option_widget.fixed_y_axis_checkbox.setChecked(True)

    assert persisted[-1]["view"] == "session"
    assert persisted[-1]["last_options"]["session"]["Single EMG Recordings"]["fixed_y_axis"] is True


def test_plot_widget_ignores_removed_plot_type(qapplication, monkeypatch):
    monkeypatch.setattr(
        "monstim_gui.plotting.plotting_widget.app_state.get_plot_state",
        lambda: {"version": 1, "view": "dataset", "last_plot_type": {"dataset": "Removed Plot"}, "last_options": {}},
    )
    monkeypatch.setattr("monstim_gui.plotting.plotting_widget.app_state.save_plot_state", lambda state: None)

    gui = _Gui()
    widget = PlotWidget(gui)
    gui.plot_widget = widget
    widget.initialize_plot_widget()
    qapplication.processEvents()

    assert widget.view == "dataset"
    assert widget.plot_type_combo.currentText() == "Average Reflex:Stimulus Curves"


def test_plot_types_keep_independent_options_and_channel_selection(qapplication, monkeypatch):
    monkeypatch.setattr("monstim_gui.plotting.plotting_widget.app_state.get_plot_state", lambda: {})
    monkeypatch.setattr("monstim_gui.plotting.plotting_widget.app_state.save_plot_state", lambda state: None)

    gui = _Gui()
    widget = PlotWidget(gui)
    gui.plot_widget = widget
    widget.initialize_plot_widget()

    widget.plot_type_combo.setCurrentText("Single EMG Recordings")
    single = widget.current_option_widget
    single.data_type_combo.setCurrentText("raw")
    single.channel_selector.set_selected_channels([1])
    assert widget.last_options["session"]["EMG"]["data_type"] == "filtered"
    assert widget.last_options["session"]["Single EMG Recordings"]["data_type"] == "raw"

    widget.plot_type_combo.setCurrentText("EMG")
    emg = widget.current_option_widget
    assert emg.data_type_combo.currentText() == "filtered"
    assert emg.channel_selector.get_selected_channels() == [0, 1, 2]
    emg.channel_selector.set_selected_channels([2])

    widget.plot_type_combo.setCurrentText("Single EMG Recordings")
    restored_single = widget.current_option_widget
    assert restored_single.data_type_combo.currentText() == "raw"
    assert restored_single.channel_selector.get_selected_channels() == [1]
