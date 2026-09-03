from types import SimpleNamespace

from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from monstim_gui.plotting.plot_options import ChannelSelectorWidget, EMGOptions, StableOptionGrid


class _PlotParent(QWidget):
    def __init__(self, gui_main):
        super().__init__()
        self._gui_main = gui_main

    @property
    def parent(self):
        return self._gui_main


def test_channel_selector_uses_natural_horizontal_width(qapplication):
    gui_main = SimpleNamespace(
        plot_widget=SimpleNamespace(view="session"),
        current_session=SimpleNamespace(num_channels=6),
    )
    parent = QWidget()
    layout = QVBoxLayout(parent)
    selector = ChannelSelectorWidget(gui_main, parent)
    layout.addWidget(selector)

    parent.resize(600, selector.sizeHint().height())
    parent.show()
    qapplication.processEvents()

    assert selector.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Preferred
    assert selector.width() < parent.width()
    assert all(cb.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed for cb in selector.checkboxes)


def test_plot_option_grid_packs_sparse_options_in_canonical_order(qapplication):
    grid = StableOptionGrid()
    legend = grid.add_option("show_legend", "Show Legend", "Show the plot legend.")
    relative_to_mmax = grid.add_option("relative_to_mmax", "Relative to M-max", "Normalize to M-max.")

    grid.resize(360, 120)
    qapplication.processEvents()

    assert grid.grid.itemAtPosition(0, 0).widget() is legend
    assert grid.grid.itemAtPosition(1, 0).widget() is relative_to_mmax
    assert grid.grid.itemAtPosition(0, 1) is None
    assert legend.isCheckable()
    assert legend.property("plotOptionToggle") is True
    assert legend.height() >= legend.fontMetrics().height()


def test_plot_option_grid_uses_three_columns_at_wide_width(qapplication):
    grid = StableOptionGrid()
    flags = grid.add_option("show_flags", "Show Flags", "Show flags.")
    legend = grid.add_option("show_legend", "Show Legend", "Show the legend.")
    colormap = grid.add_option("show_colormap", "Show Colormap", "Show the colormap.")
    cursor = grid.add_option("interactive_cursor", "Interactive Cursor", "Show the cursor.")

    grid.resize(500, 120)
    grid.show()
    qapplication.processEvents()

    assert grid.grid.itemAtPosition(0, 0).widget() is flags
    assert grid.grid.itemAtPosition(0, 1).widget() is legend
    assert grid.grid.itemAtPosition(0, 2).widget() is colormap
    assert grid.grid.itemAtPosition(1, 0).widget() is cursor


def test_emg_option_toggle_preserves_flag_legend_dependency(qapplication):
    gui_main = SimpleNamespace(
        plot_widget=SimpleNamespace(view="session"),
        current_session=SimpleNamespace(num_channels=2),
    )
    parent = _PlotParent(gui_main)
    options = EMGOptions(parent)

    options.all_windows_checkbox.setChecked(False)
    assert not options.latency_legend_checkbox.isEnabled()
    assert not options.latency_legend_checkbox.isChecked()

    options.all_windows_checkbox.setChecked(True)
    assert options.latency_legend_checkbox.isEnabled()
    assert options.latency_legend_checkbox.isChecked()
