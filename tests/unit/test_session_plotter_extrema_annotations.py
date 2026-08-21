from types import SimpleNamespace

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from monstim_signals.plotting.session_plotter_pyqtgraph import SessionPlotterPyQtGraph


def _result(window_index: int, maximum_index: int, minimum_index: int):
    return SimpleNamespace(
        window_index=window_index,
        window_name=f"Window {window_index}",
        amplitude=1.0,
        priority_rank=window_index,
        selected_max=SimpleNamespace(sample_index=maximum_index, value=1.0),
        selected_min=SimpleNamespace(sample_index=minimum_index, value=-1.0),
    )


def _plotter(results):
    plotter = object.__new__(SessionPlotterPyQtGraph)
    plotter.emg_object = SimpleNamespace(
        scan_rate=1000,
        stim_start=0,
        latency_windows=[SimpleNamespace(color=color) for color in ("red", "blue", "green", "orange", "purple")],
        get_recording_lw_amplitude_results=lambda *_args: results,
    )
    plotter.extrema_items = []
    return plotter


def _render(results, method):
    QApplication.instance() or QApplication([])
    plot = pg.PlotItem()
    plotter = _plotter(results)
    plotter._plot_extrema_annotations(plot, 0, "recording", method, labels=False)
    return plotter.extrema_items


def test_independent_shared_extremum_uses_two_color_circular_marker():
    items = _render([_result(0, 10, 20), _result(1, 10, 30)], "extrema_ptt")

    # Two pie sectors and their shared circular outline, plus two unshared minima.
    assert len(items) == 5
    assert items[0].opts["symbol"].elementCount() > 0
    assert items[1].opts["symbol"].elementCount() > 0
    assert items[2].opts["symbol"] == "o"


def test_independent_more_than_four_owners_uses_neutral_circle():
    items = _render([_result(index, 10, 20 + index) for index in range(5)], "extrema_ptt")

    # The shared maximum is one neutral circle; each unshared minimum remains a triangle.
    assert len(items) == 6
    assert items[0].opts["symbol"] == "o"
    assert items[0].opts["brush"].color().name() == "#808080"


def test_exclusive_method_keeps_triangle_markers():
    items = _render([_result(0, 10, 20), _result(1, 10, 30)], "exclusive_extrema_ptt")

    assert len(items) == 3
    assert [item.opts["symbol"] for item in items] == ["t", "t1", "t1"]
