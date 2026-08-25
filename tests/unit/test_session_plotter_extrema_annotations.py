from types import SimpleNamespace

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from monstim_signals.plotting.session_plotter_pyqtgraph import SessionPlotterPyQtGraph
from monstim_signals.transform.extrema import make_window_span


def _result(window_index: int, maximum_index: int, minimum_index: int):
    return SimpleNamespace(
        window_index=window_index,
        window_name=f"Window {window_index}",
        amplitude=1.0,
        priority_rank=window_index,
        selected_max=SimpleNamespace(sample_index=maximum_index, value=1.0),
        selected_min=SimpleNamespace(sample_index=minimum_index, value=-1.0),
    )


def _plotter(results, spans=()):
    plotter = object.__new__(SessionPlotterPyQtGraph)
    plotter.emg_object = SimpleNamespace(
        scan_rate=1000,
        stim_start=0,
        num_samples=100,
        latency_windows=[
            SimpleNamespace(name=f"Window {index}", color=color) for index, color in enumerate(("red", "blue", "green", "orange", "purple"))
        ],
        get_recording_lw_amplitude_results=lambda *_args: results,
        _window_spans=lambda _channel_index: spans,
    )
    plotter.extrema_items = []
    return plotter


def _render(results, method, spans=()):
    QApplication.instance() or QApplication([])
    plot = pg.PlotItem()
    plotter = _plotter(results, spans)
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


def test_independent_marker_shows_windows_that_contain_an_extremum_even_when_only_one_selected_it():
    spans = (
        make_window_span(0, "Window 0", 0, 20, 1000),
        make_window_span(1, "Window 1", 0, 20, 1000),
    )
    items = _render([_result(0, 10, 20)], "extrema_ptt", spans)

    # Each selected extremum is contained by both windows, so its marker has
    # red and blue sectors plus the white circular outline.
    assert len(items) == 6
    assert items[0].opts["brush"].color().name() == "#ff0000"
    assert items[1].opts["brush"].color().name() == "#0000ff"


def test_independent_marker_does_not_show_an_out_of_bounds_window_as_an_owner():
    spans = (
        make_window_span(0, "Window 0", 0, 20, 1000),
        make_window_span(1, "Window 1", 0, 200, 1000),
    )
    items = _render([_result(0, 10, 20)], "extrema_ptt", spans)

    # Window 1 geometrically covers these extrema but the backend rejects it,
    # so the red selected marker must not acquire an erroneous blue sector.
    assert len(items) == 2
    assert [item.opts["symbol"] for item in items] == ["t", "t1"]
