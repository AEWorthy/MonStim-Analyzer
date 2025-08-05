"""
*** DEPRECATED ***

This file is no longer updated and is kept for reference only.
Use `monstim_signals.plotting.session_plotter.SessionPlotter` instead.

"""

import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except ImportError:  # Fallback when Qt bindings are unavailable
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from typing import TYPE_CHECKING, List  # noqa: E402

if TYPE_CHECKING:
    from monstim_signals.domain.dataset import Dataset
    from monstim_signals.domain.experiment import Experiment
    from monstim_signals.domain.session import Session


class BasePlotter:
    """
    A parent class for plotting EMG data.

    Attributes:
        emg_object: The EMG data object to be imported.
    """

    def __init__(self, emg_object):
        self.emg_object: "Session" | "Dataset" | "Experiment" = emg_object

    def set_plot_defaults(self):
        """
        Set plot font/style defaults for returned graphs.
        """
        plt.rcParams.update({"figure.titlesize": self.emg_object.title_font_size})
        plt.rcParams.update(
            {
                "figure.labelsize": self.emg_object.axis_label_font_size,
                "figure.labelweight": "bold",
            }
        )
        plt.rcParams.update(
            {
                "axes.titlesize": self.emg_object.axis_label_font_size,
                "axes.titleweight": "bold",
            }
        )
        plt.rcParams.update(
            {
                "axes.labelsize": self.emg_object.axis_label_font_size,
                "axes.labelweight": "bold",
            }
        )
        plt.rcParams.update(
            {
                "xtick.labelsize": self.emg_object.tick_font_size,
                "ytick.labelsize": self.emg_object.tick_font_size,
            }
        )

    def create_fig_and_axes(
        self,
        channel_indices: List[int] = None,
        canvas: FigureCanvas = None,
        figsizes="large",
    ):
        if channel_indices is None:
            num_channels = self.emg_object.num_channels
        else:
            num_channels = len(channel_indices)

        fig, ax, axes = (
            None,
            None,
            None,
        )  # Type: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot, numpy.ndarray
        if figsizes == "large":
            single_channel_size_tuple = (4, 2)
            multi_channel_size_tuple = (7, 2)
        elif figsizes == "small":
            single_channel_size_tuple = (3, 2)
            multi_channel_size_tuple = (5, 2)

        if canvas:
            fig = canvas.figure  # Type: matplotlib.figure.Figure
            fig.clear()
            fig.set_constrained_layout(True)

            try:
                # Use predefined size tuples
                if num_channels == 1:
                    fig.set_size_inches(*single_channel_size_tuple)
                    ax = fig.add_subplot(111)
                else:
                    fig.set_size_inches(*multi_channel_size_tuple)
                    axes = fig.subplots(nrows=1, ncols=num_channels, sharey=True)
            except ValueError as e:
                if e == "Number of columns must be a positive integer, not 0":
                    raise UnableToPlotError("Please select at least one channel to plot.")
                else:
                    raise ValueError(e)

            # Update canvas size to match figure size
            dpi = fig.get_dpi()
            width_in_inches, height_in_inches = fig.get_size_inches()
            canvas_width = int(width_in_inches * dpi)
            canvas_height = int(height_in_inches * dpi)

            # Adjust canvas size and minimum size
            canvas.setMinimumSize(canvas_width, canvas_height)
            canvas.resize(canvas_width, canvas_height)
        else:
            # Create a new figure and axes
            scale = 2  # Scale factor for figure size relative to default
            if num_channels == 1:
                fig, ax = plt.subplots(
                    figsize=tuple([item * scale for item in single_channel_size_tuple]),
                    constrained_layout=True,
                )  # Type: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
            else:
                fig, axes = plt.subplots(
                    nrows=1,
                    ncols=num_channels,
                    figsize=tuple([item * scale for item in multi_channel_size_tuple]),
                    sharey=True,
                    constrained_layout=True,
                )  # Type: matplotlib.figure.Figure, numpy.ndarray

        return (
            fig,
            ax,
            axes,
        )  # Type: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot, numpy.ndarray

    def display_plot(self, canvas: FigureCanvas):
        if canvas:
            if not canvas.figure.get_constrained_layout():
                canvas.figure.subplots_adjust(**self.emg_object.subplot_adjust_args)
            canvas.draw()
        else:
            # plt.subplots_adjust(**self.emg_object.subplot_adjust_args)
            plt.show()

    def _set_y_axis_limits(self, ax, values, fallback=1.0):
        """Set a reasonable y-axis limit based on ``values``."""

        if len(values) == 0:
            y_max = fallback
        else:
            try:
                y_max = np.nanmax(values)
            except ValueError:
                y_max = fallback
            if np.isnan(y_max) or y_max <= 0:
                y_max = fallback
        ax.set_ylim(0, 1.1 * y_max)


class UnableToPlotError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
