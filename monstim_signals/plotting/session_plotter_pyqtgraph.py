import logging
from contextlib import suppress
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.graphicsItems.LegendItem import ItemSample
from pyqtgraph.graphicsItems.PlotItem import PlotItem
from pyqtgraph.Qt import QtGui

from .base_plotter_pyqtgraph import BasePlotterPyQtGraph, UnableToPlotError

if TYPE_CHECKING:
    from monstim_gui.plotting import PlotPane
    from monstim_signals.domain.session import Session

logger = logging.getLogger(__name__)

_MAX_MIXED_EXTREMA_WINDOWS = 4


def _pie_wedge_symbol(start_angle: float, span_angle: float) -> QtGui.QPainterPath:
    """Return a unit-circle sector for a multi-window extrema marker."""
    path = QtGui.QPainterPath()
    path.moveTo(0, 0)
    path.arcTo(-0.5, -0.5, 1, 1, start_angle, span_angle)
    path.closeSubpath()
    return path


class _LatencyWindowLegendSample(ItemSample):
    """Legend sample that toggles every flag line for one latency window."""

    def __init__(self, item: pg.PlotDataItem, flag_items: list[pg.InfiniteLine]):
        super().__init__(item)
        self.flag_items = flag_items

    def mouseClickEvent(self, event):
        if event.button() == pg.QtCore.Qt.MouseButton.LeftButton:
            # Treat a partially visible group as visible so one click hides it.
            show = not any(item.isVisible() for item in self.flag_items)
            for item in self.flag_items:
                item.setVisible(show)
            self.item.setVisible(show)
            self.update()

        event.accept()


class SessionPlotterPyQtGraph(BasePlotterPyQtGraph):
    """
    PyQtGraph-based plotter for Session data with interactive features.

    This class provides interactive plotting capabilities for EMG session data:
    - Real-time zooming and panning
    - Interactive latency window selection
    - Optional crosshair cursor for measurements (can be disabled for exports)
    - Multi-channel plotting support

    All plotting methods support an 'interactive_cursor' parameter to control
    whether crosshair cursors are enabled (True by default).
    """

    def __init__(self, emg_object: Session):
        super().__init__(emg_object)
        self.emg_object: Session = emg_object
        # Shared colormap for both data and colorbar. Get viridis_r from matplotlib
        self.stim_colormap = pg.colormap.get("viridis_r", source="matplotlib")
        # Store references to plotted curves for dynamic updates
        self.plotted_curves: list[pg.PlotDataItem] = []
        self.curve_stimulus_voltages: list[float] = []
        self.colorbar_item: pg.ColorBarItem = None
        # Start/end InfiniteLine items grouped by latency-window index.  A
        # legend entry uses this mapping so it controls the actual flags,
        # rather than only its dummy legend sample.
        self.latency_window_flag_items: dict[int, list[pg.InfiniteLine]] = {}
        self.extrema_items: list[object] = []
        # Brightness adjustment for colormap (0.0 = no adjustment, 0.5 = much brighter)
        # For viridis_r: higher values make colors brighter by avoiding dark end of colormap
        self.brightness_shift = 0

    def get_time_axis(self):
        """Get time axis for plotting."""
        # Calculate time values based on the scan rate
        time_values_ms = np.arange(self.emg_object.num_samples) * 1000 / self.emg_object.scan_rate  # Time values in milliseconds

        # Define the start and end times for the window
        # Start pre_stim_time ms before stimulus onset
        window_start_time = self.emg_object.stim_start - self.emg_object.pre_stim_time_ms
        # End time_window_ms after stimulus onset (not affected by pre_stim_time)
        window_end_time = self.emg_object.stim_start + self.emg_object.time_window_ms

        # Convert time window to sample indices
        window_start_sample = int(window_start_time * self.emg_object.scan_rate / 1000)
        window_end_sample = int(window_end_time * self.emg_object.scan_rate / 1000)

        # Ensure we don't have negative indexing issues that could cause confusion
        # If the calculated start is negative, it means we're asking for data before recording started
        if window_start_sample < 0:
            logger.warning(f"Requested time window starts before recording began. window_start_sample={window_start_sample}, adjusting to 0.")
            window_start_sample = 0

        # Ensure end sample doesn't exceed available data
        if window_end_sample > self.emg_object.num_samples:
            logger.warning(
                f"Requested time window extends beyond recording. window_end_sample={window_end_sample}, clamping to {self.emg_object.num_samples}."
            )
            window_end_sample = self.emg_object.num_samples

        # Slice the time array for the time window
        time_axis = time_values_ms[window_start_sample:window_end_sample] - self.emg_object.stim_start

        return time_axis, window_start_sample, window_end_sample

    def _clear_extrema_items(self):
        for item in self.extrema_items:
            with suppress(AttributeError, RuntimeError):
                item.scene().removeItem(item)
        self.extrema_items.clear()

    def _plot_extrema_annotations(self, plot: PlotItem, channel_index: int, recording_id: str, method: str, *, labels: bool):
        """Draw domain-calculated filtered-trace extrema without recalculation here."""
        try:
            results = self.emg_object.get_recording_lw_amplitude_results(method, channel_index, recording_id)
        except Exception as exc:
            logger.warning("Could not calculate extrema annotations: %s", exc)
            return
        windows = self.emg_object.latency_windows
        try:
            spans = self.emg_object._window_spans(channel_index)
        except AttributeError, TypeError, ValueError:
            # Lightweight plotter doubles do not expose Session's span API.
            # Their selected result ownership remains the best available view.
            spans = ()
        grouped: dict[tuple[int, str], list[tuple[object, object]]] = {}
        for result in results:
            if result.selected_max is not None and result.selected_min is not None:
                for extremum, kind in ((result.selected_max, "max"), (result.selected_min, "min")):
                    grouped.setdefault((extremum.sample_index, kind), []).append((result, extremum))
        for (sample_index, kind), selections in grouped.items():
            result, extremum = selections[0]
            time_ms = sample_index * 1000 / self.emg_object.scan_rate - self.emg_object.stim_start
            selected_indices = {chosen.window_index for chosen, _item in selections}
            owner_indices = (
                [
                    span.window_index
                    for span in spans
                    if span.start_sample >= 0
                    and span.end_sample <= self.emg_object.num_samples
                    and span.end_sample - span.start_sample >= 3
                    and span.start_sample <= sample_index <= span.end_sample
                ]
                if method == "extrema_ptt" and spans
                else [chosen.window_index for chosen, _item in selections]
            )
            selected_details = "; ".join(
                f"{chosen.window_name} {kind}: PTT={chosen.amplitude:.4g}, priority={chosen.priority_rank}" for chosen, _item in selections
            )
            containment = ", ".join(windows[index].name for index in owner_indices)
            tooltip = f"{time_ms:.3f} ms, {extremum.value:.4g}; contained by: {containment}; selected for PTT: {selected_details}"
            marker_items: list[pg.ScatterPlotItem]
            if method == "extrema_ptt" and len(owner_indices) > 1:
                if len(owner_indices) <= _MAX_MIXED_EXTREMA_WINDOWS:
                    span_angle = 360 / len(owner_indices)
                    marker_items = [
                        pg.ScatterPlotItem(
                            [time_ms],
                            [extremum.value],
                            symbol=_pie_wedge_symbol(index * span_angle, span_angle),
                            size=10,
                            brush=pg.mkBrush(self._convert_matplotlib_color(windows[window_index].color)),
                            pen=pg.mkPen(None),
                        )
                        for index, window_index in enumerate(owner_indices)
                    ]
                    marker_items.append(
                        pg.ScatterPlotItem([time_ms], [extremum.value], symbol="o", size=10, brush=None, pen=pg.mkPen("w", width=1.2))
                    )
                else:
                    marker_items = [
                        pg.ScatterPlotItem(
                            [time_ms], [extremum.value], symbol="o", size=10, brush=pg.mkBrush("#808080"), pen=pg.mkPen("w", width=1.2)
                        )
                    ]
            else:
                display_window_index = next((index for index in owner_indices if index in selected_indices), result.window_index)
                marker_items = [
                    pg.ScatterPlotItem(
                        [time_ms],
                        [extremum.value],
                        symbol="t" if kind == "max" else "t1",
                        size=10,
                        brush=pg.mkBrush(self._convert_matplotlib_color(windows[display_window_index].color)),
                        pen=pg.mkPen("w", width=1.2),
                    )
                ]
            for marker in marker_items:
                marker.setToolTip(tooltip)
                plot.addItem(marker)
                self.extrema_items.append(marker)
            if labels:
                label = pg.TextItem(
                    f"{', '.join(windows[index].name for index in owner_indices)} {kind}\n{time_ms:.2f} ms",
                    color="w",
                    anchor=(0.5, 1.2),
                )
                label.setPos(time_ms, extremum.value)
                plot.addItem(label, ignoreBounds=True)
                self.extrema_items.append(label)

    def get_emg_recordings(self, data_type, use_all=False) -> list[np.ndarray]:
        """
        Get the EMG recordings based on the specified data type.

        Parameters
        ----------
        data_type : str
            Type of data to retrieve ('filtered', 'raw', 'rectified_raw', 'rectified_filtered')
        use_all : bool
            If False (default), returns only active/non-excluded recordings.
            If True, returns all recordings including excluded ones ('all_recordings_<data_type>').

        Returns
        -------
        list[np.ndarray]
            list of EMG recordings corresponding to the specified data type and exclusion criteria.

        """

        if data_type in ["raw", "filtered", "rectified_raw", "rectified_filtered"]:
            prefix = "all_" if use_all else ""
            attribute_name = f"{prefix}recordings_{data_type}"
            data = getattr(self.emg_object, attribute_name)
            if data is None:
                raise AttributeError(
                    f"Data type '{attribute_name}' is not available in the Session object."
                    f" Please ensure that the data has been processed and stored correctly."
                )
            return data
        else:
            raise ValueError(f"Data type '{data_type}' is not supported. Please use 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered'.")

    def plot_channel_data(
        self,
        plot_item: PlotItem,
        time_axis: np.ndarray,
        channel_data: np.ndarray,
        start: int,
        end: int,
        stimulus_v: float,
        channel_index: int,
        norm=None,
    ):
        """
        Plot channel data with stimulus voltage-based coloring and PyQtGraph optimizations.

        Parameters match the original matplotlib plotter interface.
        """
        # Get data segment
        data_segment = channel_data[start:end]
        time_segment = time_axis

        # Create color based on stimulus voltage using shared colormap
        if norm is not None:
            color_value = norm(stimulus_v)
            color = self.stim_colormap.map(color_value, mode="qcolor")
        else:
            # Use default color cycling
            color = self.default_colors[channel_index % len(self.default_colors)]

        # Plot via base helper (adds optional decimation and pyqtgraph optimizations)
        curve = self.plot_time_series(
            plot_item,
            time_segment,
            data_segment,
            color=color,
            line_width=1.5,
        )

        curve.setClipToView(True)  # Optimize rendering by clipping to view

        # Store curve reference for dynamic colormap updates
        if norm is not None:
            self.plotted_curves.append(curve)
            self.curve_stimulus_voltages.append(stimulus_v)

        # Set title and grid
        plot_item.setTitle(f"{self.emg_object.channel_names[channel_index]}")
        plot_item.showGrid(True, True)

        return curve

    def plot_latency_windows(self, plot_item: PlotItem, all_flags: bool, channel_index: int):
        """
        Plot latency windows as vertical lines.

        Parameters match the original matplotlib plotter interface.
        """
        if all_flags:
            for window_index, window in enumerate(self.emg_object.latency_windows):
                # Convert matplotlib color to PyQtGraph-compatible color
                color = self._convert_matplotlib_color(window.color)

                # Create vertical lines for window start and end
                start_line = pg.InfiniteLine(
                    pos=window.start_times[channel_index],
                    angle=90,
                    pen=pg.mkPen(
                        color=color,
                        style=self._get_line_style(window.linestyle),
                        width=2,
                    ),
                )
                end_line = pg.InfiniteLine(
                    pos=window.end_times[channel_index],
                    angle=90,
                    pen=pg.mkPen(
                        color=color,
                        style=self._get_line_style(window.linestyle),
                        width=2,
                    ),
                )

                plot_item.addItem(start_line)
                plot_item.addItem(end_line)
                self.latency_window_flag_items.setdefault(window_index, []).extend((start_line, end_line))

    def add_latency_window_legend(self, plot_item: PlotItem):
        """
        Add a native PyQtGraph legend to the plot item for latency windows.
        """
        legend = plot_item.addLegend(offset=(10, 10))
        # Use a short horizontal line as a dummy item for each latency window
        for window_index, window in enumerate(self.emg_object.latency_windows):
            color = self._convert_matplotlib_color(window.color)
            pen = pg.mkPen(color=color, style=self._get_line_style(window.linestyle), width=2)
            # Create a dummy PlotDataItem (short horizontal line)
            x = np.array([0, 1])
            y = np.array([0, 0])
            dummy_item = pg.PlotDataItem(x, y, pen=pen)
            sample = _LatencyWindowLegendSample(dummy_item, self.latency_window_flag_items.get(window_index, []))
            legend.addItem(sample, window.label if hasattr(window, "label") else str(window))
        return legend

    def _get_line_style(self, matplotlib_style):
        """Convert matplotlib line style to PyQtGraph line style."""
        style_map = {
            "-": pg.QtCore.Qt.PenStyle.SolidLine,
            "--": pg.QtCore.Qt.PenStyle.DashLine,
            ":": pg.QtCore.Qt.PenStyle.DotLine,
            "-.": pg.QtCore.Qt.PenStyle.DashDotLine,
        }
        return style_map.get(matplotlib_style, pg.QtCore.Qt.PenStyle.SolidLine)

    def get_brightness_adjusted_norm(self, min_val: float | None = None, max_val: float | None = None):
        """
        Create a normalization function that shifts values to brighter colors.

        For viridis_r colormap: 0.0 = bright (yellow), 1.0 = dark (purple)
        So we map values to [0.0, 1.0 - brightness_shift] to prefer brighter colors.

        Parameters
        ----------
        min_val : float, optional
            Minimum stimulus voltage. If None, uses min of self.emg_object.stimulus_voltages
        max_val : float, optional
            Maximum stimulus voltage. If None, uses max of self.emg_object.stimulus_voltages

        Returns
        -------
        callable
            Normalization function that maps stimulus voltages to [0.0, 1.0 - brightness_shift] range
        """
        if min_val is None:
            min_val = min(self.emg_object.stimulus_voltages)
        if max_val is None:
            max_val = max(self.emg_object.stimulus_voltages)

        def norm(v):
            if max_val != min_val:
                # Standard normalization to 0-1 range
                normalized = (v - min_val) / (max_val - min_val)
                # For viridis_r: map to [0.0, 1.0 - brightness_shift] to prefer brighter colors
                return normalized * (1.0 - self.brightness_shift)
            else:
                return 0.3  # Use a bright default when all values are the same

        return norm

    def add_colormap_scalebar(
        self,
        layout: pg.GraphicsLayout,
        plot_items: list[PlotItem],
        value_range: tuple[float, float],
    ):
        brightness_limit = np.clip(1.0 - self.brightness_shift, 0.0, 1.0)
        if brightness_limit < 1.0:
            # Keep the colormap's input domain aligned with the curve normalization.
            positions = np.linspace(0.0, brightness_limit, 256)
            colors = self.stim_colormap.map(positions, mode="byte")
            colorbar_colormap = pg.ColorMap(
                np.append(positions, 1.0),
                np.vstack((colors, colors[-1])),
            )
        else:
            colorbar_colormap = self.stim_colormap

        colorbar = pg.ColorBarItem(
            colorMap=colorbar_colormap,
            values=value_range,
            label="Stimulus Voltage (V)",
            orientation="vertical",
            interactive=False,
            colorMapMenu=False,
        )
        # Store reference to colorbar for dynamic updates
        self.colorbar_item = colorbar

        # Connect colorbar signals to update curve colors
        self.connect_colorbar_signals()

        # Add to the right of the last plot (assuming one row)
        layout.addItem(colorbar, row=0, col=len(plot_items))

    def plot_emg(
        self,
        channel_indices: list[int] | None = None,
        all_flags: bool = True,
        plot_legend: bool = True,
        plot_colormap: bool = False,
        data_type: str = "filtered",
        interactive_cursor: bool = True,
        show_extrema_labels: bool = False,
        extrema_label_method: str = "exclusive_extrema_ptt",
        collect_raw_data: bool = True,
        canvas: PlotPane = None,
    ):
        """
        Plot EMG data with interactive features and performance optimizations.

        Parameters
        ----------
        channel_indices : list[int], optional
            list of channel indices to plot
        all_flags : bool, optional
            Whether to show all latency windows (default: True)
        plot_legend : bool, optional
            Whether to show legend (default: True)
        plot_colormap : bool, optional
            Whether to show colormap (default: False)
        data_type : str, optional
            Type of data to plot (default: 'filtered')
        interactive_cursor : bool, optional
            Whether to enable interactive crosshair cursor (default: True)
        collect_raw_data : bool, optional
            Whether to build and return the tabular plot-data export.  The GUI
            disables this for ordinary rendering because the table is not used.
        canvas : PlotPane, optional
            Canvas to plot on

        Returns
        -------
        pd.DataFrame
            Raw data used for plotting
        """
        if canvas is None:
            raise UnableToPlotError("Canvas must be provided for PyQtGraph plotting")
        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))

        # Clear previous curve references for new plot
        self.clear_curve_references()
        self._clear_extrema_items()

        time_axis, window_start_sample, window_end_sample = self.get_time_axis()

        # Create plot layout
        plot_items: list[PlotItem]
        layout: pg.GraphicsLayout
        plot_items, layout = self.create_plot_layout(canvas, channel_indices)

        # Track Y-axis bounds across all channels as we plot
        overall_y_min = float("inf")
        overall_y_max = float("-inf")

        # Add synchronized crosshairs to all plots (if enabled)
        if interactive_cursor:
            self.add_synchronized_crosshairs(plot_items)

        # Plot latency windows once per channel (not per recording)
        self.latency_window_flag_items.clear()
        for plot_idx, channel_index in enumerate(channel_indices):
            current_plot = plot_items[plot_idx]
            self.plot_latency_windows(current_plot, all_flags, channel_index)

        # Get EMG recordings
        emg_recordings = self.get_emg_recordings(data_type)

        # Check for empty recordings
        if not emg_recordings:
            raise UnableToPlotError("No active recordings available to plot in this session. All recordings may have been excluded.")

        # Create normalization for stimulus voltages (matching matplotlib version)
        norm = self.get_brightness_adjusted_norm()

        # Initialize raw data collection (matching matplotlib version exactly)
        raw_data_dict = (
            {
                "recording_index": [],
                "channel_index": [],
                "stimulus_V": [],
                "time_point": [],
                "amplitude_mV": [],
            }
            if collect_raw_data
            else None
        )

        # Plot each recording (matching matplotlib version logic)
        for recording_idx, recording in enumerate(emg_recordings):
            stimulus_v = self.emg_object.stimulus_voltages[recording_idx]

            # Process each channel (matching matplotlib version)
            for channel_index, channel_data in enumerate(recording.T):
                if channel_index not in channel_indices:
                    continue

                # Get the current plot item
                plot_idx = channel_indices.index(channel_index)
                current_plot = plot_items[plot_idx]

                # Plot EMG data with PyQtGraph built-in optimizations
                self.plot_channel_data(
                    current_plot,
                    time_axis,
                    channel_data,
                    window_start_sample,
                    window_end_sample,
                    stimulus_v,
                    channel_index,
                    norm=norm,
                )
                if show_extrema_labels and data_type == "filtered":
                    self._plot_extrema_annotations(
                        current_plot, channel_index, self.emg_object.recordings[recording_idx].id, extrema_label_method, labels=False
                    )

                data_segment = channel_data[window_start_sample:window_end_sample]
                if raw_data_dict is not None:
                    # Store the original (non-downsampled) data only for an
                    # explicit data-export request.
                    num_points = len(time_axis)
                    raw_data_dict["recording_index"].extend([recording_idx] * num_points)
                    raw_data_dict["channel_index"].extend([channel_index] * num_points)
                    raw_data_dict["stimulus_V"].extend([stimulus_v] * num_points)
                    raw_data_dict["time_point"].extend(time_axis)
                    raw_data_dict["amplitude_mV"].extend(data_segment)

                # Track min/max for Y-axis scaling across ALL channels
                valid_data = data_segment[~(np.isnan(data_segment) | np.isinf(data_segment))]
                if len(valid_data) > 0:
                    overall_y_min = min(overall_y_min, np.min(valid_data))
                    overall_y_max = max(overall_y_max, np.max(valid_data))

        # Set labels for each plot
        for plot_idx, _channel_index in enumerate(channel_indices):
            current_plot: PlotItem = plot_items[plot_idx]
            current_plot.setLabel("bottom", "Time (ms)")
            current_plot.setLabel("left", "EMG (mV)")

        if plot_legend:
            # Add legend of the latency windows to the first plot item
            self.add_latency_window_legend(plot_items[0])

        if plot_colormap:
            value_range = (
                min(self.emg_object.stimulus_voltages),
                max(self.emg_object.stimulus_voltages),
            )
            self.add_colormap_scalebar(layout, plot_items, value_range)

        # Set X-axis range explicitly to prevent flickering from auto-calculation
        # The time_axis is already computed and represents the actual data range
        if len(time_axis) > 0:
            x_min = time_axis[0]
            x_max = time_axis[-1]
            # Set X range on first plot (propagates to all X-linked plots)
            plot_items[0].setXRange(x_min, x_max, padding=0)

        # Set Y-axis range for all linked plots based on tracked bounds
        if overall_y_min != float("inf") and overall_y_max != float("-inf"):
            y_range = overall_y_max - overall_y_min
            if y_range == 0:
                y_range = max(abs(overall_y_max), 1.0) * 0.1
            padding = 0.05  # 5% padding
            padded_y_min = overall_y_min - (y_range * padding)
            padded_y_max = overall_y_max + (y_range * padding)
            # Set range on first plot (propagates to all Y-linked plots)
            plot_items[0].setYRange(padded_y_min, padded_y_max, padding=0)
        else:
            # Fallback to auto-range if no valid data found
            plot_items[0].enableAutoRange(axis="y", enable=True)

        if raw_data_dict is None:
            return None

        # Create DataFrame with multi-level index (matching matplotlib version exactly)
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(
            ["recording_index", "channel_index", "stimulus_V", "time_point"],
            inplace=True,
        )
        return raw_data_df

    def plot_singleEMG(
        self,
        channel_indices: list[int] | None = None,
        recording_index: int = 0,
        fixed_y_axis: bool = True,
        all_flags: bool = True,
        plot_legend: bool = True,
        plot_colormap: bool = False,
        data_type: str = "filtered",
        interactive_cursor: bool = True,
        show_extrema_labels: bool = False,
        extrema_label_method: str = "exclusive_extrema_ptt",
        collect_raw_data: bool = True,
        canvas: PlotPane = None,
    ):
        """
        Plot single EMG recording with interactive features.

        Parameters
        ----------
        channel_indices : list[int], optional
            list of channel indices to plot
        recording_index : int, optional
            Index of recording to plot (default: 0)
        fixed_y_axis : bool, optional
            Whether to fix y-axis across channels (default: True)
        all_flags : bool, optional
            Whether to show all latency windows (default: True)
        plot_legend : bool, optional
            Whether to show legend (default: True)
        plot_colormap : bool, optional
            Whether to show colormap (default: False)
        data_type : str, optional
            Type of data to plot (default: 'filtered')
        interactive_cursor : bool, optional
            Whether to enable interactive crosshair cursor (default: True)
        collect_raw_data : bool, optional
            Whether to build and return the tabular plot-data export.
        canvas : PlotPane, optional
            Canvas to plot on

        Returns
        -------
        pd.DataFrame
            Raw data used for plotting
        """
        if canvas is None:
            raise UnableToPlotError("Canvas must be provided for PyQtGraph plotting")

        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))
        self._clear_extrema_items()

        # num_channels = len(channel_indices)
        time_axis, window_start_sample, window_end_sample = self.get_time_axis()

        # Create plot layout
        plot_items, layout = self.create_plot_layout(canvas, channel_indices)

        # Add crosshair for precise measurements (if enabled)
        if interactive_cursor:
            self.add_synchronized_crosshairs(plot_items)

        # Plot latency windows once per channel
        self.latency_window_flag_items.clear()
        for plot_idx, channel_index in enumerate(channel_indices):
            current_plot = plot_items[plot_idx]
            self.plot_latency_windows(current_plot, all_flags, channel_index)

        # Get EMG recordings
        # We access ALL recordings here (including excluded) to ensure that the recording_index
        # from the GUI matches the list index. The GUI cycler iterates over all recordings.
        emg_recordings = self.get_emg_recordings(data_type, use_all=True)

        # Strict recording index validation: never auto-wrap or coerce silently.
        # Rationale: Silent wrapping can mask upstream logic errors (e.g., stale cached
        # counts after exclusions). By raising an UnableToPlotError we surface the
        # problematic index to the GUI layer, which can log and handle it explicitly.
        if not emg_recordings:
            raise UnableToPlotError("No recordings available to plot in this session")
        if recording_index < 0 or recording_index >= len(emg_recordings):
            raise UnableToPlotError(f"Recording index {recording_index} out of range (0-{len(emg_recordings) - 1})")

        # Calculate fixed y-axis limits if needed
        # Note: We likely want to scale the y-axis based on valid (non-excluded) recordings only,
        # so that artifacts in excluded recordings don't squash the view of good data.
        if fixed_y_axis:
            try:
                # Use active recordings for auto-scaling
                active_recordings = self.get_emg_recordings(data_type, use_all=False)
                max_y = []
                min_y = []
                for rec in active_recordings:
                    for channel_index, channel_data in enumerate(rec.T):
                        if channel_index in channel_indices:
                            segment = channel_data[window_start_sample:window_end_sample]
                            if segment.size > 0:
                                max_y.append(np.max(segment))
                                min_y.append(np.min(segment))
                if max_y and min_y:
                    y_max = max(max_y)
                    y_min = min(min_y)
                    y_range = y_max - y_min
                    y_max += 0.01 * y_range
                    y_min -= 0.01 * y_range
                else:
                    y_min, y_max = None, None  # No valid data found; will fallback to auto-scaling
            except Exception as e:
                logger.warning(f"Error calculating fixed y-axis limits: {e}")
                y_max, y_min = None, None  # Fallback to auto-scaling if error occurs

        # call this after calculating y_min and y_max
        recording = emg_recordings[recording_index]
        # Retrieve stimulus voltage from the correct recording object (using all_recordings)
        stimulus_v = 0.0
        # Use simple attribute access now that the domain object supports it
        if recording_index < len(self.emg_object.all_stimulus_voltages):
            stimulus_v = self.emg_object.all_stimulus_voltages[recording_index]
        else:
            logger.warning(f"Recording index {recording_index} out of range for stimulus voltages.")

        # Prepare colormap normalization if needed
        norm = None
        if plot_colormap:
            norm = self.get_brightness_adjusted_norm()

        # Raw data collection
        raw_data_dict = (
            {
                "channel_index": [],
                "stimulus_V": [],
                "time_point": [],
                "amplitude_mV": [],
            }
            if collect_raw_data
            else None
        )

        # Plot each channel (only the ones specified in channel_indices)
        for channel_index, channel_data in enumerate(recording.T):
            if channel_index not in channel_indices:
                continue
            # Get the correct plot index for this channel
            plot_idx = channel_indices.index(channel_index)
            current_plot = plot_items[plot_idx]

            # Show grid by default
            current_plot.showGrid(x=True, y=True)

            # Get channel data
            data_segment = channel_data[window_start_sample:window_end_sample]
            if data_segment.size == 0:
                continue

            # Plot the channel data with colormap if requested
            if norm is not None:
                # Use colormap for this stimulus voltage
                color_value = norm(stimulus_v)
                color = self.stim_colormap.map(color_value, mode="qcolor")
            else:
                color = self.default_colors[channel_index % len(self.default_colors)]
            self.plot_time_series(current_plot, time_axis, data_segment, color=color, line_width=1.5)
            if show_extrema_labels and data_type == "filtered":
                self._plot_extrema_annotations(
                    current_plot, channel_index, self.emg_object.all_recordings[recording_index].id, extrema_label_method, labels=True
                )

            # Set fixed y-axis if requested
            if fixed_y_axis and y_min is not None and y_max is not None:
                current_plot.setYRange(y_min, y_max)

            if raw_data_dict is not None:
                # Store the original (non-downsampled) data only for export.
                num_points = len(time_axis)
                raw_data_dict["channel_index"].extend([channel_index] * num_points)
                raw_data_dict["stimulus_V"].extend([stimulus_v] * num_points)
                raw_data_dict["time_point"].extend(time_axis)
                raw_data_dict["amplitude_mV"].extend(data_segment)

            # Set labels
            channel_name = self.emg_object.channel_names[channel_index]
            self.set_labels(
                current_plot,
                title=f"{channel_name}",
                x_label="Time (ms)",
                y_label="EMG (mV)",
            )

        # Add legend if requested
        if plot_legend:
            self.add_latency_window_legend(plot_item=plot_items[0])

        # Add colormap scalebar if requested
        if plot_colormap:
            value_range = (
                min(self.emg_object.stimulus_voltages),
                max(self.emg_object.stimulus_voltages),
            )
            self.add_colormap_scalebar(layout, plot_items, value_range)

        # Add an info bubble showing the primary stimulus amplitude (formatted like dataset/experiment inlays)
        # This inlay mirrors the pg.TextItem usage in dataset/experiment plotters
        try:
            info_text = f"Stimulus: {stimulus_v:.2f} V"
            # Use same styling as other plotters' inlay TextItem
            text_item = pg.TextItem(
                info_text,
                anchor=(1, 0),
                color="white",
                border=pg.mkPen("w"),
                fill=pg.mkBrush(255, 255, 255, 200),
            )
            # Position the text in the top-right of the first plot item
            first_plot = plot_items[0]
            vb = first_plot.vb

            def update_info_bubble(*_args):
                """Keep the bubble anchored to the visible top-right corner."""
                x_range, y_range = vb.viewRange()
                x_min, x_max = x_range
                y_min, y_max = y_range
                x_span = x_max - x_min
                y_span = y_max - y_min
                if x_span <= 0 or y_span <= 0:
                    return
                text_item.setPos(x_max - 0.02 * x_span, y_max - 0.02 * y_span)

            # This view-relative annotation must not contribute its moving anchor
            # to the plot's initial data bounds (which would cause auto-ranging to
            # expand continuously as the anchor follows the top-right corner).
            first_plot.addItem(text_item, ignoreBounds=True)
            # TextItem keeps its screen size as the ViewBox transforms; update its
            # data-coordinate anchor whenever the user zooms or pans.
            vb.sigRangeChanged.connect(update_info_bubble)
            update_info_bubble()
        except Exception:
            # Fail silently for non-critical UI addition
            logger.error("Failed to add stimulus info inlay to single EMG plot", exc_info=True)

        if not fixed_y_axis:
            # Auto-range Y-axis for all linked plots
            self.auto_range_y_axis_linked_plots(plot_items)

        if raw_data_dict is None:
            return None

        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(["channel_index", "stimulus_V", "time_point"], inplace=True)
        return raw_data_df

    def plot_reflexCurves(
        self,
        channel_indices: list[int] | None = None,
        method=None,
        plot_legend=True,
        relative_to_mmax=False,
        manual_mmax=None,
        interactive_cursor: bool = True,
        canvas: PlotPane = None,
    ):
        """
        Plot reflex curves (M-wave and H-reflex vs stimulus voltage).

        Parameters
        ----------
        channel_indices : list[int], optional
            list of channel indices to plot
        method : str, optional
            Method for amplitude calculation
        plot_legend : bool, optional
            Whether to show legend (default: True)
        relative_to_mmax : bool, optional
            Whether to normalize to M-max (default: False)
        manual_mmax : float, optional
            Manual M-max value
        interactive_cursor : bool, optional
            Whether to enable interactive crosshair cursor (default: True)
        canvas : PlotPane, optional
            Canvas to plot on

        Returns
        -------
        pd.DataFrame
            Raw data used for plotting
        """
        if canvas is None:
            raise UnableToPlotError("Canvas must be provided for PyQtGraph plotting")

        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))

        if len(self.emg_object.latency_windows) == 0:
            raise ValueError("No latency windows found. Add some to plot reflex curves.")

        # Check for empty recordings - reflex curves require active recordings
        if not self.emg_object.recordings_filtered:
            raise UnableToPlotError("No active recordings available to plot reflex curves. All recordings may have been excluded.")

        # Create plot layout
        plot_items, _ = self.create_plot_layout(canvas, channel_indices)

        # Add synchronized crosshairs to all plots (if enabled)
        if interactive_cursor:
            self.add_synchronized_crosshairs(plot_items)

        # Raw data collection
        raw_data_dict = {"channel_index": [], "stimulus_V": []}

        # Plot each channel
        for plot_idx, channel_index in enumerate(channel_indices):
            current_plot = plot_items[plot_idx]
            stimulus_voltages: np.ndarray = self.emg_object.stimulus_voltages

            # Plot all latency window reflex amplitudes
            window_amplitudes_dict = {}  # window label -> amplitude list
            window_colors = {}
            for window in self.emg_object.latency_windows:
                amps: np.ndarray = self.emg_object.get_lw_reflex_amplitudes(method=method, channel_index=channel_index, window=window)
                logger.info(f"Channel {channel_index}, Window {window.label}: {len(amps)} amplitudes")

                # Normalize if requested
                if relative_to_mmax:
                    m_max = self.emg_object.get_m_max(method=method, channel_index=channel_index) if manual_mmax is None else manual_mmax
                    if m_max != 0:
                        amps = [amp / m_max for amp in amps]
                window_amplitudes_dict[window.label] = amps
                window_colors[window.label] = self._convert_matplotlib_color(window.color)

            # Plot each window's amplitudes
            for label, amps in window_amplitudes_dict.items():
                color = window_colors[label]
                self.plot_scatter(
                    current_plot,
                    stimulus_voltages,
                    amps,
                    color=color,
                    size=8,
                    symbol="o",
                )
                raw_data_dict.setdefault(f"{label}_amplitudes", []).extend(amps)
            raw_data_dict["channel_index"].extend([channel_index] * len(stimulus_voltages))
            raw_data_dict["stimulus_V"].extend(stimulus_voltages)

            # Set labels
            channel_name = self.emg_object.channel_names[channel_index]
            y_label = f"Reflex Ampl. (mV, {method})" if method else "Reflex Ampl. (mV)"
            if relative_to_mmax:
                y_label = f"Reflex Ampl. (M-max, {method})" if method else "Reflex Ampl. (M-max)"

            self.set_labels(
                current_plot,
                title=f"{channel_name}",
                x_label="Stimulus Voltage (V)",
                y_label=y_label,
            )

            # Add grid
            current_plot.showGrid(True, True)

            # Add legend if requested
            if plot_legend:
                legend = self.add_legend(current_plot)
                for i, label in enumerate(window_amplitudes_dict.keys()):
                    legend.addItem(current_plot.listDataItems()[i], label)

        for plot_item in plot_items:
            # Remove link y-axes to the first plot item
            plot_item.setYLink(None)
            plot_item.enableAutoRange(axis="y", enable=True)

        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(["channel_index", "stimulus_V"], inplace=True)
        return raw_data_df

    def plot_mmax(
        self,
        channel_indices: list[int] | None = None,
        method: str | None = None,
        interactive_cursor: bool = True,
        canvas: PlotPane = None,
    ):
        """
        DEPRECATED: Session-level M-max plots are not supported.

        Since Session objects already represent the unit used to calculate one M-max per channel,
        plotting M-max at the session level is not meaningful. Use Dataset or Experiment plotters for M-max summaries.
        """
        raise UnableToPlotError(
            "Session-level M-max plots are not supported because a Session already represents the unit used to calculate one M-max per channel. "
            "Use Dataset or Experiment plotters for M-max summaries."
        )

    def connect_colorbar_signals(self):
        """Connect colorbar signals to update plotted curve colors dynamically."""
        if self.colorbar_item is not None:
            # Connect to colormap level changes
            self.colorbar_item.sigLevelsChangeFinished.connect(self.update_curve_colors)
            # Fallback: also connect to generic update signal if available (for older pyqtgraph)
            if hasattr(self.colorbar_item, "sigColorMapChanged"):
                self.colorbar_item.sigColorMapChanged.connect(self.update_curve_colors)
            elif hasattr(self.colorbar_item, "sigUpdated"):
                # sigUpdated is emitted on any change, including colormap changes
                self.colorbar_item.sigUpdated.connect(self.update_curve_colors)

    def update_curve_colors(self):
        """Update all plotted curve colors based on current colormap and levels."""
        if not self.plotted_curves or not self.curve_stimulus_voltages:
            return

        # Get current colormap from colorbar
        current_colormap = self.colorbar_item.colorMap() if self.colorbar_item else self.stim_colormap
        logger.debug(f"colormap type: {type(current_colormap)}, {current_colormap}")

        # Get current value range from colorbar
        if self.colorbar_item:
            min_val, max_val = self.colorbar_item.levels()
        else:
            min_val = min(self.emg_object.stimulus_voltages)
            max_val = max(self.emg_object.stimulus_voltages)

        # Get brightness-adjusted normalization function
        norm = self.get_brightness_adjusted_norm(min_val, max_val)

        # Update colors for all stored curves
        for curve, stimulus_v in zip(self.plotted_curves, self.curve_stimulus_voltages, strict=True):
            color_value = norm(stimulus_v)
            new_color = current_colormap.map(color_value, mode="qcolor")
            curve.setPen(pg.mkPen(color=new_color, width=1.0))

    def plot_reflexAverages(
        self,
        channel_indices: list[int] | None = None,
        method: str | None = None,
        plot_legend: bool = True,
        relative_to_mmax: bool = False,
        manual_mmax: float | None = None,
        interactive_cursor: bool = True,
        canvas: PlotPane = None,
    ):
        """
        Plot average reflex amplitudes with error bars for all latency windows.

        Shows scatter plots with error bars representing the mean ± standard deviation
        of amplitudes for each latency window across all recordings in the session.

        Parameters
        ----------
        channel_indices : list[int], optional
            list of channel indices to plot
        method : str, optional
            Method for amplitude calculation
        plot_legend : bool, optional
            Whether to show legend (default: True)
        relative_to_mmax : bool, optional
            Whether to normalize to M-max (default: False)
        manual_mmax : float, optional
            Manual M-max value for normalization
        interactive_cursor : bool, optional
            Whether to enable interactive crosshair cursor (default: True)
        canvas : PlotPane, optional
            Canvas to plot on

        Returns
        -------
        pd.DataFrame
            Raw data used for plotting with mean and std columns
        """
        if canvas is None:
            raise UnableToPlotError("Canvas must be provided for PyQtGraph plotting")

        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))

        if len(self.emg_object.latency_windows) == 0:
            raise ValueError("No latency windows found. Add some to plot reflex averages.")

        # Check for empty recordings - reflex averages require active recordings
        if not self.emg_object.recordings_filtered:
            raise UnableToPlotError("No active recordings available to plot reflex averages. All recordings may have been excluded.")

        # Create plot layout
        plot_items, _ = self.create_plot_layout(canvas, channel_indices)

        # Add synchronized crosshairs to all plots (if enabled)
        if interactive_cursor:
            self.add_synchronized_crosshairs(plot_items)

        # Raw data collection
        raw_data_dict = {
            "channel_index": [],
            "window_label": [],
            "mean_amplitude": [],
            "std_amplitude": [],
            "n_recordings": [],
        }

        # Plot each channel
        for plot_idx, channel_index in enumerate(channel_indices):
            current_plot = plot_items[plot_idx]

            # Collect data for all latency windows
            window_data = []
            x_positions = []
            window_labels = []
            window_colors = []

            for i, window in enumerate(self.emg_object.latency_windows):
                # Get amplitudes for this window
                amps = self.emg_object.get_lw_reflex_amplitudes(method=method, channel_index=channel_index, window=window)

                # Filter out NaN values
                valid_amps = [amp for amp in amps if not np.isnan(amp)]

                if len(valid_amps) == 0:
                    logger.warning(f"No valid amplitudes found for channel {channel_index}, window {window.label}")
                    continue

                # Normalize if requested
                if relative_to_mmax:
                    m_max = self.emg_object.get_m_max(method=method, channel_index=channel_index) if manual_mmax is None else manual_mmax
                    if m_max != 0:
                        valid_amps = [amp / m_max for amp in valid_amps]

                # Calculate statistics
                mean_amp = np.mean(valid_amps)
                std_amp = np.std(valid_amps)
                n_recordings = len(valid_amps)

                # Store data
                window_data.append((mean_amp, std_amp, n_recordings))
                x_positions.append(i)
                window_labels.append(window.label if hasattr(window, "label") else str(window))
                window_colors.append(self._convert_matplotlib_color(window.color))

                # Collect raw data
                raw_data_dict["channel_index"].append(channel_index)
                raw_data_dict["window_label"].append(window_labels[-1])
                raw_data_dict["mean_amplitude"].append(mean_amp)
                raw_data_dict["std_amplitude"].append(std_amp)
                raw_data_dict["n_recordings"].append(n_recordings)

                logger.info(f"Channel {channel_index}, Window {window_labels[-1]}: mean={mean_amp:.3f}, std={std_amp:.3f}, n={n_recordings}")

            if not window_data:
                logger.warning(f"No valid data found for channel {channel_index}")
                continue

            # Plot scatter points with error bars for each window
            for i, ((mean_amp, std_amp, n_rec), x_pos, _labels, color) in enumerate(
                zip(window_data, x_positions, window_labels, window_colors, strict=True)
            ):
                # Get the actual individual amplitudes for this window to plot as scatter
                amps = self.emg_object.get_lw_reflex_amplitudes(
                    method=method,
                    channel_index=channel_index,
                    window=self.emg_object.latency_windows[i],
                )
                valid_amps = [amp for amp in amps if not np.isnan(amp)]

                # Normalize individual amplitudes if requested
                if relative_to_mmax:
                    m_max = self.emg_object.get_m_max(method=method, channel_index=channel_index) if manual_mmax is None else manual_mmax
                    if m_max != 0:
                        valid_amps = [amp / m_max for amp in valid_amps]

                # Plot individual data points as scatter
                if len(valid_amps) > 0:
                    scatter = pg.ScatterPlotItem(
                        x=np.array([x_pos] * len(valid_amps)),
                        y=np.array(valid_amps),
                        pen=pg.mkPen("white", width=0.1),
                        brush=pg.mkBrush(color),
                        size=8,
                        symbol="o",
                    )
                    current_plot.addItem(scatter)

                # Add error bars (white)
                error_bars = pg.ErrorBarItem(
                    x=np.array([x_pos]),
                    y=np.array([mean_amp]),
                    top=np.array([std_amp]),
                    bottom=np.array([std_amp]),
                    pen=pg.mkPen("white", width=2),
                    beam=0.2,
                )
                current_plot.addItem(error_bars)

                # Add mean marker (white cross)
                mean_marker = pg.ScatterPlotItem(
                    x=np.array([x_pos]),
                    y=np.array([mean_amp]),
                    pen=pg.mkPen("white", width=2),
                    brush=pg.mkBrush("white"),
                    size=12,
                    symbol="+",
                )
                current_plot.addItem(mean_marker)

                # Add annotation text
                text_item = pg.TextItem(
                    f"n={n_rec}\nAvg. Ampl.: {mean_amp:.2f}mV\nStdev. Ampl.: {std_amp:.2f}mV",
                    anchor=(0, 0.5),
                    color="black",
                    border=pg.mkPen("w"),
                    fill=pg.mkBrush(255, 255, 255, 180),
                )
                text_item.setPos(x_pos + 0.2, mean_amp)
                current_plot.addItem(text_item)

            # Set labels and formatting
            channel_name = self.emg_object.channel_names[channel_index]
            y_label = f"Mean Reflex Ampl. (mV, {method})" if method else "Mean Reflex Ampl. (mV)"
            if relative_to_mmax:
                y_label = f"Mean Reflex Ampl. (M-max, {method})" if method else "Mean Reflex Ampl. (M-max)"

            self.set_labels(
                current_plot,
                title=f"{channel_name}",
                x_label="Latency Window",
                y_label=y_label,
            )

            # Set x-axis ticks to show window labels
            if window_labels:
                tick_pairs = [(i, label) for i, label in enumerate(window_labels)]
                current_plot.getAxis("bottom").setTicks([tick_pairs])
                # Set x-axis range to show all windows with some padding
                current_plot.setXRange(-0.5, len(window_labels) - 0.5)

            # Enable grid
            current_plot.showGrid(True, True)

            # Add legend if requested
            if plot_legend and window_labels:
                legend = current_plot.addLegend(offset=(10, 10))
                for _i, (label, color) in enumerate(zip(window_labels, window_colors, strict=True)):
                    # Create dummy scatter item for legend
                    dummy_scatter = pg.ScatterPlotItem(
                        x=np.array([0]),
                        y=np.array([0]),
                        pen=pg.mkPen("white", width=2),
                        brush=pg.mkBrush(color),
                        size=12,
                        symbol="o",
                    )
                    legend.addItem(dummy_scatter, label)

        for plot_item in plot_items:
            # Remove link y-axes to the first plot item
            plot_item.setYLink(None)
            plot_item.enableAutoRange(axis="y", enable=True)

        # Create DataFrame with appropriate index
        raw_data_df = pd.DataFrame(raw_data_dict)
        if not raw_data_df.empty:
            raw_data_df.set_index(["channel_index", "window_label"], inplace=True)
        return raw_data_df

    def plot_averageReflexCurves(
        self,
        channel_indices: list[int] | None = None,
        method: str | None = None,
        plot_legend: bool = True,
        relative_to_mmax: bool = False,
        manual_mmax: float | None = None,
        interactive_cursor: bool = True,
        canvas: PlotPane = None,
    ):
        """
        Plot latency window amplitudes binned by stimulus voltage showing trends over time.

        Shows average trend lines with shadowed standard deviations for each latency window,
        binning stimulus voltages using the config's bin_size like in plot_emg.

        Parameters
        ----------
        channel_indices : list[int], optional
            list of channel indices to plot
        method : str, optional
            Method for amplitude calculation
        plot_legend : bool, optional
            Whether to show legend (default: True)
        relative_to_mmax : bool, optional
            Whether to normalize to M-max (default: False)
        manual_mmax : float, optional
            Manual M-max value for normalization
        interactive_cursor : bool, optional
            Whether to enable interactive crosshair cursor (default: True)
        canvas : PlotPane, optional
            Canvas to plot on

        Returns
        -------
        pd.DataFrame
            Raw data used for plotting with binned voltages and window amplitudes
        """
        if canvas is None:
            raise UnableToPlotError("Canvas must be provided for PyQtGraph plotting")

        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))

        if len(self.emg_object.latency_windows) == 0:
            raise ValueError("No latency windows found. Add some to plot latency window trends.")

        # Check for empty recordings - average reflex curves require active recordings
        if not self.emg_object.recordings_filtered:
            raise UnableToPlotError("No active recordings available to plot average reflex curves. All recordings may have been excluded.")

        # Create plot layout
        plot_items, _ = self.create_plot_layout(canvas, channel_indices)

        # Add synchronized crosshairs to all plots (if enabled)
        if interactive_cursor:
            self.add_synchronized_crosshairs(plot_items)

        # Raw data collection
        raw_data_dict = {
            "channel_index": [],
            "window_label": [],
            "stimulus_voltage": [],
            "mean_amplitude": [],
            "std_amplitude": [],
            "n_recordings": [],
        }

        # Plot each channel
        for plot_idx, channel_index in enumerate(channel_indices):
            current_plot = plot_items[plot_idx]

            # Add legend for this plot if requested (before plotting named curves)
            if plot_legend:
                current_plot.addLegend()

            # Get stimulus voltages and apply binning
            stimulus_voltages = self.emg_object.stimulus_voltages
            binned_voltages = np.round(stimulus_voltages / self.emg_object.bin_size) * self.emg_object.bin_size
            unique_voltages = np.array(sorted(set(binned_voltages)))

            # Plot each latency window
            for window in self.emg_object.latency_windows:
                window_label = window.label if hasattr(window, "label") else str(window)
                # Get all amplitudes for this window
                all_amps = self.emg_object.get_lw_reflex_amplitudes(method=method, channel_index=channel_index, window=window)

                # Create bins for amplitudes based on stimulus voltages
                voltage_bins = {v: [] for v in unique_voltages}
                for volt, amp in zip(binned_voltages, all_amps, strict=True):
                    if not np.isnan(amp):
                        voltage_bins[volt].append(amp)

                # Calculate means and stds for each voltage bin
                plot_voltages = []
                plot_means = []
                plot_stds = []

                for voltage in unique_voltages:
                    amps_at_voltage = voltage_bins[voltage]
                    if len(amps_at_voltage) > 0:
                        mean_amp = np.mean(amps_at_voltage)
                        std_amp = np.std(amps_at_voltage) if len(amps_at_voltage) > 1 else 0.0

                        # Normalize if requested
                        if relative_to_mmax:
                            m_max = self.emg_object.get_m_max(method=method, channel_index=channel_index) if manual_mmax is None else manual_mmax
                            if m_max != 0:
                                mean_amp /= m_max
                                std_amp /= m_max

                        plot_voltages.append(voltage)
                        plot_means.append(mean_amp)
                        plot_stds.append(std_amp)

                        # Collect raw data
                        raw_data_dict["channel_index"].append(channel_index)
                        raw_data_dict["window_label"].append(window_label)
                        raw_data_dict["stimulus_voltage"].append(voltage)
                        raw_data_dict["mean_amplitude"].append(mean_amp)
                        raw_data_dict["std_amplitude"].append(std_amp)
                        raw_data_dict["n_recordings"].append(len(amps_at_voltage))

                if len(plot_voltages) == 0:
                    logger.warning(f"No valid data found for channel {channel_index}, window {window.label}")
                    continue

                plot_voltages = np.array(plot_voltages)
                plot_means = np.array(plot_means)
                plot_stds = np.array(plot_stds)

                # Convert matplotlib color to PyQtGraph compatible color
                window_color = self._convert_matplotlib_color(window.color)
                pale_color = self._pale_color(window_color, blend=0.25)

                # Plot error bands with dotted boundary lines
                upper_bound = plot_means + plot_stds
                lower_bound = plot_means - plot_stds

                # Create dotted boundary lines
                transparent_pen = pg.mkPen(color=pale_color, width=1, style=pg.QtCore.Qt.PenStyle.DotLine)
                upper_curve = current_plot.plot(plot_voltages, upper_bound, pen=transparent_pen)
                lower_curve = current_plot.plot(plot_voltages, lower_bound, pen=transparent_pen)

                # Store curve references if needed for cleanup
                if not hasattr(current_plot, "_fill_curves_refs"):
                    current_plot._fill_curves_refs = []
                current_plot._fill_curves_refs.extend([upper_curve, lower_curve])

                # Create fill between curves
                fill_item = pg.FillBetweenItem(
                    curve1=upper_curve,
                    curve2=lower_curve,
                    brush=pg.mkBrush(color=pale_color, alpha=50),
                )
                current_plot.addItem(fill_item)

                # Plot mean line with symbols on top
                current_plot.plot(
                    plot_voltages,
                    plot_means,
                    pen=pg.mkPen(color=window_color, width=2),
                    symbol="o",
                    symbolSize=6,
                    symbolBrush=window_color,
                    name=window_label,
                )

                logger.info(f"Channel {channel_index}, Window {window.label}: plotted {len(plot_voltages)} voltage bins")

            # Set labels and formatting
            channel_name = self.emg_object.channel_names[channel_index]
            y_label = f"Average Reflex Ampl. (mV{', ' + method if method else ''})"
            if relative_to_mmax:
                y_label = f"Average Reflex Ampl. (mV{', rel. to M-max' if relative_to_mmax else ''})"

            self.set_labels(
                current_plot,
                title=f"{channel_name}",
                x_label="Stimulus Intensity (V)",
                y_label=y_label,
            )

            # Enable grid
            current_plot.showGrid(True, True)

        # Auto-range both axes
        self.auto_range_y_axis_linked_plots(plot_items)

        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        if not raw_data_df.empty:
            raw_data_df.set_index(["channel_index", "window_label", "stimulus_voltage"], inplace=True)
        return raw_data_df

    def clear_curve_references(self):
        """Clear stored curve references (call this before plotting new data)."""
        self.plotted_curves.clear()
        self.curve_stimulus_voltages.clear()
        self.colorbar_item = None

    def plot_latency_window_distribution(
        self,
        channel_indices: list[int] | None = None,
        method: str | None = None,
        bins: int | np.ndarray = 30,
        density: bool = False,
        plot_legend: bool = True,
        canvas: PlotPane = None,
    ):
        """Plot distribution (histogram) of latency-window reflex amplitudes for a SESSION.

        Similar semantics to the dataset-level plot: x = binned EMG amplitudes, y = counts or
        probability density. Each latency window is shown as a dot plot with connecting lines.

        Returns a pandas.DataFrame indexed by (channel_index, window_label, bin_center) with
        column 'value'.
        """
        if canvas is None:
            raise UnableToPlotError("Canvas must be provided for PyQtGraph plotting")

        if method is None:
            method = self.emg_object.default_method

        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))

        if len(self.emg_object.latency_windows) == 0:
            raise ValueError("No latency windows found. Add some to plot distributions.")

        plot_items, _ = self.create_plot_layout(canvas, channel_indices)

        # Unlink X-axes so each channel can auto-range independently
        for _pi in plot_items:
            try:
                _pi.setXLink(None)
            except Exception:
                logger.exception("Failed to unlink X-axis for plot item")

        raw_data_dict = {
            "channel_index": [],
            "window_label": [],
            "bin_left": [],
            "bin_right": [],
            "bin_center": [],
            "value": [],
        }

        # Per-channel plotting
        for plot_item, channel_index in zip(plot_items, channel_indices, strict=True):
            if channel_index >= self.emg_object.num_channels:
                continue

            windows = self.emg_object.latency_windows
            distribution = self.emg_object.get_lw_distribution(method, channel_index, bins, density)
            bin_edges = distribution["bin_edges"]
            bin_centers = distribution["bin_centers"]
            window_values = distribution["values"]
            if bin_edges.size == 0:
                # Nothing to plot for this channel
                self.set_labels(plot_item, title=f"{self.emg_object.channel_names[channel_index]}", x_label="EMG Amp.", y_label="Frequency")
                plot_item.showGrid(True, True)
                continue

            # Add legend container if requested
            if plot_legend:
                try:
                    plot_item.addLegend()
                except Exception:
                    logger.exception("Failed to add legend to plot item")

            # Plot each window
            for window_label, counts in window_values.items():
                if counts.size == 0:
                    continue
                try:
                    for left, right, center, freq in zip(bin_edges[:-1], bin_edges[1:], bin_centers, counts, strict=True):
                        raw_data_dict["channel_index"].append(channel_index)
                        raw_data_dict["window_label"].append(window_label)
                        raw_data_dict["bin_left"].append(float(left))
                        raw_data_dict["bin_right"].append(float(right))
                        raw_data_dict["bin_center"].append(float(center))
                        raw_data_dict["value"].append(float(freq))

                    # Choose color from window object if possible
                    color_hex = None
                    try:
                        # Find window object by label
                        for w in windows:
                            if getattr(w, "label", str(w)) == window_label:
                                color_hex = w.color
                                break
                    except Exception:
                        pass
                    if color_hex is None:
                        color_hex = "white"

                    color = self._convert_matplotlib_color(color_hex)

                    plot_item.plot(
                        bin_centers,
                        counts,
                        pen=pg.mkPen(color=color, width=2),
                        symbol="o",
                        symbolSize=6,
                        symbolBrush=color,
                        name=window_label if plot_legend else None,
                    )
                except Exception:
                    # Skip window on error
                    continue

            # Ensure each plot auto-ranges its X axis independently
            try:
                plot_item.enableAutoRange(axis="x", enable=True)
            except Exception:
                logger.exception("Failed to enable auto-range for X-axis on plot item")

            self.set_labels(
                plot_item,
                title=f"{self.emg_object.channel_names[channel_index]}",
                x_label="EMG Amp. (binned)",
                y_label=("Probability Density" if density else "Frequency"),
            )
            plot_item.showGrid(True, True)

        # Auto-range Y-axis across linked plots
        self.auto_range_y_axis_linked_plots(plot_items)

        raw_data_df = pd.DataFrame(raw_data_dict)
        if not raw_data_df.empty:
            raw_data_df.set_index(["channel_index", "window_label", "bin_center"], inplace=True)
        return raw_data_df
