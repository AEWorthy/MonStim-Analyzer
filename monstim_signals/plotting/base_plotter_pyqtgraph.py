from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import pyqtgraph as pg

if TYPE_CHECKING:
    from monstim_gui.plotting import PlotPane
    from monstim_signals.domain.dataset import Dataset
    from monstim_signals.domain.experiment import Experiment
    from monstim_signals.domain.session import Session


class BasePlotterPyQtGraph:
    """
    A base class for plotting EMG data using PyQtGraph.

    This class provides interactive plotting capabilities with features like:
    - Real-time zooming and panning
    - Interactive region selection for latency windows
    - Crosshair cursor for precise measurements
    - Multi-channel plotting support
    """

    def __init__(self, emg_object):
        self.emg_object: "Session" | "Dataset" | "Experiment" = emg_object
        self.current_plot_items: List[pg.PlotItem] = []
        self.current_regions: List[pg.LinearRegionItem] = []

        # Set up default colors
        self.default_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

    def create_plot_layout(
        self, canvas: "PlotPane", channel_indices: List[int] = None
    ) -> Tuple[List[pg.PlotItem], pg.GraphicsLayout]:
        """
        Create plot layout with subplots for multiple channels with performance optimizations.

        Parameters
        ----------
        canvas : PlotPane
            The plot pane to draw on
        channel_indices : List[int], optional
            List of channel indices to plot

        Returns
        -------
        tuple
            (plot_items, layout) where plot_items is a list of PlotItem objects
        """
        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))

        num_channels = len(channel_indices)

        # Clear existing plots
        self.clear_current_plots(canvas)

        plot_items = []

        if num_channels == 1:
            # Single plot
            plot_item: pg.PlotItem = canvas.graphics_layout.addPlot(row=0, col=0)
            # Enable performance optimizations
            plot_item.setClipToView(True)
            plot_item.setDownsampling(auto=True)
            plot_items.append(plot_item)
        elif num_channels == 0:
            raise UnableToPlotError("No channels to plot. Select at least one channel.")
        else:
            # Multiple plots in a row
            for i, channel_index in enumerate(channel_indices):
                plot_item: pg.PlotItem = canvas.graphics_layout.addPlot(row=0, col=i)
                # Enable performance optimizations for each plot
                plot_item.setClipToView(True)
                plot_item.setDownsampling(auto=True)
                plot_items.append(plot_item)

                # Share axes for all plots
                if i > 0:
                    plot_item.setYLink(plot_items[0])
                    plot_item.setXLink(plot_items[0])

        # Store reference to current plots
        canvas.current_plots = plot_items
        canvas.current_plot_items = plot_items

        return plot_items, canvas.graphics_layout

    def add_synchronized_crosshairs(self, plot_items):
        """
        Add synchronized crosshairs and a cursor indicator to all plot_items. Only the active plot shows a horizontal crosshair and indicator.

        This is an optimized version that:
        1. Uses throttling to limit cursor updates (16ms = ~60fps)
        2. Only updates the text when the position changes significantly
        3. Caches calculations where possible
        4. Only shows text on demand (when stationary for a short time)
        5. Positions tooltip at top-right corner of active plot (avoids legend overlap)
        """
        import time

        from pyqtgraph.Qt import QtCore

        v_lines = []
        h_lines = []
        cursor_texts = []

        # Create crosshair lines and text items once
        for plot_item in plot_items:
            v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=1))
            h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("w", width=1))
            plot_item.addItem(v_line, ignoreBounds=True)
            plot_item.addItem(h_line, ignoreBounds=True)
            v_lines.append(v_line)
            h_lines.append(h_line)
            # Add a cursor indicator (TextItem) to each plot
            cursor_text = pg.TextItem(
                "", anchor=(1, 0), color="black", fill=pg.mkBrush(255, 255, 255, 200), border=pg.mkPen(color="black", width=1)
            )
            plot_item.addItem(cursor_text)
            cursor_text.hide()
            cursor_texts.append(cursor_text)

        # State variables for optimizations
        last_update_time = time.time()
        update_interval = 0.016  # 16ms (~60fps)
        last_active_plot_idx = None
        last_x = None
        last_y = None
        mouse_idle_timer = QtCore.QTimer()
        mouse_idle_timer.setSingleShot(True)
        mouse_idle_timer.setInterval(300)  # 300ms of idle time before showing text
        show_text = False

        # Function to show text indicators (called when mouse is idle)
        def enable_text_display():
            nonlocal show_text
            show_text = True
            # Only update if we have a valid position
            if last_active_plot_idx is not None and last_x is not None and last_y is not None:
                update_cursor_display(last_active_plot_idx, last_x, last_y)

        mouse_idle_timer.timeout.connect(enable_text_display)

        # Function to update the cursor display with minimal recomputation
        def update_cursor_display(active_plot_idx, x, y):
            # Update all vertical crosshairs to this x
            for v in v_lines:
                v.setPos(x)

            # Only show horizontal crosshair and indicator on the active plot
            for idx in range(len(plot_items)):
                if idx == active_plot_idx:
                    h_lines[idx].setPos(y)
                    h_lines[idx].show()
                    v_lines[idx].show()

                    # Only update text if it's visible
                    if show_text:
                        # Position tooltip at top-left corner of the plot view
                        try:
                            view_range = plot_items[idx].vb.viewRange()
                            x_min, x_max = view_range[0]
                            y_min, y_max = view_range[1]

                            # Fixed position: top-left corner with small offset
                            x_range = x_max - x_min
                            y_range = y_max - y_min

                            # Ensure we have valid ranges
                            if x_range > 0 and y_range > 0:
                                tooltip_x = x_max - 0.08 * x_range  # 8% from right edge (more padding)
                                tooltip_y = y_max - 0.05 * y_range  # 5% from top edge

                                cursor_texts[idx].setText(f" x: {x:.3f}, y: {y:.3f} ")
                                cursor_texts[idx].setPos(tooltip_x, tooltip_y)
                                cursor_texts[idx].setZValue(1000)
                                cursor_texts[idx].show()
                            else:
                                cursor_texts[idx].hide()
                        except (IndexError, TypeError, AttributeError):
                            # Fallback: hide text if positioning fails
                            cursor_texts[idx].hide()
                    else:
                        cursor_texts[idx].hide()
                else:
                    h_lines[idx].hide()
                    cursor_texts[idx].hide()

        # Hide all lines and text
        def hide_all_indicators():
            for h, v, t in zip(h_lines, v_lines, cursor_texts):
                h.hide()
                v.hide()
                t.hide()

        # Throttled mouse move event handler
        def mouse_moved(evt):
            nonlocal last_update_time, last_active_plot_idx, last_x, last_y, show_text

            # Throttle updates for performance
            current_time = time.time()
            if current_time - last_update_time < update_interval:
                return

            last_update_time = current_time

            # Reset idle timer and hide text temporarily during movement
            mouse_idle_timer.start()
            show_text = False

            # Extract position from event
            if isinstance(evt, (list, tuple)):
                pos = evt[0]
            else:
                pos = evt

            # Find which plot contains the cursor
            active_plot_idx = None
            for idx, plot_item in enumerate(plot_items):
                if plot_item.sceneBoundingRect().contains(pos):
                    active_plot_idx = idx
                    break

            # Update cursor position if over a plot
            if active_plot_idx is not None:
                active_plot = plot_items[active_plot_idx]
                mouse_point = active_plot.vb.mapSceneToView(pos)
                x = mouse_point.x()
                y = mouse_point.y()

                # Only update if position changed significantly
                position_changed = (
                    last_active_plot_idx != active_plot_idx
                    or last_x is None
                    or last_y is None
                    or abs(x - last_x) > 0.01
                    or abs(y - last_y) > 0.01
                )

                if position_changed:
                    last_active_plot_idx = active_plot_idx
                    last_x = x
                    last_y = y
                    update_cursor_display(active_plot_idx, x, y)
            else:
                # Hide all indicators if not over any plot
                hide_all_indicators()
                last_active_plot_idx = None
                last_x = None
                last_y = None

        # Connect to the scene of the first plot (all plots share the same scene)
        if plot_items:
            plot_items[0].scene().sigMouseMoved.connect(mouse_moved)

        return v_lines, h_lines, cursor_texts

    def add_latency_region(
        self,
        plot_item: pg.PlotItem,
        start_time: float,
        end_time: float,
        color: str = "#ff000030",
        label: str = "",
    ) -> pg.LinearRegionItem:
        """
        Add a latency window region to a plot.

        Parameters
        ----------
        plot_item : pg.PlotItem
            The plot item to add region to
        start_time : float
            Start time of the region in milliseconds
        end_time : float
            End time of the region in milliseconds
        color : str, optional
            Color of the region (default: semi-transparent red)
        label : str, optional
            Label for the region

        Returns
        -------
        pg.LinearRegionItem
            The created region item
        """
        region = pg.LinearRegionItem(
            [start_time, end_time],
            brush=pg.mkBrush(color),
            movable=True,
            bounds=[start_time - 50, end_time + 50],  # Allow some movement
        )

        plot_item.addItem(region)
        self.current_regions.append(region)

        if label:
            # Add text label for the region
            text_item = pg.TextItem(label, anchor=(0.5, 0), color="white")
            text_item.setPos((start_time + end_time) / 2, plot_item.viewRange()[1][1] * 0.9)
            plot_item.addItem(text_item)

        return region

    def plot_time_series(
        self,
        plot_item: pg.PlotItem,
        time_axis: np.ndarray,
        data: np.ndarray,
        color: str = None,
        label: str = None,
        line_width: float = 1.0,
    ) -> pg.PlotDataItem:
        """
        Plot time series data on a plot item with performance optimizations.

        Parameters
        ----------
        plot_item : pg.PlotItem
            The plot item to draw on
        time_axis : np.ndarray
            Time axis data
        data : np.ndarray
            Y-axis data
        color : str, optional
            Color of the line
        label : str, optional
            Label for the line
        line_width : float, optional
            Width of the line (default: 1.0)

        Returns
        -------
        pg.PlotDataItem
            The created plot data item
        """
        if color is None:
            color = self.default_colors[len(self.current_plot_items) % len(self.default_colors)]

        pen = pg.mkPen(color, width=line_width)

        curve = plot_item.plot(time_axis, data, pen=pen, name=label)

        # Enable performance optimizations
        curve.setClipToView(True)  # Only render visible portions
        curve.setDownsampling(auto=True)  # Auto-downsample when zoomed out

        self.current_plot_items.append(curve)

        return curve

    def plot_scatter(
        self,
        plot_item: pg.PlotItem,
        x_data: np.ndarray,
        y_data: np.ndarray,
        color: str = None,
        size: float = 5.0,
        symbol: str = "o",
        label: str = None,
    ) -> pg.ScatterPlotItem:
        """
        Plot scatter data on a plot item.

        Parameters
        ----------
        plot_item : pg.PlotItem
            The plot item to draw on
        x_data : np.ndarray
            X-axis data
        y_data : np.ndarray
            Y-axis data
        color : str, optional
            Color of the points
        size : float, optional
            Size of the points (default: 5.0)
        symbol : str, optional
            Symbol type (default: 'o')
        label : str, optional
            Label for the scatter plot

        Returns
        -------
        pg.ScatterPlotItem
            The created scatter plot item
        """
        if color is None:
            color = self.default_colors[len(self.current_plot_items) % len(self.default_colors)]

        scatter = pg.ScatterPlotItem(
            x=x_data,
            y=y_data,
            pen=pg.mkPen(color),
            brush=pg.mkBrush(color),
            size=size,
            symbol=symbol,
        )

        plot_item.addItem(scatter)
        self.current_plot_items.append(scatter)

        return scatter

    def add_error_bars(
        self,
        plot_item: pg.PlotItem,
        x_data: np.ndarray,
        y_data: np.ndarray,
        y_error: np.ndarray,
        color: str = None,
    ) -> pg.ErrorBarItem:
        """
        Add error bars to a plot.

        Parameters
        ----------
        plot_item : pg.PlotItem
            The plot item to draw on
        x_data : np.ndarray
            X-axis data
        y_data : np.ndarray
            Y-axis data (center values)
        y_error : np.ndarray
            Y-axis error values
        color : str, optional
            Color of the error bars

        Returns
        -------
        pg.ErrorBarItem
            The created error bar item
        """
        if color is None:
            color = self.default_colors[len(self.current_plot_items) % len(self.default_colors)]

        error_bars = pg.ErrorBarItem(x=x_data, y=y_data, top=y_error, bottom=y_error, pen=pg.mkPen(color))

        plot_item.addItem(error_bars)
        self.current_plot_items.append(error_bars)

        return error_bars

    def set_labels(
        self,
        plot_item: pg.PlotItem,
        title: str = None,
        x_label: str = None,
        y_label: str = None,
    ):
        """
        Set labels for a plot.

        Parameters
        ----------
        plot_item : pg.PlotItem
            The plot item to set labels for
        title : str, optional
            Plot title
        x_label : str, optional
            X-axis label
        y_label : str, optional
            Y-axis label
        """
        if title:
            plot_item.setTitle(title)
        if x_label:
            plot_item.setLabel("bottom", x_label)
        if y_label:
            plot_item.setLabel("left", y_label)

    def add_legend(self, plot_item: pg.PlotItem):
        """
        Add a legend to a plot.

        Parameters
        ----------
        plot_item : pg.PlotItem
            The plot item to add legend to
        """
        legend = plot_item.addLegend()
        return legend

    def clear_current_plots(self, canvas: "PlotPane"):
        """Clear all current plot items and regions."""
        self.current_plot_items = []
        self.current_regions = []
        canvas.clear_plots()

    def _convert_matplotlib_color(self, mpl_color):
        """Convert matplotlib color names to PyQtGraph-compatible colors."""
        color_map = {
            "tab:red": "#d62728",
            "tab:blue": "#1f77b4",
            "tab:orange": "#ff7f0e",
            "tab:green": "#2ca02c",
            "tab:purple": "#9467bd",
            "tab:brown": "#8c564b",
            "tab:pink": "#e377c2",
            "tab:gray": "#7f7f7f",
            "tab:olive": "#bcbd22",
            "tab:cyan": "#17becf",
            "red": "#ff0000",
            "blue": "#0000ff",
            "green": "#00ff00",
            "black": "#000000",
            "white": "#ffffff",
            "yellow": "#ffff00",
            "cyan": "#00ffff",
            "magenta": "#ff00ff",
        }
        return color_map.get(mpl_color, mpl_color)

    def _pale_color(self, color, blend=0.8):
        """Return a pale version of the given RGB color by blending with white."""

        def _hex_to_rgb(hex_color):
            """Convert hex color string to RGB tuple."""
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        if isinstance(color, str) and color.startswith("#"):
            r, g, b = _hex_to_rgb(color)
        else:
            r, g, b = color[:3]
        pale = (
            int(r * (1 - blend) + 255 * blend),
            int(g * (1 - blend) + 255 * blend),
            int(b * (1 - blend) + 255 * blend),
        )
        return pale

    def auto_range_y_axis_linked_plots(self, plot_items: List[pg.PlotItem], padding: float = 0.05):
        """
        Auto-range Y-axis for linked plots by calculating the optimal range across all plots.

        This solves the issue where individual auto-ranging on linked plots causes the last
        plot's range to override all others, potentially cutting off data from earlier plots.

        Parameters
        ----------
        plot_items : List[pg.PlotItem]
            List of plot items that are Y-linked
        padding : float, optional
            Fraction of padding to add above and below the data range (default: 0.05 = 5%)
        """
        if not plot_items:
            return

        # Calculate the overall data range across all plots
        overall_y_min = float("inf")
        overall_y_max = float("-inf")

        for plot_item in plot_items:
            # Get the data range for this plot
            data_bounds = plot_item.vb.childrenBounds()
            if data_bounds is not None and len(data_bounds) == 2:
                y_bounds = data_bounds[1]  # y-bounds are the second element
                if y_bounds is not None and len(y_bounds) == 2:
                    y_min, y_max = y_bounds
                    if not (np.isnan(y_min) or np.isnan(y_max) or np.isinf(y_min) or np.isinf(y_max)):
                        overall_y_min = min(overall_y_min, y_min)
                        overall_y_max = max(overall_y_max, y_max)

        # Check if we found valid bounds
        if overall_y_min == float("inf") or overall_y_max == float("-inf"):
            # Fallback: use individual auto-range on the first plot only
            if plot_items:
                plot_items[0].enableAutoRange(axis="y", enable=True)
            return

        # Add padding to the range
        y_range = overall_y_max - overall_y_min
        if y_range == 0:
            y_range = max(abs(overall_y_max), 1.0) * 0.1  # Fallback for zero range

        padded_y_min = overall_y_min - (y_range * padding)
        padded_y_max = overall_y_max + (y_range * padding)

        # Apply the calculated range to all plots
        # We only need to set it on the first plot since they're Y-linked
        if plot_items:
            plot_items[0].setYRange(padded_y_min, padded_y_max, padding=0)

    def _resolve_to_scalar(self, value):
        """
        Resolve a possibly-array-like or numpy value to a Python scalar (float) or None.

        Many domain methods may return numpy arrays, numpy scalars, lists, or None.
        Using plain truthiness checks on numpy arrays raises "The truth value of an
        array with more than one element is ambiguous.". This helper centralizes
        the conversion to a safe scalar for plotting logic.

        Rules:
        - None -> None
        - numpy scalar -> float
        - numpy array or list: if it has size==0 -> None, if size==1 -> scalar value
          else -> raise ValueError so callers decide how to handle multi-value cases
        - other scalars -> float
        """
        import numpy as _np

        if value is None:
            return None

        # numpy scalar
        if isinstance(value, (_np.floating, _np.integer)):
            return float(value)

        # numpy array
        if isinstance(value, _np.ndarray):
            if value.size == 0:
                return None
            if value.size == 1:
                return float(value.reshape(-1)[0])
            # ambiguous multi-element array
            raise ValueError("Ambiguous array with multiple elements cannot be resolved to scalar")

        # list or tuple
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            if len(value) == 1:
                return float(value[0])
            raise ValueError("Ambiguous sequence with multiple elements cannot be resolved to scalar")

        # fallback for other numeric-like types
        try:
            return float(value)
        except Exception:
            return None


class UnableToPlotError(Exception):
    """Exception raised when plotting is not possible."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
