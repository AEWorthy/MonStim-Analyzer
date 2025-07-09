import pyqtgraph as pg
import numpy as np
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from monstim_signals.domain.session import Session
    from monstim_signals.domain.dataset import Dataset
    from monstim_signals.domain.experiment import Experiment
    from monstim_gui.widgets.plotting.plotting_widget import PlotPane


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
        self.emg_object: 'Session'|'Dataset'|'Experiment' = emg_object
        self.current_plot_items: List[pg.PlotItem] = []
        self.current_regions: List[pg.LinearRegionItem] = []
        
        # Set up default colors
        self.default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
    def create_plot_layout(self, canvas: 'PlotPane', channel_indices: List[int] = None) -> tuple:
        """
        Create plot layout with subplots for multiple channels.
        
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
        canvas.clear_plots()
        
        plot_items = []
        
        if num_channels == 1:
            # Single plot
            plot_item = canvas.graphics_layout.addPlot(row=0, col=0)
            plot_items.append(plot_item)
        else:
            # Multiple plots in a row
            for i, channel_index in enumerate(channel_indices):
                plot_item = canvas.graphics_layout.addPlot(row=0, col=i)
                plot_items.append(plot_item)
                
                # Share axes for all plots
                if i > 0:
                    plot_item.setYLink(plot_items[0])
                    plot_item.setXLink(plot_items[0])
        
        # Store reference to current plots
        canvas.current_plots = plot_items
        canvas.current_plot_items = plot_items
        
        return plot_items, canvas.graphics_layout
    
    def add_crosshair(self, plot_item: pg.PlotItem) -> tuple:
        """
        Add crosshair cursor to a plot.
        
        Parameters
        ----------
        plot_item : pg.PlotItem
            The plot item to add crosshair to
            
        Returns
        -------
        tuple
            (v_line, h_line) - vertical and horizontal line objects
        """
        # Create crosshair lines
        v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=1))
        h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('w', width=1))
        
        plot_item.addItem(v_line, ignoreBounds=True)
        plot_item.addItem(h_line, ignoreBounds=True)
        
        # Connect mouse move events
        def mouse_moved(evt):
            # Handle different event formats
            if isinstance(evt, (list, tuple)):
                pos = evt[0]
            else:
                pos = evt
            
            if plot_item.sceneBoundingRect().contains(pos):
                mouse_point = plot_item.vb.mapSceneToView(pos)
                v_line.setPos(mouse_point.x())
                h_line.setPos(mouse_point.y())
        
        plot_item.scene().sigMouseMoved.connect(mouse_moved)
        
        return v_line, h_line
    
    def add_synchronized_crosshairs(self, plot_items):
        """
        Add synchronized crosshairs and a cursor indicator to all plot_items. Only the active plot shows a horizontal crosshair and indicator.
        """
        v_lines = []
        h_lines = []
        cursor_texts = []
        for plot_item in plot_items:
            v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=1))
            h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('w', width=1))
            plot_item.addItem(v_line, ignoreBounds=True)
            plot_item.addItem(h_line, ignoreBounds=True)
            v_lines.append(v_line)
            h_lines.append(h_line)
            # Add a cursor indicator (TextItem) to each plot
            cursor_text = pg.TextItem("", anchor=(0, 1), color='k', fill=pg.mkBrush(255, 255, 255, 180))
            plot_item.addItem(cursor_text)
            cursor_text.hide()
            cursor_texts.append(cursor_text)

        # Shared mouse move event
        def mouse_moved(evt):
            if isinstance(evt, (list, tuple)):
                pos = evt[0]
            else:
                pos = evt
            active_plot_idx = None
            for idx, plot_item in enumerate(plot_items):
                if plot_item.sceneBoundingRect().contains(pos):
                    active_plot_idx = idx
                    break
            if active_plot_idx is not None:
                active_plot = plot_items[active_plot_idx]
                mouse_point = active_plot.vb.mapSceneToView(pos)
                x = mouse_point.x()
                y = mouse_point.y()
                # Update all vertical crosshairs to this x
                for v in v_lines:
                    v.setPos(x)
                # Only show horizontal crosshair and indicator on the active plot
                for idx in range(len(plot_items)):
                    if idx == active_plot_idx:
                        h_lines[idx].setPos(y)
                        h_lines[idx].show()
                        v_lines[idx].show()
                        # Show and update the cursor indicator - position in view coordinates
                        view_range = plot_items[idx].vb.viewRange()
                        x_min, x_max = view_range[0][0], view_range[0][1] * 0.55
                        y_min, y_max = view_range[1]
                        x_clamped = min(max(x, x_min), x_max)
                        y_top = y_max - 0.05 * (y_max - y_min)  # 5% below the top
                        cursor_texts[idx].setText(f"x={x:.2f}, y={y:.2f}")
                        cursor_texts[idx].setPos(x_clamped, y_top)
                        cursor_texts[idx].setZValue(1000)
                        cursor_texts[idx].show()

                    else:
                        h_lines[idx].hide()
                        cursor_texts[idx].hide()
            else:
                # Hide all horizontal crosshairs and indicators if not over any plot
                for h, v, t in zip(h_lines, v_lines, cursor_texts):
                    h.hide()
                    v.hide()
                    t.hide()

        # Connect to the scene of the first plot (all plots share the same scene)
        if plot_items:
            plot_items[0].scene().sigMouseMoved.connect(mouse_moved)

        return v_lines, h_lines, cursor_texts

    # REMOVED: add_cursor_indicator. Now handled by add_synchronized_crosshairs.

    def add_latency_region(self, plot_item: pg.PlotItem, start_time: float, end_time: float, 
                          color: str = '#ff000030', label: str = '') -> pg.LinearRegionItem:
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
            bounds=[start_time - 50, end_time + 50]  # Allow some movement
        )
        
        plot_item.addItem(region)
        self.current_regions.append(region)
        
        if label:
            # Add text label for the region
            text_item = pg.TextItem(label, anchor=(0.5, 0), color='white')
            text_item.setPos((start_time + end_time) / 2, plot_item.viewRange()[1][1] * 0.9)
            plot_item.addItem(text_item)
        
        return region
    
    def plot_time_series(self, plot_item: pg.PlotItem, time_axis: np.ndarray, 
                        data: np.ndarray, color: str = None, label: str = None,
                        line_width: float = 1.0) -> pg.PlotDataItem:
        """
        Plot time series data on a plot item.
        
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
        self.current_plot_items.append(curve)
        
        return curve
    
    def plot_scatter(self, plot_item: pg.PlotItem, x_data: np.ndarray, 
                    y_data: np.ndarray, color: str = None, size: float = 5.0,
                    symbol: str = 'o', label: str = None) -> pg.ScatterPlotItem:
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
            x=x_data, y=y_data,
            pen=pg.mkPen(color),
            brush=pg.mkBrush(color),
            size=size,
            symbol=symbol
        )
        
        plot_item.addItem(scatter)
        self.current_plot_items.append(scatter)
        
        return scatter
    
    def add_error_bars(self, plot_item: pg.PlotItem, x_data: np.ndarray, 
                      y_data: np.ndarray, y_error: np.ndarray, 
                      color: str = None) -> pg.ErrorBarItem:
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
        
        error_bars = pg.ErrorBarItem(
            x=x_data, y=y_data, 
            top=y_error, bottom=y_error,
            pen=pg.mkPen(color)
        )
        
        plot_item.addItem(error_bars)
        self.current_plot_items.append(error_bars)
        
        return error_bars
    
    def set_labels(self, plot_item: pg.PlotItem, title: str = None, 
                  x_label: str = None, y_label: str = None):
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
            plot_item.setLabel('bottom', x_label)
        if y_label:
            plot_item.setLabel('left', y_label)
    
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
    
    def display_plot(self, canvas: 'PlotPane'):
        """
        Display the plot (equivalent to matplotlib's show()).
        
        Parameters
        ----------
        canvas : PlotPane
            The plot pane containing the plots
        """
        # PyQtGraph updates automatically, so we just need to process events
        canvas.graphics_layout.update()
    
    def clear_current_plots(self):
        """Clear all current plot items and regions."""
        self.current_plot_items = []
        self.current_regions = []


class UnableToPlotError(Exception):
    """Exception raised when plotting is not possible."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
