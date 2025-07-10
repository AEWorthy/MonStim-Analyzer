import numpy as np
import pandas as pd
import pyqtgraph as pg
import logging
from typing import TYPE_CHECKING, List, Tuple
from .base_plotter_pyqtgraph import BasePlotterPyQtGraph, UnableToPlotError

if TYPE_CHECKING:
    from monstim_signals.domain.session import Session
    from monstim_gui.widgets.plotting.plotting_widget import PlotPane

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
    
    def __init__(self, emg_object: 'Session'):
        super().__init__(emg_object)
        self.emg_object: 'Session' = emg_object
        # Shared colormap for both data and colorbar
        self.stim_colormap = pg.colormap.get('viridis')
        self.stim_colormap.reverse()  # Reverse the colormap for better visibility
        # Store references to plotted curves for dynamic updates
        self.plotted_curves : List[pg.PlotDataItem] = []
        self.curve_stimulus_voltages : List[float] = []
        self.colorbar_item : pg.ColorBarItem = None
        
    def get_time_axis(self, offset=2):
        """Get time axis for plotting."""
        # Calculate time values based on the scan rate
        time_values_ms = np.arange(self.emg_object.num_samples) * 1000 / self.emg_object.scan_rate  # Time values in milliseconds

        # Define the start and end times for the window
        window_start_time = self.emg_object.stim_start - offset  # Start [offset]ms before stimulus onset
        window_end_time = window_start_time + self.emg_object.time_window_ms

        # Convert time window to sample indices
        window_start_sample = int(window_start_time * self.emg_object.scan_rate / 1000)
        window_end_sample = int(window_end_time * self.emg_object.scan_rate / 1000)

        # Ensure indices are within bounds
        window_start_sample = max(0, window_start_sample)
        window_end_sample = min(self.emg_object.num_samples, window_end_sample)

        # Slice the time array for the time window
        time_axis = time_values_ms[window_start_sample:window_end_sample] - self.emg_object.stim_start

        return time_axis, window_start_sample, window_end_sample
    
    def get_emg_recordings(self, data_type, original=False) -> List[np.ndarray]:
        """
        Get the EMG recordings based on the specified data type.
        
        This method matches the original matplotlib plotter interface exactly.
        """
        if original:
            raise NotImplementedError("Original recordings are not supported in EMGSessionPlotter. Please use the 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered' data types.")
        else:
            if data_type in ['raw', 'filtered', 'rectified_raw', 'rectified_filtered']:
                attribute_name = f'recordings_{data_type}'
                data = getattr(self.emg_object, attribute_name)
                if data is None:
                    raise AttributeError(f"Data type '{attribute_name}' is not available in the Session object. Please ensure that the data has been processed and stored correctly.")
                return data
            else:
                raise ValueError(f"Data type '{data_type}' is not supported. Please use 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered'.")
    
    def plot_channel_data(self, plot_item: pg.PlotItem, time_axis: np.ndarray, 
                         channel_data: np.ndarray, start: int, end: int, 
                         stimulus_v: float, channel_index: int, norm=None):
        """
        Plot channel data with stimulus voltage-based coloring.
        
        Parameters match the original matplotlib plotter interface.
        """
        # Get data segment
        data_segment = channel_data[start:end]
        
        # Create color based on stimulus voltage using shared colormap
        if norm is not None:
            color_value = norm(stimulus_v)
            color = self.stim_colormap.map(color_value, mode='qcolor')
        else:
            # Use default color cycling
            color = self.default_colors[channel_index % len(self.default_colors)]
        
        # Plot the data
        curve = plot_item.plot(time_axis, data_segment, 
                              pen=pg.mkPen(color=color, width=1.0))
        
        # Store curve reference for dynamic colormap updates
        if norm is not None:
            self.plotted_curves.append(curve)
            self.curve_stimulus_voltages.append(stimulus_v)
        
        # Set title and grid
        plot_item.setTitle(f'{self.emg_object.channel_names[channel_index]}')
        plot_item.showGrid(True, True)
        
        return curve
           
    def plot_latency_windows(self, plot_item: pg.PlotItem, all_flags: bool, channel_index: int):
        """
        Plot latency windows as vertical lines.
        
        Parameters match the original matplotlib plotter interface.
        """
        if all_flags:
            for window in self.emg_object.latency_windows:
                # Convert matplotlib color to PyQtGraph-compatible color
                color = self._convert_matplotlib_color(window.color)
                
                # Create vertical lines for window start and end
                start_line = pg.InfiniteLine(
                    pos=window.start_times[channel_index],
                    angle=90,
                    pen=pg.mkPen(color=color, style=self._get_line_style(window.linestyle), width=2)
                )
                end_line = pg.InfiniteLine(
                    pos=window.end_times[channel_index],
                    angle=90,
                    pen=pg.mkPen(color=color, style=self._get_line_style(window.linestyle), width=2)
                )
                
                plot_item.addItem(start_line)
                plot_item.addItem(end_line)
    
    def add_latency_window_legend(self, plot_item: pg.PlotItem):
        """
        Add a native PyQtGraph legend to the plot item for latency windows.
        """
        legend = plot_item.addLegend(offset=(10, 10))
        # Use a short horizontal line as a dummy item for each latency window
        for window in self.emg_object.latency_windows:
            color = self._convert_matplotlib_color(window.color)
            pen = pg.mkPen(color=color, style=self._get_line_style(window.linestyle), width=2)
            # Create a dummy PlotDataItem (short horizontal line)
            x = np.array([0, 1])
            y = np.array([0, 0])
            dummy_item = pg.PlotDataItem(x, y, pen=pen)
            legend.addItem(dummy_item, window.label if hasattr(window, 'label') else str(window))
        return legend

    def _get_line_style(self, matplotlib_style):
        """Convert matplotlib line style to PyQtGraph line style."""
        style_map = {
            '-': pg.QtCore.Qt.PenStyle.SolidLine,
            '--': pg.QtCore.Qt.PenStyle.DashLine,
            ':': pg.QtCore.Qt.PenStyle.DotLine,
            '-.': pg.QtCore.Qt.PenStyle.DashDotLine
        }
        return style_map.get(matplotlib_style, pg.QtCore.Qt.PenStyle.SolidLine)

    def add_colormap_scalebar(self, layout : pg.GraphicsLayout, plot_items: List[pg.PlotItem], value_range : Tuple[float, float]):
        colorbar = pg.ColorBarItem(
            colorMap=self.stim_colormap,
            values=value_range,
            label='Stimulus Voltage (V)',
            orientation='vertical',
            interactive=False,
            colorMapMenu=False,
        )
        # Store reference to colorbar for dynamic updates
        self.colorbar_item = colorbar
        
        # Connect colorbar signals to update curve colors
        self.connect_colorbar_signals()
        
        # Add to the right of the last plot (assuming one row)
        layout.addItem(colorbar, row=0, col=len(plot_items))

    def plot_emg(self, channel_indices: List[int] = None, all_flags: bool = True, 
                plot_legend: bool = True, plot_colormap: bool = False, 
                data_type: str = 'filtered', stimuli_to_plot: List[str] = None, 
                interactive_cursor: bool = True, canvas: 'PlotPane' = None):
        """
        Plot EMG data with interactive features.
        
        Parameters
        ----------
        channel_indices : List[int], optional
            List of channel indices to plot
        all_flags : bool, optional
            Whether to show all latency windows (default: True)
        plot_legend : bool, optional
            Whether to show legend (default: True)
        plot_colormap : bool, optional
            Whether to show colormap (default: False)
        data_type : str, optional
            Type of data to plot (default: 'filtered')
        stimuli_to_plot : List[str], optional
            List of stimuli to plot
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
        
        # Clear previous curve references for new plot
        self.clear_curve_references()
        
        time_axis, window_start_sample, window_end_sample = self.get_time_axis()
        
        # Create plot layout
        plot_items : List[pg.PlotItem]
        layout : pg.GraphicsLayout
        plot_items, layout = self.create_plot_layout(canvas, channel_indices)

        # Add synchronized crosshairs to all plots (if enabled)
        if interactive_cursor:
            self.add_synchronized_crosshairs(plot_items)
        
        # Get EMG recordings
        emg_recordings = self.get_emg_recordings(data_type)
        
        # Create normalization for stimulus voltages (matching matplotlib version)
        def norm(v):
            min_v = min(self.emg_object.stimulus_voltages)
            max_v = max(self.emg_object.stimulus_voltages)
            return (v - min_v) / (max_v - min_v) if max_v != min_v else 0.5
        
        # Initialize raw data collection (matching matplotlib version exactly)
        raw_data_dict = {
            'recording_index': [],
            'channel_index': [],
            'stimulus_V': [],
            'time_point': [],
            'amplitude_mV': []
        }
        
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
                
                # Plot EMG data
                self.plot_channel_data(current_plot, time_axis, channel_data, 
                                     window_start_sample, window_end_sample, 
                                     stimulus_v, channel_index, norm=norm)
                
                # Plot latency windows
                self.plot_latency_windows(current_plot, all_flags, channel_index)
                
                # Collect raw data with hierarchical index structure (matching matplotlib)
                num_points = len(time_axis)
                raw_data_dict['recording_index'].extend([recording_idx] * num_points)
                raw_data_dict['channel_index'].extend([channel_index] * num_points)
                raw_data_dict['stimulus_V'].extend([stimulus_v] * num_points)
                raw_data_dict['time_point'].extend(time_axis)
                raw_data_dict['amplitude_mV'].extend(channel_data[window_start_sample:window_end_sample])
        
        # Set labels for each plot
        for plot_idx, channel_index in enumerate(channel_indices):
            current_plot : pg.PlotItem = plot_items[plot_idx]
            current_plot.setLabel('bottom', 'Time (ms)')
            current_plot.setLabel('left', 'EMG (mV)')
        
        if plot_legend:
            # Add legend of the latency windows to the first plot item
            self.add_latency_window_legend(plot_items[0])

        if plot_colormap:
            value_range = (min(self.emg_object.stimulus_voltages), max(self.emg_object.stimulus_voltages))
            self.add_colormap_scalebar(layout, plot_items, value_range)

        # Display the plot
        self.display_plot(canvas)
        
        # Create DataFrame with multi-level index (matching matplotlib version exactly)
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(['recording_index', 'channel_index', 'stimulus_V', 'time_point'], inplace=True)
        return raw_data_df
    
    def plot_singleEMG(self, channel_indices: List[int] = None, recording_index: int = 0, 
                      fixed_y_axis: bool = True, all_flags: bool = True, 
                      plot_legend: bool = True, plot_colormap: bool = False, 
                      data_type: str = 'filtered', interactive_cursor: bool = True, 
                      canvas: 'PlotPane' = None):
        """
        Plot single EMG recording with interactive features.
        
        Parameters
        ----------
        channel_indices : List[int], optional
            List of channel indices to plot
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
        
        # num_channels = len(channel_indices)
        time_axis, window_start_sample, window_end_sample = self.get_time_axis()
        
        # Create plot layout
        plot_items, layout = self.create_plot_layout(canvas, channel_indices)
        
        # Add crosshair for precise measurements (if enabled)
        if interactive_cursor:
            self.add_synchronized_crosshairs(plot_items)
        
        # Get EMG recordings
        emg_recordings = self.get_emg_recordings(data_type)
        
        if recording_index >= len(emg_recordings):
            raise ValueError(f"Recording index {recording_index} out of range")
        
        # Calculate fixed y-axis limits if needed
        y_min, y_max = None, None
        if fixed_y_axis:
            max_y = []
            min_y = []
            for rec in emg_recordings:
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
                y_max, y_min = 1, -1

        # call this after calculating y_min and y_max
        recording = emg_recordings[recording_index]
        stimulus_v = self.emg_object.stimulus_voltages[recording_index]

        # Prepare colormap normalization if needed
        norm = None
        if plot_colormap:
            min_v = min(self.emg_object.stimulus_voltages)
            max_v = max(self.emg_object.stimulus_voltages)
            def norm_func(v):
                return (v - min_v) / (max_v - min_v) if max_v != min_v else 0.5
            norm = norm_func

        # Raw data collection
        raw_data_dict = {
            'channel_index': [],
            'stimulus_V': [],
            'time_point': [],
            'amplitude_mV': []
        }
        
        # Plot each channel (only the ones specified in channel_indices)
        for channel_index, channel_data in enumerate(recording.T):
            if channel_index not in channel_indices:
                continue
            # Get the correct plot index for this channel
            plot_idx = channel_indices.index(channel_index)
            current_plot = plot_items[plot_idx]

            # Show grid by default
            current_plot.showGrid(x=True, y=True)

            # Plot latency windows
            self.plot_latency_windows(current_plot, all_flags, channel_index)

            # Get channel data
            data_segment = channel_data[window_start_sample:window_end_sample]
            if data_segment.size == 0:
                continue

            # Plot the channel data with colormap if requested
            if norm is not None:
                # Use colormap for this stimulus voltage
                color_value = norm(stimulus_v)
                color = self.stim_colormap.map(color_value, mode='qcolor')
            else:
                color = self.default_colors[channel_index % len(self.default_colors)]
            self.plot_time_series(
                current_plot, time_axis, data_segment,
                color=color,
                line_width=1.5
            )

            # Set fixed y-axis if requested
            if fixed_y_axis and y_min is not None and y_max is not None:
                current_plot.setYRange(y_min, y_max)

            # Collect raw data
            num_points = len(time_axis)
            raw_data_dict['channel_index'].extend([channel_index] * num_points)
            raw_data_dict['stimulus_V'].extend([stimulus_v] * num_points)
            raw_data_dict['time_point'].extend(time_axis)
            raw_data_dict['amplitude_mV'].extend(data_segment)

            # Set labels
            channel_name = self.emg_object.channel_names[channel_index]
            self.set_labels(
                current_plot,
                title=f'{channel_name}',
                x_label='Time (ms)',
                y_label='EMG (mV)'
            )
            
        # Add legend if requested
        if plot_legend:
            self.add_latency_window_legend(plot_item=plot_items[0])

        # Add colormap scalebar if requested
        if plot_colormap:
            value_range = (min(self.emg_object.stimulus_voltages), max(self.emg_object.stimulus_voltages))
            self.add_colormap_scalebar(layout, plot_items, value_range)

        # Display the plot
        self.display_plot(canvas)

        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(['channel_index', 'stimulus_V', 'time_point'], inplace=True)
        return raw_data_df
    
    def plot_reflexCurves(self, channel_indices: List[int] = None, method=None, 
                         plot_legend=True, relative_to_mmax=False, manual_mmax=None, 
                         interactive_cursor: bool = True, canvas: 'PlotPane' = None):
        """
        Plot reflex curves (M-wave and H-reflex vs stimulus voltage).
        
        Parameters
        ----------
        channel_indices : List[int], optional
            List of channel indices to plot
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
                
        # Create plot layout
        plot_items, layout = self.create_plot_layout(canvas, channel_indices)
        
        # Add synchronized crosshairs to all plots (if enabled)
        if interactive_cursor:
            self.add_synchronized_crosshairs(plot_items)
        
        # Raw data collection
        raw_data_dict = {
            'channel_index': [],
            'stimulus_V': []
        }
        
        # Plot each channel
        for plot_idx, channel_index in enumerate(channel_indices):
            current_plot = plot_items[plot_idx]
            stimulus_voltages : np.ndarray = self.emg_object.stimulus_voltages

            # Plot all latency window reflex amplitudes
            window_amplitudes_dict = {}  # window label -> amplitude list
            window_colors = {}
            for window in self.emg_object.latency_windows:
                amps : np.ndarray = self.emg_object.get_lw_reflex_amplitudes(
                    method=method, channel_index=channel_index, window=window
                )
                logging.info(f"Channel {channel_index}, Window {window.label}: {len(amps)} amplitudes")
                
                # Normalize if requested
                if relative_to_mmax:
                    if manual_mmax is not None:
                        m_max = manual_mmax
                    else:
                        m_max = self.emg_object.get_m_max(method=method, channel_index=channel_index)
                    if m_max != 0:
                        amps = [amp / m_max for amp in amps]
                window_amplitudes_dict[window.label] = amps
                window_colors[window.label] = self._convert_matplotlib_color(window.color)

            # Plot each window's amplitudes
            for label, amps in window_amplitudes_dict.items():
                color = window_colors[label]
                self.plot_scatter(
                    current_plot, stimulus_voltages, amps,
                    color=color,
                    size=8, symbol='o'
                )
                raw_data_dict.setdefault(f'{label}_amplitudes', []).extend(amps)
            raw_data_dict['channel_index'].extend([channel_index] * len(stimulus_voltages))
            raw_data_dict['stimulus_V'].extend(stimulus_voltages)

            # Set labels
            channel_name = self.emg_object.channel_names[channel_index]
            y_label = f'Reflex Ampl. (mV, {method})' if method else 'Reflex Ampl. (mV)'
            if relative_to_mmax:
                y_label = f'Reflex Ampl. (M-max, {method})' if method else 'Reflex Ampl. (M-max)'

            self.set_labels(
                current_plot,
                title=f'{channel_name}',
                x_label='Stimulus Voltage (V)',
                y_label=y_label
            )

            # Add grid
            current_plot.showGrid(True, True)

            # Add legend if requested
            if plot_legend:
                legend = self.add_legend(current_plot)
                for i, label in enumerate(window_amplitudes_dict.keys()):
                    legend.addItem(current_plot.listDataItems()[i], label)
        
        # Display the plot
        self.display_plot(canvas)
        
        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(['channel_index', 'stimulus_V'], inplace=True)
        return raw_data_df
    
    def plot_mmax(self, channel_indices: List[int] = None, method: str = None, 
                 interactive_cursor: bool = True, canvas: 'PlotPane' = None):
        """
        Plot M-max values for each channel.
        
        Parameters
        ----------
        channel_indices : List[int], optional
            List of channel indices to plot
        method : str, optional
            Method for amplitude calculation
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
                
        # Create plot layout
        plot_items, layout = self.create_plot_layout(canvas, channel_indices)
        
        # Add synchronized crosshairs to all plots (if enabled)
        if interactive_cursor:
            self.add_synchronized_crosshairs(plot_items)
        
        # Raw data collection
        raw_data_dict = {
            'channel_index': [],
            'm_max_threshold': [],
            'm_max_amplitudes': []
        }
        
        # Set a light background for the canvas if possible
        if hasattr(canvas, 'setBackground'):  # GraphicsLayoutWidget
            canvas.setBackground('#f5f5f5')

        # Plot each channel
        for plot_idx, channel_index in enumerate(channel_indices):
            current_plot = plot_items[plot_idx]

            # Get M-max data
            m_max_amplitudes = self.emg_object.get_m_wave_amplitudes(method=method, channel_index=channel_index)
            m_max, mmax_low_stim, _ = self.emg_object.get_m_max(method=method, channel_index=channel_index, return_mmax_stim_range=True)

            # Plot M-max values as scatter with black edge
            x_pos = 0  # Single position on x-axis
            m_color = self._convert_matplotlib_color(self.emg_object.m_color) if hasattr(self.emg_object, 'm_color') else '#ff3333'
            scatter = pg.ScatterPlotItem(
                x=np.array([x_pos] * len(m_max_amplitudes)),
                y=np.array(m_max_amplitudes),
                pen=pg.mkPen('black', width=1.5),
                brush=pg.mkBrush(m_color),
                size=14,
                symbol='o',
                pxMode=True
            )
            current_plot.addItem(scatter)

            # Add mean line (dark green, thick, dashed)
            if len(m_max_amplitudes) > 0:
                mean_amp = np.mean(m_max_amplitudes)
                mean_line = pg.InfiniteLine(pos=mean_amp, angle=0, pen=pg.mkPen('#228B22', width=3, style=pg.QtCore.Qt.PenStyle.DashLine))
                current_plot.addItem(mean_line)

            # Add error bars (standard deviation, semi-transparent blue)
            if len(m_max_amplitudes) > 1:
                std_amp = np.std(m_max_amplitudes)
                error_bar = pg.ErrorBarItem(
                    x=np.array([x_pos]),
                    y=np.array([mean_amp]),
                    top=np.array([std_amp]),
                    bottom=np.array([std_amp]),
                    beam=0.22,
                    pen=pg.mkPen((30, 144, 255, 180), width=4)
                )
                current_plot.addItem(error_bar)

            # Collect raw data
            raw_data_dict['channel_index'].extend([channel_index] * len(m_max_amplitudes))
            raw_data_dict['m_max_threshold'].extend([mmax_low_stim] * len(m_max_amplitudes))
            raw_data_dict['m_max_amplitudes'].extend(m_max_amplitudes)

            # Set labels
            channel_name = self.emg_object.channel_names[channel_index]
            self.set_labels(
                current_plot,
                title=f'{channel_name}',
                x_label='Response Type',
                y_label=f'M-max (mV, {method})' if method else 'M-max (mV)'
            )

            # Set x-axis ticks and center the plot visually
            current_plot.getAxis('bottom').setTicks([[(x_pos, 'M-response')]])
            # Center the data by setting x-axis range
            current_plot.setXRange(-0.5, 0.5)

            # Add annotation (pin box to the top right corner of the plot window)
            if len(m_max_amplitudes) > 0:
                # Get the viewbox to determine the visible range
                vb = current_plot.getViewBox()
                # Use the current y-range, or fallback to data range
                view_range = vb.viewRange()
                y_min, y_max = view_range[1]
                x_min, x_max = view_range[0]
                y_range = y_max - y_min if y_max != y_min else 1
                x_range = x_max - x_min if x_max != x_min else 1
                # Pin to top right, with a small margin
                margin_x = 0.04 * x_range
                margin_y = 0.04 * y_range
                x_annot = x_max - margin_x
                y_annot = y_max - margin_y
                text_item = pg.TextItem(
                    f'n={len(m_max_amplitudes)}\nM-max: {m_max:.2f}mV\nStim.: >{mmax_low_stim:.2f}V',
                    anchor=(1, 1), color='black', border=pg.mkPen('w'), fill=(255,255,255,220)
                )
                text_item.setFont(pg.QtGui.QFont('Arial', 13, pg.QtGui.QFont.Weight.Bold))
                text_item.setPos(x_annot, y_annot)
                current_plot.addItem(text_item)

            # Prettify: add grid (lighter), adjust axis font, tighten y-axis
            current_plot.showGrid(x=True, y=True, alpha=0.18)
            ax = current_plot.getAxis('left')
            ax.setStyle(tickFont=pg.QtGui.QFont('Arial', 13, pg.QtGui.QFont.Weight.Bold))
            ax = current_plot.getAxis('bottom')
            ax.setStyle(tickFont=pg.QtGui.QFont('Arial', 13, pg.QtGui.QFont.Weight.Bold))

            # Tighten y-axis
            if len(m_max_amplitudes) > 0:
                y_min = min(m_max_amplitudes)
                y_max = max(m_max_amplitudes)
                y_range = y_max - y_min if y_max != y_min else 1
                current_plot.setYRange(y_min - 0.08*y_range, y_max + 0.15*y_range)

        # Display the plot
        self.display_plot(canvas)

        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(['channel_index'], inplace=True)
        return raw_data_df
    
    def connect_colorbar_signals(self):
        """Connect colorbar signals to update plotted curve colors dynamically."""
        if self.colorbar_item is not None:
            # Connect to colormap level changes
            self.colorbar_item.sigLevelsChangeFinished.connect(self.update_curve_colors)
            # Fallback: also connect to generic update signal if available (for older pyqtgraph)
            if hasattr(self.colorbar_item, 'sigColorMapChanged'):
                self.colorbar_item.sigColorMapChanged.connect(self.update_curve_colors)
            elif hasattr(self.colorbar_item, 'sigUpdated'):
                # sigUpdated is emitted on any change, including colormap changes
                self.colorbar_item.sigUpdated.connect(self.update_curve_colors)
    
    def update_curve_colors(self):
        """Update all plotted curve colors based on current colormap and levels."""
        if not self.plotted_curves or not self.curve_stimulus_voltages:
            return
        
        # Get current colormap from colorbar
        current_colormap = self.colorbar_item.colorMap() if self.colorbar_item else self.stim_colormap
        print(f"colormap type: {type(current_colormap)}, {current_colormap}")

        # Get current value range from colorbar
        if self.colorbar_item:
            min_val, max_val = self.colorbar_item.levels()
        else:
            min_val = min(self.emg_object.stimulus_voltages)
            max_val = max(self.emg_object.stimulus_voltages)
        
        # Update normalization function
        def norm(v):
            return (v - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        
        # Update colors for all stored curves
        for curve, stimulus_v in zip(self.plotted_curves, self.curve_stimulus_voltages):
            color_value = norm(stimulus_v)
            new_color = current_colormap.map(color_value, mode='qcolor')
            curve.setPen(pg.mkPen(color=new_color, width=1.0))
    
    def clear_curve_references(self):
        """Clear stored curve references (call this before plotting new data)."""
        self.plotted_curves.clear()
        self.curve_stimulus_voltages.clear()
        self.colorbar_item = None
