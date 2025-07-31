import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from typing import TYPE_CHECKING, List
from .base_plotter_pyqtgraph import BasePlotterPyQtGraph, UnableToPlotError

if TYPE_CHECKING:
    from monstim_signals.domain.dataset import Dataset
    from monstim_gui.plotting import PlotPane


class DatasetPlotterPyQtGraph(BasePlotterPyQtGraph):
    """
    PyQtGraph-based plotter for Dataset data with interactive features.
    
    This class provides interactive plotting capabilities for EMG dataset data:
    - Real-time zooming and panning
    - Interactive latency window selection
    - Crosshair cursor for measurements
    - Multi-channel plotting support
    """
    
    def __init__(self, dataset: 'Dataset'):
        super().__init__(dataset)
        self.emg_object: 'Dataset' = dataset
        
    def plot_averageReflexCurves(self, channel_indices: List[int] = None, method : str = None, plot_legend : bool = True, 
                         relative_to_mmax : bool = False, manual_mmax : float | int | None =None,
                         interactive_cursor : bool = False, canvas : 'PlotPane'=None):
        """Plot average reflex curves for the dataset with interactive features, using domain Dataset object API."""
        try:
            if canvas is None:
                raise UnableToPlotError("Canvas is required for plotting.")

            # Get channel information from domain object
            if channel_indices is None:
                channel_indices = list(range(self.emg_object.num_channels))

            plot_items, layout = self.create_plot_layout(canvas, channel_indices)

            # Raw data collection
            raw_data_dict = {
                'channel_index': [],
                'window_name': [],
                'voltage': [],
                'mean_amplitude': [],
                'stdev_amplitude': [],
            }

            for plot_item, channel_idx in zip(plot_items, channel_indices):
                plot_item : 'pg.PlotItem'

                # Create plot item
                self.set_labels(plot_item=plot_item, title=f'{self.emg_object.channel_names[channel_idx]}',
                                y_label=f'Average Reflex Ampl. (mV{", rel. to M-max" if relative_to_mmax else ""})',
                                x_label='Stimulus Intensity (V)')
                plot_item.showGrid(True, True)
                
                # Plot reflex curves data and collect raw data
                self._plot_reflex_curves_data(plot_item, channel_idx, method, relative_to_mmax, manual_mmax, plot_legend, raw_data_dict)

            if interactive_cursor:
                self.add_synchronized_crosshairs(plot_items)

            # Add legend if requested, after all plotting is done
            if plot_legend and plot_items:
                self.add_legend(plot_items[0])

            for plot_item in plot_items:
                plot_item.enableAutoRange(axis='y', enable=True)

            # Display the plot
            self.display_plot(canvas)

            # Create DataFrame with multi-level index
            raw_data_df = pd.DataFrame(raw_data_dict)
            raw_data_df.set_index(['channel_index', 'window_name', 'voltage'], inplace=True)
            return raw_data_df

        except Exception as e:
            raise UnableToPlotError(f"Error plotting reflex curves: {str(e)}")

    def _plot_reflex_curves_data(self, plot_item: pg.PlotItem, channel_idx, method, relative_to_mmax, manual_mmax, plot_legend, raw_data_dict):
        """Plot reflex curves data for a specific channel, robust to domain object return types."""
        try:
            # Get the data from the domain Dataset object
            for window in self.emg_object.latency_windows:
                window_reflex_data = self.emg_object.get_average_lw_reflex_curve(
                    method=method,
                    channel_index=channel_idx,
                    window=window
                )

                # Make a copy of the data for potential normalization
                means = window_reflex_data['means']
                stdevs = window_reflex_data['stdevs']
                voltages = window_reflex_data['voltages']

                # Normalize to M-max if needed
                if relative_to_mmax and means is not None:
                    if manual_mmax is None:
                        mmax = self.emg_object.get_avg_m_max(
                            channel_index=channel_idx,
                            method=method
                        )
                    else:
                        mmax = manual_mmax
                    if mmax == 0:
                        raise ValueError("M-max cannot be zero when normalizing reflex curves.")
                    # Normalize means and stdevs by M-max
                    means = means / mmax
                    stdevs = stdevs / mmax

                # Collect raw data
                for v, m, s in zip(voltages, means, stdevs):
                    raw_data_dict['channel_index'].append(channel_idx)
                    raw_data_dict['window_name'].append(window.name)
                    raw_data_dict['voltage'].append(v)
                    raw_data_dict['mean_amplitude'].append(m)
                    raw_data_dict['stdev_amplitude'].append(s)

                color = self._convert_matplotlib_color(window.color)
                pale_color = self._pale_color(color, blend=0.25)
                
                # Plot the error bands
                upper = means + stdevs
                lower = means - stdevs
                transparent_pen = pg.mkPen(color=pale_color, width=1, style=QtCore.Qt.PenStyle.DotLine)
                upper_curve = plot_item.plot(voltages, upper, pen=transparent_pen)
                lower_curve = plot_item.plot(voltages, lower, pen=transparent_pen)
                if not hasattr(plot_item, '_fill_curves_refs'):
                    plot_item._fill_curves_refs = []
                plot_item._fill_curves_refs.extend([upper_curve, lower_curve])
                fill = pg.FillBetweenItem(curve1=upper_curve, curve2=lower_curve, brush=pg.mkBrush(color=pale_color, alpha=50))
                plot_item.addItem(fill)

                # Plot mean line last so it appears on top and is visible
                if plot_legend:
                    plot_item.addLegend()
                plot_item.plot(
                    voltages,
                    means,
                    pen=pg.mkPen(color=color, width=2),
                    symbol='o', symbolSize=6, symbolBrush=color,
                    name=window.name
                )

        except Exception as e:
            print(f"Warning: Could not plot reflex curves for channel {channel_idx}: {e}")
    
    def plot_maxH(self, channel_indices: List[int] = None, method=None, relative_to_mmax=False, 
                  manual_mmax=None, max_stim_value=None, bin_margin=0, interactive_cursor=None, canvas=None):
        """Plot max H-reflex data with interactive features."""
        try:
            if canvas is None:
                raise UnableToPlotError("Canvas is required for plotting.")
            
            # Set method to default if not specified
            if method is None:
                method = self.emg_object.default_method
            
            # Get channel information
            if channel_indices is None:
                channel_indices = list(range(self.emg_object.num_channels))
            
            # Create plot layout
            plot_items, layout = self.create_plot_layout(canvas, channel_indices)
            
            # Raw data collection
            raw_data_dict = {
                'channel_index': [],
                'stimulus_v': [],
                'm_wave_amplitudes': [],
                'h_wave_amplitudes': [],
            }
            
            # Process each channel
            for plot_idx, channel_index in enumerate(channel_indices):
                if channel_index >= self.emg_object.num_channels:
                    continue
                    
                plot_item = plot_items[plot_idx]
                
                # Get average H-wave amplitudes to find max H voltage
                h_response_means, _ = self.emg_object.get_avg_h_wave_amplitudes(method, channel_index)
                stimulus_voltages = self.emg_object.stimulus_voltages

                # Filter out stimulus voltages greater than max_stim_value if specified
                if max_stim_value is not None:
                    mask = stimulus_voltages <= max_stim_value
                    stimulus_voltages = stimulus_voltages[mask]
                    h_response_means = h_response_means[mask]

                # Find the voltage with the maximum average H-reflex amplitude
                max_h_reflex_idx = np.argmax(h_response_means)
                max_h_reflex_voltage = stimulus_voltages[max_h_reflex_idx]
                
                # Define the range of voltages around the max H-reflex voltage
                start_idx = max(0, max_h_reflex_idx - bin_margin)
                end_idx = min(len(stimulus_voltages), max_h_reflex_idx + bin_margin + 1)
                voltage_indices = range(start_idx, end_idx)
                marginal_voltages = stimulus_voltages[voltage_indices]

                # Collect M-wave and H-response amplitudes for the marginal bins
                stimulus_voltages_for_channel = []
                m_wave_amplitudes = []
                h_response_amplitudes = []
                for voltage in marginal_voltages:
                    m_waves = self.emg_object.get_m_wave_amplitudes_at_voltage(method, channel_index, voltage)
                    h_responses = self.emg_object.get_h_wave_amplitudes_at_voltage(method, channel_index, voltage)
                    m_wave_amplitudes.extend(m_waves)
                    h_response_amplitudes.extend(h_responses)
                    stimulus_voltages_for_channel.extend([voltage] * len(m_waves))

                # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified
                if relative_to_mmax:
                    if manual_mmax is not None:
                        m_max = manual_mmax[channel_index] if isinstance(manual_mmax, list) else manual_mmax
                    else:
                        m_max = self.emg_object.get_avg_m_max(method=method, channel_index=channel_index)
                    
                    if m_max and m_max != 0:
                        m_wave_amplitudes = m_wave_amplitudes / m_max
                        h_response_amplitudes = h_response_amplitudes / m_max
                    else:
                        raise UnableToPlotError(f'M-max could not be calculated or is zero for channel {channel_index}.')

                # Append data to raw data dictionary
                raw_data_dict['channel_index'].extend([channel_index] * len(m_wave_amplitudes))
                raw_data_dict['stimulus_v'].extend(stimulus_voltages_for_channel)
                raw_data_dict['m_wave_amplitudes'].extend(m_wave_amplitudes)
                raw_data_dict['h_wave_amplitudes'].extend(h_response_amplitudes)

                # Plot the M-wave and H-response amplitudes
                m_x = 1.0
                h_x = 2.5
                
                # Convert colors
                m_color = self._convert_matplotlib_color(self.emg_object.m_color)
                h_color = self._convert_matplotlib_color(self.emg_object.h_color)
                
                # Plot M-wave data points
                if m_wave_amplitudes:                   
                    # Calculate and plot mean with error bars
                    mean_m = np.mean(m_wave_amplitudes)
                    std_m = np.std(m_wave_amplitudes)
                    
                    # Add error bars
                    error_bars_m = pg.ErrorBarItem(
                        x=np.array([m_x]), y=np.array([mean_m]),
                        top=np.array([std_m]), bottom=np.array([std_m]),
                        pen=pg.mkPen('white', width=2),
                        beam=0.2
                    )
                    plot_item.addItem(error_bars_m)
                    
                    # Add mean marker
                    mean_marker_m = pg.ScatterPlotItem(
                        x=np.array([m_x]), y=np.array([mean_m]),
                        pen=pg.mkPen('white', width=2),
                        brush=pg.mkBrush('white'),
                        size=12,
                        symbol='+'
                    )
                    plot_item.addItem(mean_marker_m)

                    # Plot the scatter points for M-wave
                    m_scatter = pg.ScatterPlotItem(
                        x=np.array([m_x] * len(m_wave_amplitudes)),
                        y=np.array(m_wave_amplitudes),
                        pen=pg.mkPen('white', width=0.1),
                        brush=pg.mkBrush(m_color),
                        size=8,
                        symbol='o'
                    )
                    plot_item.addItem(m_scatter)
                    
                    # Add annotation text
                    text_m = pg.TextItem(
                        f'n={len(m_wave_amplitudes)}\navg. = {mean_m:.2f}\nstd. = {std_m:.2f}',
                        anchor=(0, 0.5), color=m_color, border=pg.mkPen('w'), fill=pg.mkBrush(255, 255, 255, 180)
                    )
                    text_m.setPos(m_x + 0.4, mean_m)
                    plot_item.addItem(text_m)
                
                # Plot H-response data points
                if h_response_amplitudes:                   
                    # Calculate and plot mean with error bars
                    mean_h = np.mean(h_response_amplitudes)
                    std_h = np.std(h_response_amplitudes)
                    
                    # Add error bars
                    error_bars_h = pg.ErrorBarItem(
                        x=np.array([h_x]), y=np.array([mean_h]),
                        top=std_h, bottom=std_h,
                        pen=pg.mkPen('white', width=2),
                        beam=0.2
                    )
                    plot_item.addItem(error_bars_h)
                    
                    # Add mean marker
                    mean_marker_h = pg.ScatterPlotItem(
                        x=np.array([h_x]), y=np.array([mean_h]),
                        pen=pg.mkPen('white', width=2),
                        brush=pg.mkBrush('white'),
                        size=12,
                        symbol='+'
                    )
                    plot_item.addItem(mean_marker_h)

                    # Plot the scatter points for H-response
                    h_scatter = pg.ScatterPlotItem(
                        x=np.array([h_x] * len(h_response_amplitudes)),
                        y=np.array(h_response_amplitudes),
                        pen=pg.mkPen('white', width=0.1),
                        brush=pg.mkBrush(h_color),
                        size=8,
                        symbol='o'
                    )
                    plot_item.addItem(h_scatter)
                    
                    # Add annotation text
                    text_h = pg.TextItem(
                        f'n={len(h_response_amplitudes)}\navg. = {mean_h:.2f}\nstd. = {std_h:.2f}',
                        anchor=(0, 0.5), color=h_color, border=pg.mkPen('w'), fill=pg.mkBrush(255, 255, 255, 180)
                    )
                    text_h.setPos(h_x + 0.4, mean_h)
                    plot_item.addItem(text_h)
                
                # Set labels and formatting
                channel_name = self.emg_object.channel_names[channel_index]
                voltage_range = round((self.emg_object.bin_size/2) + (self.emg_object.bin_size * bin_margin), 2)
                title = f'{channel_name} ({round(max_h_reflex_voltage, 2)} ± {voltage_range}V)'
                
                y_label = f'EMG Amp. (mV, {method})'
                if relative_to_mmax:
                    y_label = f'EMG Amp. (M-max, {method})'
                
                self.set_labels(plot_item, title=title, x_label='Response Type', y_label=y_label)
                
                # Set x-axis ticks and labels
                plot_item.getAxis('bottom').setTicks([[(m_x, 'M-response'), (h_x, 'H-reflex')]])
                plot_item.setXRange(m_x - 1, h_x + 1)
                
                # Enable grid
                plot_item.showGrid(True, True)
            
            for plot_item in plot_items:
                plot_item.enableAutoRange(axis='y', enable=True)
            
            # Display the plot
            self.display_plot(canvas)
            
            # Create DataFrame with multi-level index
            raw_data_df = pd.DataFrame(raw_data_dict)
            raw_data_df.set_index(['channel_index', 'stimulus_v'], inplace=True)
            return raw_data_df
            
        except Exception as e:
            raise UnableToPlotError(f"Error plotting max H-reflex: {str(e)}")

    def plot_mmax(self, channel_indices: List[int] = None, method: str = None, interactive_cursor: bool = False, canvas=None):
        """Plot M-max data with interactive features."""
        try:
            if canvas is None:
                raise UnableToPlotError("Canvas is required for plotting.")
            
            # Set method to default if not specified
            if method is None:
                method = self.emg_object.default_method
            
            # Get channel information
            if channel_indices is None:
                channel_indices = list(range(self.emg_object.num_channels))
            
            # Create plot layout
            plot_items, layout = self.create_plot_layout(canvas, channel_indices)
            
            # Raw data collection
            raw_data_dict = {
                'channel_index': [],
                'session_id': [],
                'm_max_threshold': [],
                'm_max_amplitude': [],
            }
            
            all_m_max_amplitudes = []
            
            # Process each channel
            for plot_idx, channel_index in enumerate(channel_indices):
                if channel_index >= self.emg_object.num_channels:
                    continue
                    
                plot_item = plot_items[plot_idx]
                
                # Get M-max data for each session
                m_max_amplitudes = []
                m_max_thresholds = []
                session_ids = []
                
                for session in self.emg_object.sessions:
                    try:
                        m_max, mmax_low_stim, _ = session.get_m_max(
                            method=method,
                            channel_index=channel_index,
                            return_mmax_stim_range=True,
                        )
                    except (ValueError, AttributeError):
                        m_max = np.nan
                        mmax_low_stim = np.nan
                    
                    m_max_amplitudes.append(m_max)
                    m_max_thresholds.append(mmax_low_stim)
                    session_ids.append(session.id)
                
                # Filter out NaN values for plotting
                valid_amplitudes = [amp for amp in m_max_amplitudes if not np.isnan(amp)]
                valid_thresholds = [thr for thr in m_max_thresholds if not np.isnan(thr)]
                
                # Append to superlist for y-axis adjustment
                all_m_max_amplitudes.extend(valid_amplitudes)
                
                # Calculate statistics
                mean_amp = np.mean(valid_amplitudes) if valid_amplitudes else np.nan
                std_amp = np.std(valid_amplitudes) if valid_amplitudes else np.nan
                mean_thresh = np.mean(valid_thresholds) if valid_thresholds else np.nan
                
                # Plot the data
                m_x = 1.0
                m_color = self._convert_matplotlib_color(self.emg_object.m_color)
                
                
                if valid_amplitudes:                
                    # Add error bars
                    error_bars = pg.ErrorBarItem(
                        x=np.array([m_x]), y=np.array([mean_amp]),
                        top=np.array([std_amp]), bottom=np.array([std_amp]),
                        pen=pg.mkPen('white', width=2),
                        beam=0.2
                    )
                    plot_item.addItem(error_bars)
                    
                    # Add mean marker
                    mean_marker = pg.ScatterPlotItem(
                        x=np.array([m_x]), y=np.array([mean_amp]),
                        pen=pg.mkPen('white', width=2),
                        brush=pg.mkBrush('white'),
                        size=12,
                        symbol='+'
                    )
                    plot_item.addItem(mean_marker)
                    
                    # Plot the scatter points
                    scatter = pg.ScatterPlotItem(
                        x=np.array([m_x] * len(valid_amplitudes)),
                        y=np.array(valid_amplitudes),
                        pen=pg.mkPen('white', width=0.1),
                        brush=pg.mkBrush(m_color),
                        size=8,
                        symbol='o'
                    )
                    plot_item.addItem(scatter)
                    
                    # Add annotation text
                    text = pg.TextItem(
                        f'n={len(valid_amplitudes)}\nAvg. M-max: {mean_amp:.2f}mV\nStdev. M-Max: {std_amp:.2f}mV\nAvg. Stim.: above {mean_thresh:.2f} mV',
                        anchor=(0, 0.5), color='black', border=pg.mkPen('w'), fill=pg.mkBrush(255, 255, 255, 180)
                    )
                    text.setPos(m_x + 0.2, mean_amp)
                    plot_item.addItem(text)
                
                # Set labels and formatting
                channel_name = self.emg_object.channel_names[channel_index]
                self.set_labels(plot_item, title=f'{channel_name}', x_label='Response Type', y_label=f'M-max (mV, {method})')
                
                # Set x-axis ticks and labels
                plot_item.getAxis('bottom').setTicks([[(m_x, 'M-response')]])
                plot_item.setXRange(m_x - 1, m_x + 1.5)
                
                # Set y-axis limits
                if all_m_max_amplitudes:
                    y_max = np.nanmax(all_m_max_amplitudes)
                    if not np.isnan(y_max):
                        plot_item.setYRange(0, 1.1 * y_max)
                
                # Enable grid
                plot_item.showGrid(True, True)
                
                # Append data to raw data dictionary
                datapoints = len(m_max_amplitudes)
                raw_data_dict['channel_index'].extend([channel_index] * datapoints)
                raw_data_dict['session_id'].extend(session_ids)
                raw_data_dict['m_max_threshold'].extend(m_max_thresholds)
                raw_data_dict['m_max_amplitude'].extend(m_max_amplitudes)

            # Display the plot
            self.display_plot(canvas)
            
            # Create DataFrame with multi-level index
            raw_data_df = pd.DataFrame(raw_data_dict)
            raw_data_df.set_index(['channel_index'], inplace=True)
            return raw_data_df
            
        except Exception as e:
            raise UnableToPlotError(f"Error plotting M-max: {str(e)}")
    
