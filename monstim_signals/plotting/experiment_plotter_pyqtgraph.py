import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from typing import TYPE_CHECKING, List
from .base_plotter_pyqtgraph import BasePlotterPyQtGraph, UnableToPlotError

if TYPE_CHECKING:
    from monstim_signals.domain.experiment import Experiment


class ExperimentPlotterPyQtGraph(BasePlotterPyQtGraph):
    """
    PyQtGraph-based plotter for Experiment data with interactive features.
    
    This class provides interactive plotting capabilities for EMG experiment data:
    - Real-time zooming and panning
    - Interactive latency window selection
    - Crosshair cursor for measurements
    - Multi-channel plotting support
    """
    
    def __init__(self, experiment: 'Experiment'):
        super().__init__(experiment)
        self.emg_object: 'Experiment' = experiment
        
    def plot_reflexCurves(self, channel_indices: List[int] = None, method=None, plot_legend=True, 
                         relative_to_mmax=False, manual_mmax=None, canvas=None):
        """Plot average reflex curves for the experiment with interactive features."""
        try:
            if canvas is None:
                raise UnableToPlotError("Canvas is required for plotting.")
            
            if method is None:
                method = self.emg_object.default_method
                        
            # Get channel information
            if channel_indices is None:
                channel_indices = list(range(self.emg_object.num_channels))
            
            plot_items, layout = self.create_plot_layout(canvas, channel_indices)
            
            # Raw data collection
            raw_data_dict = {
                'channel_index': [],
                'stimulus_v': [],
                'avg_m_wave': [],
                'stderr_m_wave': [],
                'avg_h_wave': [],
                'stderr_h_wave': []
            }
            
            for plot_item, channel_idx in zip(plot_items, channel_indices):
                # Set labels
                y_label = f'Average Reflex Ampl. (mV{", rel. to M-max" if relative_to_mmax else ""})'
                self.set_labels(plot_item=plot_item, title=f'Channel {channel_idx + 1}',
                                y_label=y_label, x_label='Stimulus Intensity (V)')
                plot_item.showGrid(True, True)
                
                # Plot reflex curves data and collect raw data
                self._plot_reflex_curves_data(plot_item, channel_idx, method, relative_to_mmax, manual_mmax, raw_data_dict)
            
            # Add legend if requested
            if plot_legend and plot_items:
                self.add_legend(plot_items[0])
            
            # Enable auto-range for all plots
            for plot_item in plot_items:
                plot_item.enableAutoRange(axis='y', enable=True)
            
            # Display the plot
            self.display_plot(canvas)
            
            # Create DataFrame with multi-level index
            raw_data_df = pd.DataFrame(raw_data_dict)
            raw_data_df.set_index(['channel_index', 'stimulus_v'], inplace=True)
            return raw_data_df
            
        except Exception as e:
            raise UnableToPlotError(f"Error plotting reflex curves: {str(e)}")

    def _plot_reflex_curves_data(self, plot_item: pg.PlotItem, channel_idx: int, method: str, relative_to_mmax: bool, manual_mmax: float | None, raw_data_dict: dict):
        """Plot reflex curves data for a specific channel."""
        try:
            # Get M-wave and H-reflex data using the Experiment domain object API
            m_wave_means, m_wave_error = self.emg_object.get_avg_m_wave_amplitudes(method, channel_idx)
            h_response_means, h_response_error = self.emg_object.get_avg_h_wave_amplitudes(method, channel_idx)
            
            # Get stimulus voltages
            stimulus_voltages = self.emg_object.stimulus_voltages

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified
            if relative_to_mmax:
                if manual_mmax is not None:
                    m_max_amplitude = manual_mmax
                else:
                    m_max_amplitude = self.emg_object.get_avg_m_max(method, channel_idx)
                    
                if m_max_amplitude and m_max_amplitude > 0:
                    m_wave_means = m_wave_means / m_max_amplitude
                    m_wave_error = m_wave_error / m_max_amplitude

            # Append data to raw data dictionary
            raw_data_dict['channel_index'].extend([channel_idx] * len(stimulus_voltages))
            raw_data_dict['stimulus_v'].extend(stimulus_voltages)
            raw_data_dict['avg_m_wave'].extend(m_wave_means)
            raw_data_dict['stderr_m_wave'].extend(m_wave_error)
            raw_data_dict['avg_h_wave'].extend(h_response_means)
            raw_data_dict['stderr_h_wave'].extend(h_response_error)

            # Plot M-wave curve with error bands
            m_color = self._convert_matplotlib_color(self.emg_object.m_color)
            m_pale_color = self._pale_color(m_color, blend=0.25)
            
            # Plot M-wave error bands
            m_upper = m_wave_means + m_wave_error
            m_lower = m_wave_means - m_wave_error
            m_upper_curve = plot_item.plot(stimulus_voltages, m_upper, 
                                         pen=pg.mkPen(color=m_pale_color, width=1, style=QtCore.Qt.PenStyle.DotLine))
            m_lower_curve = plot_item.plot(stimulus_voltages, m_lower, 
                                         pen=pg.mkPen(color=m_pale_color, width=1, style=QtCore.Qt.PenStyle.DotLine))
            
            # Create fill between curves for M-wave
            if not hasattr(plot_item, '_fill_curves_refs'):
                plot_item._fill_curves_refs = []
            plot_item._fill_curves_refs.extend([m_upper_curve, m_lower_curve])
            m_fill = pg.FillBetweenItem(curve1=m_upper_curve, curve2=m_lower_curve, 
                                      brush=pg.mkBrush(color=m_pale_color, alpha=50))
            plot_item.addItem(m_fill)

            # Plot M-wave mean line (on top of fill)
            plot_item.plot(stimulus_voltages, m_wave_means, 
                         pen=pg.mkPen(color=m_color, width=2),
                         symbol='o', symbolSize=6, symbolBrush=m_color,
                         name='M-wave')

            # Plot H-reflex curve with error bands
            h_color = self._convert_matplotlib_color(self.emg_object.h_color)
            h_pale_color = self._pale_color(h_color, blend=0.25)
            
            # Plot H-reflex error bands
            h_upper = h_response_means + h_response_error
            h_lower = h_response_means - h_response_error
            h_upper_curve = plot_item.plot(stimulus_voltages, h_upper, 
                                         pen=pg.mkPen(color=h_pale_color, width=1, style=QtCore.Qt.PenStyle.DotLine))
            h_lower_curve = plot_item.plot(stimulus_voltages, h_lower, 
                                         pen=pg.mkPen(color=h_pale_color, width=1, style=QtCore.Qt.PenStyle.DotLine))
            
            # Create fill between curves for H-reflex
            plot_item._fill_curves_refs.extend([h_upper_curve, h_lower_curve])
            h_fill = pg.FillBetweenItem(curve1=h_upper_curve, curve2=h_lower_curve, 
                                      brush=pg.mkBrush(color=h_pale_color, alpha=50))
            plot_item.addItem(h_fill)

            # Plot H-reflex mean line (on top of fill)
            plot_item.plot(stimulus_voltages, h_response_means,
                         pen=pg.mkPen(color=h_color, width=2),
                         symbol='o', symbolSize=6, symbolBrush=h_color,
                         name='H-reflex')
                                 
        except Exception as e:
            logging.warning(f"Could not plot reflex curves for channel {channel_idx}: {e}")
    
    def plot_maxH(self, channel_indices: List[int] = None, method=None, relative_to_mmax=False, 
                  manual_mmax=None, max_stim_value=None, bin_margin=0, canvas=None):
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
            
            plot_items, layout = self.create_plot_layout(canvas, channel_indices)
            
            # Raw data collection
            raw_data_dict = {
                'channel_index': [],
                'stimulus_v': [],
                'avg_m_wave_amplitudes': [],
                'avg_h_wave_amplitudes': [],
            }
            
            for plot_item, channel_idx in zip(plot_items, channel_indices):
                plot_item.showGrid(True, True)
                
                # Plot max H data and collect raw data
                self._plot_max_h_data(plot_item, channel_idx, method, relative_to_mmax, 
                                    manual_mmax, max_stim_value, bin_margin, raw_data_dict)
            
            
            # Enable auto-range for all plots
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
    
    def _plot_max_h_data(self, plot_item, channel_idx, method, relative_to_mmax, 
                        manual_mmax, max_stim_value, bin_margin, raw_data_dict):
        """Plot max H-reflex data for a specific channel."""
        try:
            # Get average H-wave amplitudes to find max H voltage
            h_response_means, _ = self.emg_object.get_avg_h_wave_amplitudes(method, channel_idx)
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
            m_wave_amplitudes = np.array([])
            h_response_amplitudes = np.array([])
            stimulus_voltages_for_channel = np.array([])
            for voltage in marginal_voltages:
                m_amplitudes = self.emg_object.get_m_wave_amplitude_avgs_at_voltage(method, channel_idx, voltage)
                h_amplitudes = self.emg_object.get_h_wave_amplitude_avgs_at_voltage(method, channel_idx, voltage)
                m_wave_amplitudes = np.concatenate((m_wave_amplitudes, m_amplitudes))
                h_response_amplitudes = np.concatenate((h_response_amplitudes, h_amplitudes))
                stimulus_voltages_for_channel = np.concatenate((stimulus_voltages_for_channel, [voltage] * len(m_amplitudes)))

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified
            if relative_to_mmax:
                if manual_mmax is not None:
                    m_max_amplitude = manual_mmax[channel_idx]
                else:
                    m_max_amplitude = self.emg_object.get_avg_m_max(method, channel_idx)
                    
                if m_max_amplitude and m_max_amplitude > 0:
                    m_wave_amplitudes = m_wave_amplitudes / m_max_amplitude
                    h_response_amplitudes = h_response_amplitudes / m_max_amplitude

            # Append data to raw data dictionary
            raw_data_dict['channel_index'].extend([channel_idx] * len(m_wave_amplitudes))
            raw_data_dict['stimulus_v'].extend(stimulus_voltages_for_channel)
            raw_data_dict['avg_m_wave_amplitudes'].extend(m_wave_amplitudes)
            raw_data_dict['avg_h_wave_amplitudes'].extend(h_response_amplitudes)

            # Plot the M-wave and H-response amplitudes
            m_x = 1.0
            h_x = 2.5
            
            # Convert colors
            m_color = self._convert_matplotlib_color(self.emg_object.m_color)
            h_color = self._convert_matplotlib_color(self.emg_object.h_color)
            
            # Plot M-wave data points
            if len(m_wave_amplitudes) > 0:                   
                # Calculate and plot mean with error bars
                mean_m = np.mean(m_wave_amplitudes)
                std_m = np.std(m_wave_amplitudes)
                
                # Add error bars
                error_bars_m = pg.ErrorBarItem(
                    x=np.array([m_x]), y=np.array([mean_m]),
                    top=std_m, bottom=std_m,
                    pen=pg.mkPen('white', width=2),
                    beam=0.2
                )
                plot_item.addItem(error_bars_m)
                
                # Add mean marker
                mean_marker_m = pg.ScatterPlotItem(
                    x=[m_x], y=[mean_m],
                    pen=pg.mkPen('white', width=2),
                    brush=pg.mkBrush('white'),
                    size=12,
                    symbol='+'
                )
                plot_item.addItem(mean_marker_m)

                # Plot the scatter points for M-wave
                m_scatter = pg.ScatterPlotItem(
                    x=[m_x] * len(m_wave_amplitudes),
                    y=m_wave_amplitudes,
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
            if len(h_response_amplitudes) > 0:                   
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
                    x=[h_x], y=[mean_h],
                    pen=pg.mkPen('white', width=2),
                    brush=pg.mkBrush('white'),
                    size=12,
                    symbol='+'
                )
                plot_item.addItem(mean_marker_h)

                # Plot the scatter points for H-response
                h_scatter = pg.ScatterPlotItem(
                    x=[h_x] * len(h_response_amplitudes),
                    y=h_response_amplitudes,
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
            voltage_range = round((self.emg_object.bin_size/2) + (self.emg_object.bin_size * bin_margin), 2) if hasattr(self.emg_object, 'bin_size') else 0.5
            title = f'Channel {channel_idx + 1} ({round(max_h_reflex_voltage, 2)} ± {voltage_range}V)'
            
            y_label = f'EMG Amp. (mV, {method})'
            if relative_to_mmax:
                y_label = f'EMG Amp. (M-max, {method})'
            
            self.set_labels(plot_item, title=title, x_label='Response Type', y_label=y_label)
            
            # Set x-axis ticks and labels
            plot_item.getAxis('bottom').setTicks([[(m_x, 'M-response'), (h_x, 'H-reflex')]])
            plot_item.setXRange(m_x - 1, h_x + 1)
            
        except Exception as e:
            print(f"Warning: Could not plot max H data for channel {channel_idx}: {e}")
    
    def plot_mmax(self, channel_indices: List[int] = None, method: str = None, interactive_cursor: bool =False, canvas=None):
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
            
            plot_items, layout = self.create_plot_layout(canvas, channel_indices)
            
            # Raw data collection
            raw_data_dict = {
                'channel_index': [],
                'animal_id': [],
                'avg_m_max_threshold': [],
                'avg_m_max_amplitude': [],
            }
            
            for plot_item, channel_idx in zip(plot_items, channel_indices):
                plot_item.showGrid(True, True)
                
                # Plot M-max data and collect raw data
                self._plot_mmax_data(plot_item, channel_idx, method, raw_data_dict)
            
            # Add synchronized crosshairs
            if interactive_cursor:
                self.add_synchronized_crosshairs(plot_items)
            
            # Enable auto-range for all plots
            for plot_item in plot_items:
                plot_item.enableAutoRange(axis='y', enable=True)
            
            # Display the plot
            self.display_plot(canvas)
            
            # Create DataFrame with multi-level index
            raw_data_df = pd.DataFrame(raw_data_dict)
            raw_data_df.set_index(['channel_index'], inplace=True)
            return raw_data_df
            
        except Exception as e:
            raise UnableToPlotError(f"Error plotting M-max: {str(e)}")
    
    def _plot_mmax_data(self, plot_item, channel_idx, method, raw_data_dict):
        """Plot M-max data for a specific channel."""
        try:
            # Iterate through datasets to get M-max data
            mmax_amplitudes = []
            animal_ids = []
            for dataset in self.emg_object.emg_datasets:
                m_max_amplitude = dataset.get_avg_m_max(method, channel_idx)
                mmax_amplitudes.append(m_max_amplitude)
                animal_ids.append(dataset.id)

            # Filter out None/NaN values for plotting
            valid_amplitudes = [amp for amp in mmax_amplitudes if amp is not None and not np.isnan(amp)]
            
            # Calculate statistics
            mean_amp = np.mean(valid_amplitudes) if valid_amplitudes else np.nan
            std_amp = np.std(valid_amplitudes) if valid_amplitudes else np.nan

            # Append data to raw data dictionary
            raw_data_dict['channel_index'].extend([channel_idx] * len(mmax_amplitudes))
            raw_data_dict['animal_id'].extend(animal_ids)
            raw_data_dict['avg_m_max_threshold'].extend([None] * len(mmax_amplitudes))  # Not available at experiment level
            raw_data_dict['avg_m_max_amplitude'].extend(mmax_amplitudes)

            # Plot the data
            m_x = 1.0
            m_color = self._convert_matplotlib_color(self.emg_object.m_color)
            
            if valid_amplitudes:                
                # Add error bars
                error_bars = pg.ErrorBarItem(
                    x=np.array([m_x]), y=np.array([mean_amp]),
                    top=std_amp, bottom=std_amp,
                    pen=pg.mkPen('white', width=2),
                    beam=0.2
                )
                plot_item.addItem(error_bars)
                
                # Add mean marker
                mean_marker = pg.ScatterPlotItem(
                    x=[m_x], y=[mean_amp],
                    pen=pg.mkPen('white', width=2),
                    brush=pg.mkBrush('white'),
                    size=12,
                    symbol='+'
                )
                plot_item.addItem(mean_marker)
                
                # Plot the scatter points
                scatter = pg.ScatterPlotItem(
                    x=[m_x] * len(valid_amplitudes),
                    y=valid_amplitudes,
                    pen=pg.mkPen('white', width=0.1),
                    brush=pg.mkBrush(m_color),
                    size=8,
                    symbol='o'
                )
                plot_item.addItem(scatter)
                
                # Add annotation text
                text = pg.TextItem(
                    f'n={len(valid_amplitudes)}\nAvg. M-max: {mean_amp:.2f}mV\nStdev. M-Max: {std_amp:.2f}mV',
                    anchor=(0, 0.5), color='black', border=pg.mkPen('w'), fill=pg.mkBrush(255, 255, 255, 180)
                )
                text.setPos(m_x + 0.2, mean_amp)
                plot_item.addItem(text)
            
            # Set labels and formatting
            self.set_labels(plot_item, title=f'Channel {channel_idx + 1}', x_label='Response Type', y_label=f'M-max (mV, {method})')
            
            # Set x-axis ticks and labels
            plot_item.getAxis('bottom').setTicks([[(m_x, 'M-response')]])
            plot_item.setXRange(m_x - 1, m_x + 1.5)
            
        except Exception as e:
            print(f"Warning: Could not plot M-max data for channel {channel_idx}: {e}")

