import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from matplotlib.lines import Line2D
from typing import TYPE_CHECKING, List
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from monstim_signals.plotting.base_plotter import BasePlotter, UnableToPlotError
from monstim_signals import transform

if TYPE_CHECKING:
    from monstim_signals.domain.session import Session

class SessionPlotter(BasePlotter):
    """
    A class for plotting EMG data from an EMGSession object.

    Args:
        data (EMGSession): The EMGSession object to be imported containing the EMG data.

    Attributes:
        session (EMGSession): The imported EMGSession object containing the EMG data.

    Methods:
        plot_emg: Plots EMG data for a specified time window.
        plot_emg_suspectedH: Detects and plots session recordings with potential H-reflexes.
        plot_reflex_curves: Plots overlayed M-response and H-reflex curves for each recorded channel.
        plot_m_curves_smoothened: Plots overlayed M-response and H-reflex curves for each recorded channel, smoothened using a Savitzky-Golay filter.
        help: Displays the help text for the class.

    """

    def __init__(self, session : 'Session'):
        """
        Initializes an instance of the EMGSessionPlotter class.

        Args:
            session (EMGSession): The EMGSession object to be imported containing the EMG data.

        Returns:
            None
        """
        from monstim_signals.domain.session import Session
        if not isinstance(session, Session):
            raise TypeError("The session parameter must be an instance of the Session class.")
        self.emg_object : 'Session' = session # The EMGSession object to be imported containing the EMG data.
        super().__init__(self.emg_object)
        self.set_plot_defaults()
        
    def help(self):
        help_text = """
        EMGSessionPlotter class for plotting EMG data from an EMGSession object.
        ========================================================================
        Methods:
        1. plot_emg: Plots EMG data for a specified time window.
            -Example:
                plot_emg(channel_names=['Channel 1', 'Channel 2'], m_flags=True, h_flags=True, data_type='filtered')
        
        2. plot_emg_suspectedH: Detects and plots session recordings with potential H-reflexes.
            -Example:
                plot_emg_suspectedH(channel_names=['Channel 1', 'Channel 2'], h_threshold=0.3, plot_legend=True)
        
        3. plot_reflex_curves: Plots overlayed M-response and H-reflex curves for each recorded channel.
            -Example:
                plot_reflex_curves(channel_names=['Channel 1', 'Channel 2'], method='rms', relative_to_mmax=False, manual_mmax=None)
       
        4. plot_m_curves_smoothened: Plots overlayed M-response and H-reflex curves for each recorded channel, smoothened using a Savitzky-Golay filter.
            -Example:
                plot_m_curves_smoothened(channel_names=['Channel 1', 'Channel 2'], method='rms', relative_to_mmax=False, manual_mmax=None)
        
        5. help: Displays this help text.
        ========================================================================

        """
        print(help_text)

    def get_time_axis(self):
        # Get pre_stim_time from session config with fallback
        pre_stim_time = getattr(self.emg_object, 'pre_stim_time_ms', 2.0)
        
        # Calculate time values based on the scan rate
        time_values_ms = np.arange(self.emg_object.num_samples) * 1000 / self.emg_object.scan_rate  # Time values in milliseconds
        
        # Define the start and end times for the window
        # Start pre_stim_time ms before stimulus onset
        window_start_time = self.emg_object.stim_start - pre_stim_time
        # End time_window_ms after stimulus onset (not affected by pre_stim_time)
        window_end_time = self.emg_object.stim_start + self.emg_object.time_window_ms

        # Convert time window to sample indices
        window_start_sample = int(window_start_time * self.emg_object.scan_rate / 1000)
        window_end_sample = int(window_end_time * self.emg_object.scan_rate / 1000)

        # Slice the time array for the time window
        time_axis = time_values_ms[window_start_sample:window_end_sample] - self.emg_object.stim_start
        return time_axis, window_start_sample, window_end_sample

    def get_emg_recordings(self, data_type, original=False):
        """
        Get the EMG recordings based on the specified data type.

        Parameters:
        - data_type (str): The type of EMG data to retrieve. Valid options are 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered'.
        - original (bool): Flag to retrieve the original EMG recordings. Default is False.

        Returns:
        - list: The EMG recordings based on the specified data type.

        Raises:
        - ValueError: If the specified data type is not supported.

        """
        if original:
            raise NotImplementedError("Original recordings are not supported in EMGSessionPlotter. Please use the 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered' data types.")
        else: # For most uses, use the raw or filtered recordings
            if data_type in ['raw', 'filtered', 'rectified_raw', 'rectified_filtered']:
                attribute_name = f'recordings_{data_type}'
                data = getattr(self.emg_object, attribute_name)
                if data is None:
                    raise AttributeError(f"Data type '{attribute_name}' is not available in the Session object. Please ensure that the data has been processed and stored correctly.")
                return data
            else:
                raise ValueError(f"Data type '{data_type}' is not supported. Please use 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered'.")

    def plot_channel_data(self, ax, time_axis, channel_data, start, end, stimulus_v, channel_index, cmap='viridis_r', norm=None):
        # Create a colormap and normalize stimulus voltage
        colormap = plt.get_cmap(cmap)
        if norm is None:
            logging.warning("Normalization object not provided. Using default normalization.")
            norm = plt.Normalize(vmin=self.emg_object.stimulus_voltages[0], vmax=self.emg_object.stimulus_voltages[-1])
        
        # Get the color for the current stimulus voltage
        color = colormap(norm(stimulus_v))    
        
        line = ax.plot(time_axis, channel_data[start:end], color=color, label=f"Stimulus Voltage: {stimulus_v}")
        ax.set_title(f'{self.emg_object.channel_names[channel_index]}')
        ax.grid(True)

        # Store the norm and colormap as attributes of the axis for later use
        ax.norm = norm
        ax.colormap = colormap

        return line[0]
   
    def plot_latency_windows(self, ax, all_flags, channel_index):
        if all_flags:
            for window in self.emg_object.latency_windows:
                start_exists = end_exists = False
                for line in ax.lines:
                    if isinstance(line, Line2D):
                        if line.get_xdata()[0] == window.start_times[channel_index] and line.get_color() == window.color:
                            start_exists = True
                        elif line.get_xdata()[0] == window.end_times[channel_index] and line.get_color() == window.color:
                            end_exists = True
                        if start_exists and end_exists:
                            break
                
                if not start_exists:
                    ax.axvline(window.start_times[channel_index], color=window.color, linestyle=window.linestyle)
                
                if not end_exists:
                    ax.axvline(window.end_times[channel_index], color=window.color, linestyle=window.linestyle)

    def set_fig_labels_and_legends(self, fig, channel_indices : List[int], sup_title : str, x_title : str, y_title: str, plot_legend : bool, plot_colormap : bool = False, legend_elements : list = None):
        fig.suptitle(sup_title)
        
        # Get the norm and colormap from the first axis
        if len(fig.axes) > 0:
            ax = fig.axes[0]
            if hasattr(ax, 'norm') and hasattr(ax, 'colormap') and plot_colormap:
                # Create a scalar mappable for the colorbar
                sm = plt.cm.ScalarMappable(cmap=ax.colormap, norm=ax.norm)
                sm.set_array([])  # Empty array needed for the ScalarMappable
                
                # Add colorbar - position it based on the number of subplots
                if len(channel_indices) == 1:
                    fig.colorbar(sm, ax=ax, label='Stimulus Voltage (V)')
                else:
                    # add colorbar as a subplot to the right of the figure
                    data_axes = fig.axes
                    cbar = fig.colorbar(
                        sm,
                        ax=data_axes,
                        orientation='vertical',
                        fraction=0.02,   # width of cbar = 2% of figure width
                        pad=0.02         # padding between plot and bar = 2% of figure width
                    ) # Type: matplotlib.colorbar.Colorbar
                    cbar.set_label('Stimulus Voltage (V)')
                    # cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f V'))

        
        if len(channel_indices) == 1:
            fig.gca().set_xlabel(x_title)
            fig.gca().set_ylabel(y_title)
            if plot_legend and legend_elements:
                fig.gca().legend(handles=legend_elements, loc='best')
        else:
            fig.supxlabel(x_title)
            fig.supylabel(y_title)
            if plot_legend and legend_elements:
                fig.legend(handles=legend_elements, loc='upper right')

# Actual plotting functions for Session data
# TODO: implement stimuli plots for relevant plot types. May need to update which plot types receive the stimuli_to_plot parameter injection.
    def plot_emg(self, channel_indices : List[int] = None, all_flags : bool = True, plot_legend : bool = True, plot_colormap: bool = False, data_type : str = 'filtered', stimuli_to_plot: List[str] = None, canvas: FigureCanvas = None):
        """
        Plots EMG data from a Pickle file for a specified time window.

        Args:
            all_flags (bool): Flag to plot all latency windows. Default is True.
            m_flags (bool): Flag to plot markers for muscle onset and offset. Default is False.
            h_flags (bool): Flag to plot markers for hand onset and offset. Default is False.
            plot_legend (bool): Flag to plot the legend. Default is True.
            data_type (str): Type of EMG data to plot. Options are 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered'. Default is 'filtered'.
            canvas (FigureCanvas, optional): Canvas to draw on. If None, a new figure is created.

        Returns:
            None
        """
        if all_flags:
            plot_latency_windows = True
        else:
            plot_latency_windows = False

        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))
            num_channels = self.emg_object.num_channels
        else:
            num_channels = len(channel_indices)

        time_axis, window_start_sample, window_end_sample = self.get_time_axis()
        fig, ax, axes = self.create_fig_and_axes(channel_indices=channel_indices, canvas=canvas, figsizes='large')
        legend_elements = [window.get_legend_element() for window in self.emg_object.latency_windows] if plot_latency_windows else []
        emg_recordings = self.get_emg_recordings(data_type)

        # Create a normalization object for the stimulus voltages
        norm = plt.Normalize(vmin=min(self.emg_object.stimulus_voltages), vmax=max(self.emg_object.stimulus_voltages))

        # Initialize a list to store structured raw data
        raw_data_dict = {
            'recording_index': [],
            'channel_index': [],
            'stimulus_V': [],
            'time_point': [],
            'amplitude_mV': []
        }

        for recording_idx, recording in enumerate(emg_recordings):
            # stimulus_v = recording['stimulus_v']
            stimulus_v = self.emg_object.stimulus_voltages[recording_idx]
            for channel_index, channel_data in enumerate(recording.T):
                if channel_index not in channel_indices:
                    continue

                current_ax = ax if num_channels == 1 else axes[channel_indices.index(channel_index)]
                
                # Plot EMG data
                self.plot_channel_data(current_ax, time_axis, channel_data, window_start_sample, window_end_sample, stimulus_v, channel_index, norm=norm)
                self.plot_latency_windows(current_ax, all_flags, channel_index)

                # Collect raw data with hierarchical index structure
                num_points = len(time_axis)
                raw_data_dict['recording_index'].extend([recording_idx] * num_points)
                raw_data_dict['channel_index'].extend([channel_index] * num_points)
                raw_data_dict['stimulus_V'].extend([stimulus_v] * num_points)
                raw_data_dict['time_point'].extend(time_axis)

                # Add each individual EMG value for the current channel
                raw_data_dict['amplitude_mV'].extend(channel_data[window_start_sample:window_end_sample])
        
        # Sanity check to ensure all lists in the dictionary are of the same length
        lengths = [len(lst) for lst in raw_data_dict.values()]
        if len(set(lengths)) > 1:
            raise ValueError(f"Inconsistent lengths found in raw_data_dict: {lengths}")

        # Set labels and title, and display plot
        if num_channels == 1:
            sup_title = 'EMG Overlay for Channel 0 (all recordings)'
        else:
            sup_title = 'EMG Overlay for All Channels (all recordings)'
        x_title = 'Time (ms)'
        y_title = 'EMG (mV)'

        self.set_fig_labels_and_legends(fig, channel_indices, sup_title, x_title, y_title, plot_legend, legend_elements=legend_elements, plot_colormap=plot_colormap)
        self.display_plot(canvas)

        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(['recording_index', 'channel_index', 'stimulus_V', 'time_point'], inplace=True)
        return raw_data_df

###### Likely to find errors in code below this point, as it was not tested.
    def plot_singleEMG(self, channel_indices : List[int] = None, recording_index: int = 0, fixed_y_axis : bool = True, all_flags : bool = True, plot_legend: bool = True, plot_colormap : bool = False, data_type: str = 'filtered', canvas: FigureCanvas = None):
        """
        Plots EMG data for a single recording.

        Args:
            recording_index (int): Index of the recording to plot. Default is 0.
            plot_legend (bool): Flag to plot the legend. Default is True.
            data_type (str): Type of EMG data to plot. Options are 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered'. Default is 'filtered'.
            canvas (FigureCanvas, optional): Canvas to draw on. If None, a new figure is created.

        Returns:
            None
        """
        if all_flags:
            plot_latency_windows = True
        else:
            plot_latency_windows = False

        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))
            num_channels = self.emg_object.num_channels
        else:
            num_channels = len(channel_indices)
            
        raw_data_dict = {
            'channel_index': [],
            'stimulus_V': [],
            'time_point': [],
            'amplitude_mV': []
        }

        time_axis, window_start_sample, window_end_sample = self.get_time_axis()
        fig, ax, axes = self.create_fig_and_axes(channel_indices=channel_indices, canvas=canvas, figsizes='large')
        legend_elements = [window.get_legend_element() for window in self.emg_object.latency_windows] if plot_latency_windows else []
        emg_recordings = self.get_emg_recordings(data_type)

        if fixed_y_axis:
            max_y = [] # list to store maximum values for each channel
            min_y = [] # list to store minimum values for each channel
            for recording in emg_recordings:
                # recording.T returns channel data in the expected order (channels as rows, time points as columns).
                for channel_data in recording.T:
                    max_y.append(np.max(channel_data[window_start_sample:window_end_sample]))
                    min_y.append(np.min(channel_data[window_start_sample:window_end_sample]))
            y_max = max(max_y)
            y_min = min(min_y)
            y_range = y_max - y_min
            y_max += 0.01 * y_range
            y_min -= 0.01 * y_range

        if recording_index < 0 or recording_index >= len(emg_recordings):
            raise ValueError(f"Invalid recording index. Must be between 0 and {len(emg_recordings) - 1}")

        recording = emg_recordings[recording_index]
        stimulus_v = self.emg_object.stimulus_voltages[recording_index]

        for channel_index, channel_data in enumerate(recording.T):
            if channel_index not in channel_indices:
                continue

            current_ax = ax if num_channels == 1 else axes[channel_indices.index(channel_index)]

            self.plot_channel_data(current_ax, time_axis, channel_data, window_start_sample, window_end_sample, stimulus_v, channel_index)
            self.plot_latency_windows(current_ax, all_flags, channel_index)

            # Collect raw data with hierarchical index structure
            num_points = len(time_axis)
            raw_data_dict['channel_index'].extend([channel_index] * num_points)
            raw_data_dict['stimulus_V'].extend([stimulus_v] * num_points)
            raw_data_dict['time_point'].extend(time_axis)

            # Add each individual EMG value for the current channel
            raw_data_dict['amplitude_mV'].extend(channel_data[window_start_sample:window_end_sample])

            if fixed_y_axis:
                current_ax.set_ylim(y_min, y_max)

        if num_channels == 1:
            sup_title = f'EMG for Channel 0 (Recording {recording_index}, Stim. = {stimulus_v}V)'
        else:
            sup_title = f'EMG for All Channels (Recording {recording_index}, Stim. = {stimulus_v}V)'
        x_title = 'Time (ms)'
        y_title = 'EMG (mV)'

        self.set_fig_labels_and_legends(fig, channel_indices, sup_title, x_title, y_title, plot_legend, legend_elements=legend_elements, plot_colormap=plot_colormap)
        self.display_plot(canvas)

        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(['channel_index', 'stimulus_V', 'time_point'], inplace=True)
        return raw_data_df

    def plot_emg_thresholded (self, latency_window_oi_name : str, channel_indices : List[int] = None, emg_threshold_v : float = 0.3, method : str = None, all_flags : bool = False, plot_legend : bool = False, canvas : FigureCanvas = None):
        """
        Detects session recordings with potential H-reflexes and plots them.

        Args:
            channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
            h_threshold (float, optional): Detection threshold of the average rectified EMG response in millivolts in the H-relfex window. Defaults to 0.3mV.
            plot_legend (bool, optional): Whether to plot legends. Defaults to False.
        """
        if method is None:
            method = self.emg_object.default_method

        if all_flags:
            plot_latency_windows = True
        else:
            plot_latency_windows = False

        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))
            num_channels = self.emg_object.num_channels
        else:
            num_channels = len(channel_indices)

        time_axis, window_start_sample, window_end_sample = self.get_time_axis()
        fig, ax, axes = self.create_fig_and_axes(channel_indices=channel_indices, canvas=canvas, figsizes='large')
        # legend_elements = [window.get_legend_element() for window in self.emg_object.latency_windows] if plot_latency_windows else []
        emg_recordings = self.get_emg_recordings('filtered')

        latency_window_oi = next((window for window in self.emg_object.latency_windows 
                                  if window.name == latency_window_oi_name), None)
        if latency_window_oi is None:
            raise UnableToPlotError(f"No latency window named '{latency_window_oi_name}' found in the session.")

        # Plot the EMG arrays for each channel, only for the first 10ms
        for rec_idx, recording in enumerate(emg_recordings):
            stimulus_v = self.emg_object.stimulus_voltages[rec_idx]
            for channel_index, channel_data in enumerate(recording.T):
                if channel_index not in channel_indices:
                    continue
                current_ax = ax if num_channels == 1 else axes[channel_indices.index(channel_index)]
                latency_window_oi_amplitude = transform.calculate_emg_amplitude(
                    channel_data,
                    latency_window_oi.start_times[channel_index] + self.emg_object.stim_start,
                    latency_window_oi.end_times[channel_index] + self.emg_object.stim_start,
                    self.emg_object.scan_rate,
                    method=method,
                )
                if latency_window_oi_amplitude > emg_threshold_v:  # Check EMG amplitude within H-reflex window
                    self.plot_channel_data(
                        current_ax,
                        time_axis,
                        channel_data,
                        window_start_sample,
                        window_end_sample,
                        stimulus_v,
                        channel_index,
                    )
                    if plot_latency_windows:
                        self.plot_latency_windows(current_ax, all_flags=False, channel_index=channel_index)
                    if plot_legend:
                        current_ax.legend()

        # Set labels and title, and display plot
        sup_title = f'EMG Recordings Over Threshold ({emg_threshold_v} V) for {latency_window_oi_name}'
        x_title = 'Time (ms)'
        y_title = 'EMG (mV)'

        self.set_fig_labels_and_legends(fig, channel_indices, sup_title, x_title, y_title, plot_legend=False)
        self.display_plot(canvas)
    
    def plot_suspectedH(self, channel_indices : List[int] = None, h_threshold : float = 0.3, method : str = None, all_flags : bool = False, plot_legend : bool = False, canvas : FigureCanvas = None):
        '''Deprecated. Use plot_emg_thresholded instead.'''
        has_h_reflex_latency_window = False
        for window in self.emg_object.latency_windows:
            if window.name.lower() in ('h-reflex', 'h_reflex', 'h reflex', 'hreflex'):
                has_h_reflex_latency_window = True
                break
        if not has_h_reflex_latency_window:
            raise UnableToPlotError("No H-reflex latency window detected. Please add an H-reflex latency window to the session.")
        self.plot_emg_thresholded('H-reflex', channel_indices=channel_indices, emg_threshold_v=h_threshold, method=method, 
                                  all_flags=all_flags, plot_legend=plot_legend, canvas=canvas)

    def plot_mmax(self, channel_indices : List[int] = None, method : str = None, canvas : FigureCanvas = None):
        """
        Plots overlayed M-response and H-reflex curves for each recorded channel.

        Args:
            channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
            method (str, optional): The method used to calculate the mean and standard deviation. Options are 'rms', 'avg_rectified', or 'peak_to_trough'. Default is 'rms'.
        """
        # Set method to default if not specified.
        if method is None:
            method = self.emg_object.default_method

        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))
            num_channels = self.emg_object.num_channels
        else:
            num_channels = len(channel_indices)

        channel_names = self.emg_object.channel_names

        fig, ax, axes = self.create_fig_and_axes(channel_indices=channel_indices, canvas=canvas, figsizes = 'small')

        emg_recordings = self.get_emg_recordings('filtered')

        # Create a superlist to store all M-wave amplitudes for each channel for later y-axis adjustment.
        all_m_max_amplitudes = []

        raw_data_dict = {
            'channel_index': [],
            'm_max_threshold': [],
            'm_max_amplitudes': [],
        }
        
        # Plot the M-wave and H-response amplitudes for each channel
        for channel_index in range(self.emg_object.num_channels):
            if channel_index not in channel_indices:
                continue
            m_wave_amplitudes = []
            stimulus_voltages = []

            # Append the M-wave and H-response amplitudes for each recording into the superlist.
            for rec_idx, recording in enumerate(emg_recordings):
                channel_data = recording[:, channel_index]
                stimulus_v = self.emg_object.stimulus_voltages[rec_idx]
                             
                try:
                    m_wave_amplitude = transform.calculate_emg_amplitude(channel_data,
                                                                             self.emg_object.m_start[channel_index] + self.emg_object.stim_start, 
                                                                             (self.emg_object.m_start[channel_index] + self.emg_object.m_duration[channel_index]) + self.emg_object.stim_start,
                                                                             self.emg_object.scan_rate, method=method)
                except ValueError:
                    raise UnableToPlotError(f"The method {method} is not supported. Please use 'rms', 'average_rectified', 'average_unrectified', or 'peak_to_trough'.")

                m_wave_amplitudes.append(m_wave_amplitude)
                stimulus_voltages.append(stimulus_v)
            
            # Convert superlists to numpy arrays.
            m_wave_amplitudes = np.array(m_wave_amplitudes)
            stimulus_voltages = np.array(stimulus_voltages)

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            m_max, mmax_low_stim, _ = transform.get_avg_mmax(stimulus_voltages, m_wave_amplitudes, return_mmax_stim_range=True, **self.emg_object.m_max_args)
            
            # Filter out M-wave amplitudes below the M-max stimulus threshold.
            mask = (stimulus_voltages >= mmax_low_stim)
            m_max_amplitudes = m_wave_amplitudes[mask]

            # Append M-wave amplitudes to superlist for y-axis adjustment.
            all_m_max_amplitudes.extend(m_max_amplitudes)

            m_x = 1 # Set x-axis position of the M-wave data.
            if num_channels == 1:
                ax.plot(m_x, [m_max_amplitudes], color=self.emg_object.m_color, marker='o', markersize=5)
                ax.annotate(f'n={len(m_max_amplitudes)}\nM-max: {m_max:.2f}mV\nStim.: above {mmax_low_stim:.2f}V', xy=(m_x + 0.2, np.mean(m_max_amplitudes) - 0.2), ha='left', va='center', color='black')
                ax.errorbar(m_x, np.mean(m_max_amplitudes), yerr=np.std(m_max_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                ax.set_xticklabels(['M-response'])
                ax.set_title(f'{channel_names[0]}')                    
                ax.set_xlim(m_x-1, m_x+1.5) # Set x-axis limits for each subplot to better center data points.
                ax.set_ylim(0, 1.1 * max(all_m_max_amplitudes))
            else:
                axes[channel_index].plot(m_x, [m_max_amplitudes], color=self.emg_object.m_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(m_max_amplitudes)}\nM-max: {m_max:.2f}mV\nStim.: above {mmax_low_stim:.2f}V', xy=(m_x + 0.2, np.mean(m_max_amplitudes)), ha='left', va='center', color='black')
                axes[channel_index].errorbar(m_x, np.mean(m_max_amplitudes), yerr=np.std(m_max_amplitudes), color='black', marker='+', markersize=10, capsize=10)

                axes[channel_index].set_title(f'{channel_names[channel_index]}')
                axes[channel_index].set_xticks([m_x])
                axes[channel_index].set_xticklabels(['M-response'])                    
                axes[channel_index].set_xlim(m_x-1, m_x+1.5) # Set x-axis limits for each subplot to better center data points.
                axes[channel_index].set_ylim(0, 1.1 * max(all_m_max_amplitudes))

            # Get raw data
            datapoints = len(m_max_amplitudes)
            raw_data_dict['channel_index'].extend([channel_index]*datapoints)
            raw_data_dict['m_max_threshold'].extend([mmax_low_stim]*datapoints)
            raw_data_dict['m_max_amplitudes'].extend(m_max_amplitudes)

        # Set labels and title
        fig.suptitle('Average M-response values at M-max for each channel')
        if num_channels == 1:
            # ax.set_xlabel('Stimulus Voltage (V)')
            ax.set_ylabel(f'Reflex Ampl. (mV, {method})')
        else:
            # fig.supxlabel('Stimulus Voltage (V)')
            fig.supylabel(f'Reflex Ampl. (mV, {method})')

        # Show the plot
        self.display_plot(canvas)

        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(['channel_index', 'm_max_threshold'], inplace=True)
        return raw_data_df

    def plot_reflexCurves (self, channel_indices : List[int] = None, method=None, plot_legend=True, relative_to_mmax=False, manual_mmax=None, canvas=None):
        """
        Plots overlayed M-response and H-reflex curves for each recorded channel.

        Args:
            channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the session.
            method (str, optional): The method used to calculate the mean and standard deviation. Options are 'rms', 'average_rectified', 'average_unrectified', or 'peak_to_trough'. Default is 'rms'.
        """
        # Set method to default if not specified.
        if method is None:
            method = self.emg_object.default_method
        
        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))
            num_channels = self.emg_object.num_channels
        else:
            num_channels = len(channel_indices)

        channel_names = self.emg_object.channel_names

        fig, ax, axes = self.create_fig_and_axes(channel_indices=channel_indices, canvas=canvas)

        raw_data_dict = {
            'channel_index': [],
            'stimulus_V': [],
            'm_wave_amplitudes': [],
            'h_wave_amplitudes': [],
        }

        # Plot the M-wave and H-response amplitudes for each channel
        for channel_index in range(self.emg_object.num_channels):
            if channel_index not in channel_indices:
                continue
            try:
                stimulus_voltages = self.emg_object.stimulus_voltages
                m_wave_amplitudes = self.emg_object.get_m_wave_amplitudes(method=method, channel_index=channel_index)
                h_wave_amplitudes = self.emg_object.get_h_wave_amplitudes(method=method, channel_index=channel_index)
            except ValueError:
                raise UnableToPlotError(f"The method {method} is not supported. Please use 'rms', 'average_rectified', 'average_unrectified', or 'peak_to_trough'.")

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            if relative_to_mmax:
                if manual_mmax is not None:
                    m_max = manual_mmax[channel_index]
                else:
                    try:
                        m_max = self.emg_object.get_m_max(method=method, channel_index=channel_index)
                    except transform.NoCalculableMmaxError:
                        raise UnableToPlotError(f'M-max could not be calculated for channel {channel_index}.')
                try:
                    m_wave_amplitudes = [amplitude / m_max for amplitude in m_wave_amplitudes]
                    h_wave_amplitudes = [amplitude / m_max for amplitude in h_wave_amplitudes]
                except ZeroDivisionError:
                    raise UnableToPlotError(f'M-max for channel {channel_index} is 0. Cannot divide by 0.')

            # Append data to raw data dictionary - will be relative to M-max if specified.
            raw_data_dict['channel_index'].extend([channel_index]*len(stimulus_voltages))
            raw_data_dict['stimulus_V'].extend(stimulus_voltages)
            raw_data_dict['m_wave_amplitudes'].extend(m_wave_amplitudes)
            raw_data_dict['h_wave_amplitudes'].extend(h_wave_amplitudes)

            if num_channels == 1:
                ax.scatter(stimulus_voltages, m_wave_amplitudes, color=self.emg_object.m_color, label='M-wave', marker='o')
                ax.scatter(stimulus_voltages, h_wave_amplitudes, color=self.emg_object.h_color, label='H-response', marker='o')
                ax.set_title(f'{channel_names[0]}')
                ax.grid(True)
                if plot_legend:
                    ax.legend()         
            else:
                axes[channel_index].scatter(stimulus_voltages, m_wave_amplitudes, color=self.emg_object.m_color, label='M-wave', marker='o')
                axes[channel_index].scatter(stimulus_voltages, h_wave_amplitudes, color=self.emg_object.h_color, label='H-response', marker='o')
                axes[channel_index].set_title(f'{channel_names[channel_index]}')
                axes[channel_index].grid(True)
                if plot_legend:
                    axes[channel_index].legend()
                    
        # Set labels and title
        fig.suptitle('M-response and H-reflex Curves')
        if num_channels == 1:
            ax.set_xlabel('Stimulus Voltage (V)')
            ax.set_ylabel(f'Reflex Ampl. (mV, {method})')
            if relative_to_mmax:
                ax.set_ylabel(f'Reflex Ampl. (M-max, {method})')
        else:
            fig.supxlabel('Stimulus Voltage (V)')
            fig.supylabel(f'Reflex Ampl. (mV, {method})')
            if relative_to_mmax:
                fig.supylabel(f'Reflex Ampl. (M-max, {method})')
        
        # Show the plot
        self.display_plot(canvas)

        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(['channel_index', 'stimulus_V'], inplace=True)
        return raw_data_df

    def plot_m_curves_smoothened (self, channel_indices : List[int] = None, method=None, relative_to_mmax=False, manual_mmax=None, canvas=None):
        """
        Plots overlayed M-response and H-reflex curves for each recorded channel.
        This plot is smoothened using a Savitzky-Golay filter, which therefore emulates the transformation used before calculating M-max in the EMG analysis.

        Args:
            channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
            method (str, optional): The method used to calculate the mean and standard deviation. Options are 'rms', 'avg_rectified', or 'peak_to_trough'. Default is 'rms'.
        """
        # Set method to default if not specified.
        if method is None:
            method = self.emg_object.default_method

        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))
            num_channels = self.emg_object.num_channels
        else:
            num_channels = len(channel_indices)

        channel_names = self.emg_object.channel_names

        fig, ax, axes = self.create_fig_and_axes(channel_indices=channel_indices, canvas=canvas)

        # Plot the M-wave and H-response amplitudes for each channel
        for channel_index in range(self.emg_object.num_channels):
            if channel_index not in channel_indices:
                continue
            try:
                m_wave_amplitudes = self.emg_object.get_m_wave_amplitudes(method=method, channel_index=channel_index)
                h_response_amplitudes = self.emg_object.get_h_wave_amplitudes(method=method, channel_index=channel_index)
                stimulus_voltages = self.emg_object.stimulus_voltages
            except ValueError:
                logging.warning(f"Failed to retrieve EMG amplitudes for channel {channel_index} in session {self.emg_object.id}.")
                continue

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            if relative_to_mmax:
                if manual_mmax is not None:
                    m_max = manual_mmax[channel_index]
                else:
                    m_max = self.emg_object.get_m_max(method=method, channel_index=channel_index)
                try:
                    m_wave_amplitudes = [amplitude / m_max for amplitude in m_wave_amplitudes]
                    h_response_amplitudes = [amplitude / m_max for amplitude in h_response_amplitudes]
                except ZeroDivisionError:
                    raise UnableToPlotError(f'M-max for channel {channel_index} is 0. Cannot divide by 0.')

            # Smoothen the data
            m_wave_amplitudes = transform.savgol_filter_y(m_wave_amplitudes)
            # h_response_amplitudes = np.gradient(m_wave_amplitudes, stimulus_voltages)

            if num_channels == 1:
                ax.scatter(stimulus_voltages, m_wave_amplitudes, color=self.emg_object.m_color, label='M-wave', marker='o')
                ax.scatter(stimulus_voltages, h_response_amplitudes, color=self.emg_object.h_color, label='H-response', marker='o')
                ax.set_title(f'{channel_names[0]}')
                ax.grid(True)
                ax.legend()    
            else:
                axes[channel_index].scatter(stimulus_voltages, m_wave_amplitudes, color=self.emg_object.m_color, label='M-wave', marker='o')
                axes[channel_index].scatter(stimulus_voltages, h_response_amplitudes, color=self.emg_object.h_color, label='H-response', marker='o')
                axes[channel_index].set_title(f'{channel_names[channel_index]}')
                axes[channel_index].grid(True)
                axes[channel_index].legend()
                    
        # Set labels and title
        fig.suptitle('M-response and H-reflex Curves')
        if num_channels == 1:
            ax.set_xlabel('Stimulus Voltage (V)')
            ax.set_ylabel(f'Reflex Ampl. (mV, {method})')
            if relative_to_mmax:
                ax.set_ylabel(f'Reflex Ampl. (M-max, {method})')
        else:
            fig.supxlabel('Stimulus Voltage (V)')
            fig.supylabel(f'Reflex Ampl. (mV, {method})')
            if relative_to_mmax:
                fig.supylabel(f'Reflex Ampl. (M-max, {method})')
        
        # Show the plot
        self.display_plot(canvas)
