import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker # used in line 282
import copy
import logging
from typing import TYPE_CHECKING, List
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import monstim_analysis.Transform_EMG as Transform_EMG

if TYPE_CHECKING:
    from monstim_analysis.Analyze_EMG import EMGData

# Desired changes:
# - make flag numbers, positions, colors, and names customizable as a preference.
# - make the axis labels and titles customizable as a preference.
# - add option to plot a figure legend for emg plots.

class EMGPlotter:
    """
    A parent class for plotting EMG data.
    
    Attributes:
        emg_object: The EMG data object to be imported.
    """
    def __init__(self, emg_object):
        self.emg_object : 'EMGData' = emg_object
        
    def set_plot_defaults(self):
        """
        Set plot font/style defaults for returned graphs.
        """
        plt.rcParams.update({'figure.titlesize': self.emg_object.title_font_size})
        plt.rcParams.update({'figure.labelsize': self.emg_object.axis_label_font_size, 'figure.labelweight': 'bold'})
        plt.rcParams.update({'axes.titlesize': self.emg_object.axis_label_font_size, 'axes.titleweight': 'bold'})
        plt.rcParams.update({'axes.labelsize': self.emg_object.axis_label_font_size, 'axes.labelweight': 'bold'})
        plt.rcParams.update({'xtick.labelsize': self.emg_object.tick_font_size, 'ytick.labelsize': self.emg_object.tick_font_size})

    def create_fig_and_axes(self, channel_indices : List[int] = None, canvas: FigureCanvas = None, figsizes = 'large'):
        if channel_indices is None:
            num_channels = self.emg_object.num_channels
        else:
            num_channels = len(channel_indices)
        
        fig, ax, axes = None, None, None # Type: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot, numpy.ndarray
        if figsizes == 'large':
            single_channel_size_tuple = (4, 2)
            multi_channel_size_tuple = (7, 2)
        elif figsizes == 'small':
            single_channel_size_tuple = (3, 2)
            multi_channel_size_tuple = (5, 2)
        
        if canvas:
            fig = canvas.figure # Type: matplotlib.figure.Figure
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
            scale = 2 # Scale factor for figure size relative to default
            if num_channels == 1:
                fig, ax = plt.subplots(figsize=tuple([item * scale for item in single_channel_size_tuple]), constrained_layout=True) # Type: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
            else:
                fig, axes = plt.subplots(nrows=1, ncols=num_channels, figsize=tuple([item * scale for item in multi_channel_size_tuple]), sharey=True, constrained_layout=True) # Type: matplotlib.figure.Figure, numpy.ndarray

        return fig, ax, axes # Type: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot, numpy.ndarray
    
    def display_plot(self, canvas : FigureCanvas):
        if canvas:
            canvas.figure.subplots_adjust(**self.emg_object.subplot_adjust_args)
            canvas.draw()
        else:
            plt.subplots_adjust(**self.emg_object.subplot_adjust_args)
            plt.show()

class EMGSessionPlotter(EMGPlotter):
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

    def __init__(self, session):
        """
        Initializes an instance of the EMGSessionPlotter class.

        Args:
            session (EMGSession): The EMGSession object to be imported containing the EMG data.

        Returns:
            None
        """
        self.emg_object : 'EMGSession' = None # The EMGSession object to be imported containing the EMG data.
        
        super().__init__(session)

        from monstim_analysis.Analyze_EMG import EMGSession
        if isinstance(session, EMGSession):
            self.emg_object = session # Type: EMGSession
        else:
            raise UnableToPlotError("Invalid data type for EMGSessionPlotter. Please provide an EMGSession object.")

        
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

    def get_time_axis(self, offset=1):
        # Calculate time values based on the scan rate
        time_values_ms = np.arange(self.emg_object.num_samples) * 1000 / self.emg_object.scan_rate  # Time values in milliseconds
        
        # Define the start and end times for the window
        window_start_time = self.emg_object.stim_start - offset # Start [offset]ms before stimulus onset
        window_end_time = window_start_time + self.emg_object.time_window_ms

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
            # deep copy of the original recordings
            original_recordings = copy.deepcopy(self.emg_object._original_recordings)
            match data_type:
                case 'filtered':
                    return self.emg_object._process_emg_data(original_recordings, apply_filter=True, rectify=False)
                case 'raw':
                    return original_recordings
                case 'rectified_raw':
                    return self.emg_object._process_emg_data(original_recordings, apply_filter=False, rectify=True)
                case 'rectified_filtered':
                    return self.emg_object._process_emg_data(original_recordings, apply_filter=True, rectify=True)
                case _:
                    raise ValueError(f"Data type '{data_type}' is not supported. Please use 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered'.")
        else: # For most uses, use the raw or filtered recordings
            if data_type == 'filtered':
                return self.emg_object.recordings_processed
            elif data_type == 'raw':
                return self.emg_object.recordings_raw
            elif data_type in ['rectified_raw', 'rectified_filtered']:
                attribute_name = f'recordings_{data_type}'
                if not hasattr(self.emg_object, attribute_name):
                    setattr(self.emg_object, attribute_name, self.emg_object._process_emg_data(self.emg_object.recordings_raw, apply_filter=(data_type == 'rectified_filtered'), rectify=True))
                return getattr(self.emg_object, attribute_name)
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
                window.plot(ax=ax, channel_index=channel_index)
        # else:
        #     for window in self.emg_object.latency_windows:
        #         if window.should_plot:
        #             window.plot(ax=ax, channel_index=channel_index)

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

    # EMGSession plotting functions
    def plot_emg(self, channel_indices : List[int] = None, all_flags : bool = True, plot_legend : bool = True, plot_colormap: bool = False, data_type : str = 'filtered', canvas: FigureCanvas = None):
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
            stimulus_v = recording['stimulus_v']
            for channel_index, channel_data in enumerate(recording['channel_data']):
                if channel_index not in channel_indices:
                    continue

                current_ax = ax if num_channels == 1 else axes[channel_index]
                
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
        emg_recordings = self.get_emg_recordings(data_type, original=True)

        if fixed_y_axis:
            max_y = [] # list to store maximum values for each channel
            min_y = [] # list to store minimum values for each channel
            for recording in emg_recordings:
                for channel_data in recording['channel_data']:
                    max_y.append(max(channel_data[window_start_sample:window_end_sample]))
                    min_y.append(min(channel_data[window_start_sample:window_end_sample]))
            y_max = max(max_y)
            y_min = min(min_y)
            y_range = y_max - y_min
            y_max += 0.01 * y_range
            y_min -= 0.01 * y_range

        if recording_index < 0 or recording_index >= len(emg_recordings):
            raise ValueError(f"Invalid recording index. Must be between 0 and {len(emg_recordings) - 1}")

        recording = emg_recordings[recording_index]

        for channel_index, channel_data in enumerate(recording['channel_data']):
            if channel_index not in channel_indices:
                continue

            current_ax = ax if num_channels == 1 else axes[channel_index]
            
            self.plot_channel_data(current_ax, time_axis, channel_data, window_start_sample, window_end_sample, recording['stimulus_v'], channel_index)
            self.plot_latency_windows(current_ax, all_flags, channel_index)

            # Collect raw data with hierarchical index structure
            num_points = len(time_axis)
            raw_data_dict['channel_index'].extend([channel_index] * num_points)
            raw_data_dict['stimulus_V'].extend([recording['stimulus_v']] * num_points)
            raw_data_dict['time_point'].extend(time_axis)

            # Add each individual EMG value for the current channel
            raw_data_dict['amplitude_mV'].extend(channel_data[window_start_sample:window_end_sample])

            if fixed_y_axis:
                current_ax.set_ylim(y_min, y_max)

        if num_channels == 1:
            sup_title = f'EMG for Channel 0 (Recording {recording_index}, Stim. = {recording["stimulus_v"]}V)'
        else:
            sup_title = f'EMG for All Channels (Recording {recording_index}, Stim. = {recording["stimulus_v"]}V)'
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
        for recording in emg_recordings:
            for channel_index, channel_data in enumerate(recording['channel_data']):
                if channel_index not in channel_indices:
                    continue
                current_ax = ax if num_channels == 1 else axes[channel_index]
                latency_window_oi_amplitude = Transform_EMG.calculate_emg_amplitude(channel_data, 
                                                                             latency_window_oi.start_times[channel_index] + self.emg_object.stim_start, 
                                                                             latency_window_oi.end_times[channel_index] + self.emg_object.stim_start, 
                                                                             self.emg_object.scan_rate,  
                                                                             method=method)
                if latency_window_oi_amplitude > emg_threshold_v:  # Check EMG amplitude within H-reflex window
                    self.plot_channel_data(current_ax, time_axis, channel_data, window_start_sample, window_end_sample, recording['stimulus_v'], channel_index)
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
            for recording in emg_recordings:
                channel_data = recording['channel_data'][channel_index]
                stimulus_v = recording['stimulus_v']
                             
                try:
                    m_wave_amplitude = Transform_EMG.calculate_emg_amplitude(channel_data, 
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
            m_max, mmax_low_stim, _ = Transform_EMG.get_avg_mmax(stimulus_voltages, m_wave_amplitudes, return_mmax_stim_range=True, **self.emg_object.m_max_args)
            
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
        if canvas:
            canvas.figure.subplots_adjust(**self.emg_object.subplot_adjust_args)
            canvas.draw()
        else:
            plt.subplots_adjust(**self.emg_object.subplot_adjust_args)
            plt.show()

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
                    except Transform_EMG.NoCalculableMmaxError:
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
            m_wave_amplitudes = self.emg_object.get_m_wave_amplitudes(method=method, channel_index=channel_index)
            h_response_amplitudes = self.emg_object.get_h_wave_amplitudes(method=method, channel_index=channel_index)
            stimulus_voltages = self.emg_object.stimulus_voltages

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
            m_wave_amplitudes = Transform_EMG.savgol_filter_y(m_wave_amplitudes)
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

class EMGDatasetPlotter(EMGPlotter):
    """
    A class for EMG data from an EMGDataset object.

    Args:
        dataset (EMGDataset): The EMG dataset object to be imported.

    Attributes:
        dataset (EMGDataset): The imported EMG dataset object.

    Methods:
        plot_reflex_curves: Plots the average M-response and H-reflex curves for each channel.
        plot_max_h_reflex: Plots the M-wave and H-response amplitudes at the stimulation voltage where the average H-reflex is maximal.
        help: Displays the help text for the class.
    """
    def __init__(self, dataset):
        self.emg_object : 'EMGDataset' = None # The EMGDataset object to be imported.
        
        super().__init__(dataset)

        from monstim_analysis.Analyze_EMG import EMGDataset
        if isinstance(dataset, EMGDataset):
            self.emg_object = dataset # Type: EMGDataset
        else:
            raise UnableToPlotError("Invalid data type for EMGDatasetPlotter. Please provide an EMGDataset object.")

        self.set_plot_defaults()
    
    def help(self):
        help_text = """
        EMGDatasetPlotter class for plotting EMG data from an EMGDataset object.
        ========================================================================
        Methods:
        1. plot_reflex_curves: Plots the average M-response and H-reflex curves for each channel.
            -Example:
                plot_reflex_curves(channel_names=['Channel 1', 'Channel 2'], method='rms', relative_to_mmax=False, manual_mmax=None)
        
        2. plot_max_h_reflex: Plots the M-wave and H-response amplitudes at the stimulation voltage where the average H-reflex is maximal.
            -Example:
                plot_max_h_reflex(channel_names=['Channel 1', 'Channel 2'], method='rms', relative_to_mmax=False, manual_mmax=None)
        
        3. help: Displays this help text.
        ========================================================================

        """
        print(help_text)

    # EMGDataset plotting functions
    def plot_reflexCurves(self, channel_indices : List[int] = None, method=None, plot_legend=True, relative_to_mmax=False, manual_mmax=None, canvas=None):
        """
        Plots the M-response and H-reflex curves for each channel.

        Args:
            channel_names (list): A list of custom channel names. If specified, the channel names will be used in the plot titles.
            method (str): The method used to calculate the mean and standard deviation. Options are 'rms', 'average_rectified', 'average_unrectified', or 'peak_to_trough'. Default is 'rms'.

        Returns:
            None
        """
        # Set method to default if not specified.
        if method is None:
            method = self.emg_object.default_method
        channel_names = self.emg_object.channel_names
        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))
            num_channels = self.emg_object.num_channels
        else:
            num_channels = len(channel_indices)

        fig, ax, axes = self.create_fig_and_axes(channel_indices=channel_indices, canvas=canvas)

        # Get unique binned stimulus voltages
        stimulus_voltages = self.emg_object.stimulus_voltages

        raw_data_dict = {
            'channel_index': [],
            'stimulus_v': [],
            'avg_m_wave': [],
            'stderr_m_wave': [],
            'avg_h_wave': [],
            'stderr_h_wave': []
        }

        # Plot the M-wave and H-response amplitudes for each channel
        for channel_index in range(self.emg_object.num_channels):
            if channel_index not in channel_indices:
                continue
            m_wave_means, m_wave_error = self.emg_object.get_avg_m_wave_amplitudes(method, channel_index)
            h_response_means, h_response_error = self.emg_object.get_avg_h_wave_amplitudes(method, channel_index)

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            if relative_to_mmax:
                if manual_mmax is not None:
                    channel_m_max = manual_mmax[channel_index]
                else:
                    channel_m_max = self.emg_object.get_avg_m_max(method, channel_index)
                    if channel_m_max is None:
                        raise UnableToPlotError(f'M-max could not be calculated for channel {channel_index}.')
                try:
                    m_wave_means = [(amplitude / channel_m_max) for amplitude in m_wave_means]
                    m_wave_error = [(amplitude / channel_m_max) for amplitude in m_wave_error]
                    h_response_means = [(amplitude / channel_m_max) for amplitude in h_response_means]
                    h_response_error = [(amplitude / channel_m_max) for amplitude in h_response_error]
                except ZeroDivisionError:
                    raise UnableToPlotError(f'M-max for channel {channel_index} is zero. Cannot divide by zero.')

            # Append data to raw data dictionary - will be relative to M-max if specified.
            raw_data_dict['channel_index'].extend([channel_index]*len(stimulus_voltages))
            raw_data_dict['stimulus_v'].extend(stimulus_voltages)
            raw_data_dict['avg_m_wave'].extend(m_wave_means)
            raw_data_dict['stderr_m_wave'].extend(m_wave_error)
            raw_data_dict['avg_h_wave'].extend(h_response_means)
            raw_data_dict['stderr_h_wave'].extend(h_response_error)
            
            # Plot the M-wave and H-response amplitudes for each channel
            if num_channels == 1:
                ax.plot(stimulus_voltages, m_wave_means, color=self.emg_object.m_color, label='M-wave')
                ax.fill_between(stimulus_voltages, np.array(m_wave_means) - np.array(m_wave_error), np.array(m_wave_means) + np.array(m_wave_error), color='r', alpha=0.2)
                ax.plot(stimulus_voltages, h_response_means, color=self.emg_object.h_color, label='H-response')
                ax.fill_between(stimulus_voltages, np.array(h_response_means) - np.array(h_response_error), np.array(h_response_means) + np.array(h_response_error), color='b', alpha=0.2)
                ax.set_title(f'{channel_names[0]}')
                ax.grid(True)
                if plot_legend:
                    ax.legend()
            else:
                axes[channel_index].plot(stimulus_voltages, m_wave_means, color=self.emg_object.m_color, label='M-wave')
                axes[channel_index].fill_between(stimulus_voltages, np.array(m_wave_means) - np.array(m_wave_error), np.array(m_wave_means) + np.array(m_wave_error), color='r', alpha=0.2)
                axes[channel_index].plot(stimulus_voltages, h_response_means, color=self.emg_object.h_color, label='H-response')
                axes[channel_index].fill_between(stimulus_voltages, np.array(h_response_means) - np.array(h_response_error), np.array(h_response_means) + np.array(h_response_error), color='b', alpha=0.2)
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
        raw_data_df.set_index(['channel_index', 'stimulus_v'], inplace=True)
        return raw_data_df

    def plot_maxH(self, channel_indices : List[int] = None, method=None, relative_to_mmax=False, manual_mmax=None, max_stim_value=None, bin_margin=0, canvas=None):
        """
        Plots the M-wave and H-response amplitudes at the stimulation voltage where the average H-reflex is maximal.

        Args:
            channel_names (list): List of custom channel names. Default is an empty list.
            method (str): Method for calculating the amplitude. Options are 'rms', 'avg_rectified', or 'peak_to_trough'. Default is 'rms'.
            relative_to_mmax (bool): Flag indicating whether to make the M-wave amplitudes relative to the maximum M-wave amplitude. Default is False.
            manual_mmax (float): Manual value for the maximum M-wave amplitude. Default is None.

        Returns:
            None
        """
        # Set method to default if not specified.
        if method is None:
            method = self.emg_object.default_method
        channel_names = self.emg_object.channel_names
        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))
            num_channels = self.emg_object.num_channels
        else:
            num_channels = len(channel_indices)

        fig, ax, axes = self.create_fig_and_axes(channel_indices=channel_indices, canvas=canvas, figsizes='small')

        raw_data_dict = {
            'channel_index': [],
            'stimulus_v': [],
            'm_wave_amplitudes': [],
            'h_wave_amplitudes': [],
        }

        for channel_index in range(self.emg_object.num_channels):
            if channel_index not in channel_indices:
                continue
            # Get average H-wave amplitudes
            h_response_means, _ = self.emg_object.get_avg_h_wave_amplitudes(method, channel_index)
            stimulus_voltages = self.emg_object.stimulus_voltages

            # Filter out stimulus voltages greater than max_stim_value if specified
            if max_stim_value is not None:
                filtered_indices = [i for i, v in enumerate(stimulus_voltages) if v <= max_stim_value]
                stimulus_voltages = [stimulus_voltages[i] for i in filtered_indices]
                h_response_means = [h_response_means[i] for i in filtered_indices]

            # Find the voltage with the maximum average H-reflex amplitude
            max_h_reflex_amplitude = max(h_response_means)
            max_h_reflex_voltage = stimulus_voltages[h_response_means.index(max_h_reflex_amplitude)]
            
            # Define the range of voltages around the max H-reflex voltage
            voltage_indices = range(max(0, h_response_means.index(max_h_reflex_amplitude) - bin_margin),
                                min(len(stimulus_voltages), h_response_means.index(max_h_reflex_amplitude) + bin_margin + 1))
            marginal_voltages = [stimulus_voltages[i] for i in voltage_indices]

            # Collect M-wave and H-response amplitudes for the marginal bins
            stimulus_voltages = []
            m_wave_amplitudes = []
            h_response_amplitudes = []
            for voltage in marginal_voltages:
                m_waves = self.emg_object.get_m_wave_amplitudes_at_voltage(method, channel_index, voltage)
                h_responses = self.emg_object.get_h_wave_amplitudes_at_voltage(method, channel_index, voltage)
                m_wave_amplitudes.extend(m_waves)
                h_response_amplitudes.extend(h_responses)
                stimulus_voltages.extend([voltage] * len(m_waves))

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            if relative_to_mmax:
                if manual_mmax is not None:
                    m_max = manual_mmax[channel_index]
                else:
                    m_max = self.emg_object.get_avg_m_max(method=method, channel_index=channel_index)
                try:
                    m_wave_amplitudes = [amplitude / m_max for amplitude in m_wave_amplitudes]
                    h_response_amplitudes = [amplitude / m_max for amplitude in h_response_amplitudes]
                except TypeError:
                    raise UnableToPlotError(f'M-max could not be calculated for channel {channel_index}.')
                except ZeroDivisionError:
                    raise UnableToPlotError(f'M-max is zero for channel {channel_index}. Cannot divide by zero.')

            # # Crop data if max_stim_value is specified
            # if max_stim_value is not None:
            #     mask = np.array(stimulus_voltages) <= max_stim_value
            #     stimulus_voltages = np.array(stimulus_voltages)[mask]
            #     m_wave_amplitudes = np.array(m_wave_amplitudes)[mask]
            #     h_response_amplitudes = np.array(h_response_amplitudes)[mask]

            # Append data to raw data dictionary
            raw_data_dict['channel_index'].extend([channel_index] * len(m_wave_amplitudes))
            raw_data_dict['stimulus_v'].extend(stimulus_voltages)
            raw_data_dict['m_wave_amplitudes'].extend(m_wave_amplitudes)
            raw_data_dict['h_wave_amplitudes'].extend(h_response_amplitudes)

            # Plot the M-wave and H-response amplitudes for the maximum H-reflex voltage.
            m_x = 1
            h_x = 2.5
            if num_channels == 1:
                ax.plot(m_x, [m_wave_amplitudes], color=self.emg_object.m_color, marker='o', markersize=5)
                ax.annotate(f'n={len(m_wave_amplitudes)}\navg. = {np.average(m_wave_amplitudes):.2f}\nstd. = {np.std(m_wave_amplitudes):.2f}', xy=(m_x + 0.4, np.mean(m_wave_amplitudes)), ha='left', color=self.emg_object.m_color)
                ax.errorbar(m_x, np.mean(m_wave_amplitudes), yerr=np.std(m_wave_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                ax.plot(h_x, [h_response_amplitudes], color=self.emg_object.h_color, marker='o', markersize=5)
                ax.annotate(f'n={len(h_response_amplitudes)}\navg. = {np.average(h_response_amplitudes):.2f}\nstd. = {np.std(h_response_amplitudes):.2f}', xy=(h_x + 0.4, np.mean(h_response_amplitudes)), ha='left', color=self.emg_object.h_color)
                ax.errorbar(h_x, np.mean(h_response_amplitudes), yerr=np.std(h_response_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                ax.set_xticks([m_x, h_x])
                ax.set_xticklabels(['M-response', 'H-reflex'])
                ax.set_title(f'{channel_names[0]} ({round(max_h_reflex_voltage, 2)} ± {round((self.emg_object.bin_size/2)+(self.emg_object.bin_size * bin_margin),2)}V)')
                ax.set_xlim(m_x-1, h_x+1) # Set x-axis limits for each subplot to better center data points.
            else:
                axes[channel_index].plot(m_x, [m_wave_amplitudes], color=self.emg_object.m_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(m_wave_amplitudes)}\navg. = {np.average(m_wave_amplitudes):.2f}\nstd. = {np.std(m_wave_amplitudes):.2f}', xy=(m_x + 0.4, np.mean(m_wave_amplitudes)), ha='left', color=self.emg_object.m_color)
                axes[channel_index].errorbar(m_x, np.mean(m_wave_amplitudes), yerr=np.std(m_wave_amplitudes), color='black', marker='+', markersize=10, capsize=10)

                axes[channel_index].plot(h_x, [h_response_amplitudes], color=self.emg_object.h_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(h_response_amplitudes)}\navg. = {np.average(h_response_amplitudes):.2f}\nstd. = {np.std(h_response_amplitudes):.2f}', xy=(h_x + 0.4, np.mean(h_response_amplitudes)), ha='left', color=self.emg_object.h_color)
                axes[channel_index].errorbar(h_x, np.mean(h_response_amplitudes), yerr=np.std(h_response_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                axes[channel_index].set_title(f'{channel_names[channel_index]} ({round(max_h_reflex_voltage, 2)} ± {round((self.emg_object.bin_size/2)+(self.emg_object.bin_size * bin_margin),2)}V)')
                axes[channel_index].set_xticks([m_x, h_x])
                axes[channel_index].set_xticklabels(['M-response', 'H-reflex'])
                axes[channel_index].set_xlim(m_x-1, h_x+1) # Set x-axis limits for each subplot to better center data points.

        # Set labels and title
        fig.suptitle('EMG Responses at Max H-reflex Stimulation')
        if num_channels == 1:
            ax.set_xlabel('Response Type')
            ax.set_ylabel(f'EMG Amp. (mV, {method})')
            if relative_to_mmax:
                ax.set_ylabel(f'EMG Amp. (M-max, {method})')
        else:
            fig.supxlabel('Response Type')
            fig.supylabel(f'EMG Amp. (mV, {method})')
            if relative_to_mmax:
                fig.supylabel(f'EMG Amp. (M-max, {method})')

        # Show the plot
        self.display_plot(canvas)

        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(['channel_index', 'stimulus_v'], inplace=True)
        return raw_data_df

    def plot_mmax(self, channel_indices : List[int] = None, method : str = None, canvas : FigureCanvas = None):
        """
        Plots the average M-max of each session for each channel.

        Args:
            method (str, optional): The method used to calculate the mean and standard deviation. Options are 'rms', 'average_rectified', 'average_unrectified', or 'peak_to_trough'. Default is 'rms'.

        Returns:
            None
        """
        # Set method to default if not specified.
        if method is None:
            method = self.emg_object.default_method

        channel_names = self.emg_object.channel_names

        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))
            num_channels = self.emg_object.num_channels
        else:
            num_channels = len(channel_indices)

        fig, ax, axes = self.create_fig_and_axes(channel_indices=channel_indices, canvas=canvas)

        all_m_max_amplitudes = []
        raw_data_dict = {
            'channel_index': [],
            'session_id': [],
            'm_max_threshold': [],
            'm_max_amplitude': [],
        }

        for channel_index in range(self.emg_object.num_channels):
            if channel_index not in channel_indices:
                continue
            m_max_amplitudes = []
            m_max_thresholds = []
            session_ids = []
            for session in self.emg_object.emg_sessions:
                try:
                    m_max, mmax_low_stim, _ = session.get_m_max(method=method, channel_index=channel_index, return_mmax_stim_range=True)
                    m_max_amplitudes.append(m_max)
                    m_max_thresholds.append(mmax_low_stim)
                    session_ids.append(session.session_id)
                except IndexError:
                    m_max_amplitudes.append(np.nan)
                    m_max_thresholds.append(np.nan)
                    session_ids.append(session.session_id)
                
            # Append M-wave amplitudes to superlist for y-axis adjustment.
            all_m_max_amplitudes.extend(m_max_amplitudes)            

            # Plot the M-wave amplitudes for each session
            m_x = 1
            if num_channels == 1:
                ax.plot(m_x, [m_max_amplitudes], color=self.emg_object.m_color, marker='o', markersize=5)
                ax.annotate(f'n={len(m_max_amplitudes)}\nAvg. M-max: {np.mean(m_max_amplitudes):.2f}mV\nStdev. M-Max: {np.std(m_max_amplitudes):.2f}mV\nAvg. Stim.: above {np.mean(m_max_thresholds):.2f} mV', xy=(m_x + 0.2, np.mean(m_max_amplitudes)), ha='left', va='center', color='black')
                ax.errorbar(m_x, np.mean(m_max_amplitudes), yerr=np.std(m_max_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                ax.set_xticklabels(['M-response'])
                ax.set_title(f'{channel_names[0]}')                    
                ax.set_xlim(m_x-1, m_x+1.5) # Set x-axis limits for each subplot to better center data points.
                ax.set_ylim(0, 1.1 * max(all_m_max_amplitudes))
            else:
                axes[channel_index].plot(m_x, [m_max_amplitudes], color=self.emg_object.m_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(m_max_amplitudes)}\nAvg. M-max: {np.mean(m_max_amplitudes):.2f}mV\nStdev. M-Max: {np.std(m_max_amplitudes):.2f}mV\nAvg. Stim.: above {np.mean(m_max_thresholds):.2f} mV', xy=(m_x + 0.2, np.mean(m_max_amplitudes)), ha='left', va='center', color='black')
                axes[channel_index].errorbar(m_x, np.mean(m_max_amplitudes), yerr=np.std(m_max_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                axes[channel_index].set_xticks([m_x])
                axes[channel_index].set_xticklabels(['M-response'])
                axes[channel_index].set_title(f'{channel_names[channel_index]}')
                axes[channel_index].set_xlim(m_x-1, m_x+1.5) # Set x-axis limits for each subplot to better center data points.
                axes[channel_index].set_ylim(0, 1.1 * max(all_m_max_amplitudes))

            # Append data to raw data dictionary
            datapoints = len(m_max_amplitudes)
            raw_data_dict['channel_index'].extend([channel_index]*datapoints)
            raw_data_dict['session_id'].extend(session_ids)
            raw_data_dict['m_max_threshold'].extend(m_max_thresholds)
            raw_data_dict['m_max_amplitude'].extend(m_max_amplitudes)

        # Set labels and title
        fig.suptitle('Average M-max')
        if num_channels == 1:
            ax.set_xlabel('Response Type')
            ax.set_ylabel(f'M-max (mV, {method})')
        else:
            fig.supxlabel('Response Type')
            fig.supylabel(f'M-max (mV, {method})')
        
        # Show the plot
        self.display_plot(canvas)

        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(['channel_index'], inplace=True)
        return raw_data_df        
   
class EMGExperimentPlotter(EMGPlotter):
    def __init__(self, experiment):
        self.emg_object : 'EMGExperiment' = None # The EMGExperiment object to be imported.

        super().__init__(experiment)

        from monstim_analysis.Analyze_EMG import EMGExperiment
        if isinstance(experiment, EMGExperiment):
            self.emg_object = experiment # Type: EMGExperiment
        else:
            raise UnableToPlotError("Invalid data type for EMGExperimentPlotter. Please provide an EMGExperiment object.")
        
        self.set_plot_defaults()

    # EMGExperiment plotting functions
    def plot_reflexCurves(self, channel_indices : List[int] = None, method=None, plot_legend=True, relative_to_mmax=False, manual_mmax=None, canvas=None):
        """
        Plots the M-response and H-reflex curves for each channel.

        Args:
            channel_names (list): A list of custom channel names. If specified, the channel names will be used in the plot titles.
            method (str): The method used to calculate the mean and standard deviation. Options are 'rms', 'average_rectified', 'average_unrectified', or 'peak_to_trough'. Default is 'rms'.

        Returns:
            None
        """
        # Set method to default if not specified.
        if method is None:
            method = self.emg_object.default_method
        channel_names = self.emg_object.channel_names
        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))
            num_channels = self.emg_object.num_channels
        else:
            num_channels = len(channel_indices)

        fig, ax, axes = self.create_fig_and_axes(channel_indices=channel_indices, canvas=canvas)

        # Get unique binned stimulus voltages
        stimulus_voltages = self.emg_object.stimulus_voltages

        raw_data_dict = {
            'channel_index': [],
            'stimulus_v': [],
            'avg_m_wave': [],
            'stderr_m_wave': [],
            'avg_h_wave': [],
            'stderr_h_wave': []
        }

        # Plot the M-wave and H-response amplitudes for each channel
        for channel_index in range(self.emg_object.num_channels):
            if channel_index not in channel_indices:
                continue
            m_wave_means, m_wave_error = self.emg_object.get_avg_m_wave_amplitudes(method, channel_index)
            h_response_means, h_response_error = self.emg_object.get_avg_h_wave_amplitudes(method, channel_index)

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            if relative_to_mmax:
                if manual_mmax is not None:
                    channel_m_max = manual_mmax[channel_index]
                else:
                    channel_m_max = self.emg_object.get_avg_m_max(method, channel_index)
                try:
                    m_wave_means = [(amplitude / channel_m_max) for amplitude in m_wave_means]
                    m_wave_error = [(amplitude / channel_m_max) for amplitude in m_wave_error]
                    h_response_means = [(amplitude / channel_m_max) for amplitude in h_response_means]
                    h_response_error = [(amplitude / channel_m_max) for amplitude in h_response_error]
                except ZeroDivisionError:
                    raise UnableToPlotError(f'M-max for channel {channel_index} is zero. Cannot divide by zero.')

            # Append data to raw data dictionary - will be relative to M-max if specified.
            raw_data_dict['channel_index'].extend([channel_index]*len(stimulus_voltages))
            raw_data_dict['stimulus_v'].extend(stimulus_voltages)
            raw_data_dict['avg_m_wave'].extend(m_wave_means)
            raw_data_dict['stderr_m_wave'].extend(m_wave_error)
            raw_data_dict['avg_h_wave'].extend(h_response_means)
            raw_data_dict['stderr_h_wave'].extend(h_response_error)
            
            # Plot the M-wave and H-response amplitudes for each channel
            if num_channels == 1:
                ax.plot(stimulus_voltages, m_wave_means, color=self.emg_object.m_color, label='M-wave')
                ax.fill_between(stimulus_voltages, np.array(m_wave_means) - np.array(m_wave_error), np.array(m_wave_means) + np.array(m_wave_error), color='r', alpha=0.2)
                ax.plot(stimulus_voltages, h_response_means, color=self.emg_object.h_color, label='H-response')
                ax.fill_between(stimulus_voltages, np.array(h_response_means) - np.array(h_response_error), np.array(h_response_means) + np.array(h_response_error), color='b', alpha=0.2)
                ax.set_title(f'{channel_names[0]}')
                ax.grid(True)
                if plot_legend:
                    ax.legend()
            else:
                axes[channel_index].plot(stimulus_voltages, m_wave_means, color=self.emg_object.m_color, label='M-wave')
                axes[channel_index].fill_between(stimulus_voltages, np.array(m_wave_means) - np.array(m_wave_error), np.array(m_wave_means) + np.array(m_wave_error), color='r', alpha=0.2)
                axes[channel_index].plot(stimulus_voltages, h_response_means, color=self.emg_object.h_color, label='H-response')
                axes[channel_index].fill_between(stimulus_voltages, np.array(h_response_means) - np.array(h_response_error), np.array(h_response_means) + np.array(h_response_error), color='b', alpha=0.2)
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
        raw_data_df.set_index(['channel_index', 'stimulus_v'], inplace=True)
        return raw_data_df
    
    def plot_mmax(self, channel_indices : List[int] = None, method : str = None, canvas : FigureCanvas = None):
        """
        Plots the average M-max of each dataset for each channel (animal averages).

        Args:
            method (str, optional): The method used to calculate the mean and standard deviation. Options are 'rms', 'average_rectified', 'average_unrectified', or 'peak_to_trough'. Default is 'rms'.

        Returns:
            None
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

        all_m_max_amplitudes = []
        raw_data_dict = {
            'channel_index': [],
            'animal_id': [],
            'avg_m_max_threshold': [],
            'avg_m_max_amplitude': [],
        }

        for channel_index in range(self.emg_object.num_channels):
            if channel_index not in channel_indices:
                continue
            avg_m_max_amplitudes_raw = []
            avg_m_max_thresholds_raw = []
            animal_ids = []
            for dataset in self.emg_object.emg_datasets:
                try:
                    avg_m_max, avg_mmax_low_stim = dataset.get_avg_m_max(method=method, channel_index=channel_index, return_avg_mmax_thresholds=True)
                    avg_m_max_amplitudes_raw.append(avg_m_max)
                    avg_m_max_thresholds_raw.append(avg_mmax_low_stim)
                    animal_ids.append(dataset.animal_id)
                except IndexError:
                    avg_m_max_amplitudes_raw.append(None)
                    avg_m_max_thresholds_raw.append(None)
                    animal_ids.append(dataset.animal_id)

            # Drop None values from the list
            avg_m_max_amplitudes = [x for x in avg_m_max_amplitudes_raw if x is not None]
            avg_m_max_thresholds = [x for x in avg_m_max_thresholds_raw if x is not None]
                
            # Append M-wave amplitudes to superlist for y-axis adjustment.
            all_m_max_amplitudes.extend(avg_m_max_amplitudes)        

            # Plot the M-wave amplitudes for each session
            m_x = 1
            if num_channels == 1:
                ax.plot(m_x, [avg_m_max_amplitudes], color=self.emg_object.m_color, marker='o', markersize=5)
                ax.annotate(f'n={len(avg_m_max_amplitudes)}\nAvg. M-max: {np.mean(avg_m_max_amplitudes):.2f}mV\nAvg. Stim.: above {np.mean(avg_m_max_thresholds):.2f} mV', xy=(m_x + 0.2, np.mean(avg_m_max_amplitudes)), ha='left', va='center', color='black')
                ax.errorbar(m_x, np.mean(avg_m_max_amplitudes), yerr=np.std(avg_m_max_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                ax.set_xticklabels(['M-response'])
                ax.set_title(f'{channel_names[0]}')                    
                ax.set_xlim(m_x-1, m_x+1.5) # Set x-axis limits for each subplot to better center data points.
                ax.set_ylim(0, 1.1 * max(all_m_max_amplitudes))
            else:
                axes[channel_index].plot(m_x, [avg_m_max_amplitudes], color=self.emg_object.m_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(avg_m_max_amplitudes)}\nAvg. M-max: {np.mean(avg_m_max_amplitudes):.2f}mV\nAvg. Stim.: above {np.mean(avg_m_max_thresholds):.2f} mV', xy=(m_x + 0.2, np.mean(avg_m_max_amplitudes)), ha='left', va='center', color='black')
                axes[channel_index].errorbar(m_x, np.mean(avg_m_max_amplitudes), yerr=np.std(avg_m_max_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                axes[channel_index].set_xticks([m_x])
                axes[channel_index].set_xticklabels(['M-response'])
                axes[channel_index].set_title(f'{channel_names[channel_index]}')
                axes[channel_index].set_xlim(m_x-1, m_x+1.5) # Set x-axis limits for each subplot to better center data points.
                axes[channel_index].set_ylim(0, 1.1 * max(all_m_max_amplitudes))

            # Append data to raw data dictionary
            datapoints = len(avg_m_max_amplitudes_raw)
            raw_data_dict['channel_index'].extend([channel_index]*datapoints)
            raw_data_dict['animal_id'].extend(animal_ids)
            raw_data_dict['avg_m_max_threshold'].extend(avg_m_max_thresholds_raw)
            raw_data_dict['avg_m_max_amplitude'].extend(avg_m_max_amplitudes_raw)

        # Set labels and title
        fig.suptitle('Average M-max')
        if num_channels == 1:
            ax.set_xlabel('Response Type')
            ax.set_ylabel(f'M-max (mV, {method})')
        else:
            fig.supxlabel('Response Type')
            fig.supylabel(f'M-max (mV, {method})')
        
        # Show the plot
        self.display_plot(canvas)

        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(['channel_index'], inplace=True)
        return raw_data_df

    def plot_maxH(self, channel_indices : List[int] = None, method=None, relative_to_mmax=False, manual_mmax=None, bin_margin=0, canvas=None):
        """
        Plots the M-wave and H-response amplitudes at the stimulation voltage where the average H-reflex is maximal.

        Args:
            channel_names (list): List of custom channel names. Default is an empty list.
            method (str): Method for calculating the amplitude. Options are 'rms', 'avg_rectified', or 'peak_to_trough'. Default is 'rms'.
            relative_to_mmax (bool): Flag indicating whether to make the M-wave amplitudes relative to the maximum M-wave amplitude. Default is False.
            manual_mmax (float): Manual value for the maximum M-wave amplitude. Default is None.

        Returns:
            None
        """
        # Set method to default if not specified.
        if method is None:
            method = self.emg_object.default_method
        channel_names = self.emg_object.channel_names
        if channel_indices is None:
            channel_indices = list(range(self.emg_object.num_channels))
            num_channels = self.emg_object.num_channels
        else:
            num_channels = len(channel_indices)

        fig, ax, axes = self.create_fig_and_axes(channel_indices=channel_indices, canvas=canvas, figsizes='small')

        raw_data_dict = {
            'channel_index': [],
            'stimulus_v': [],
            'avg_m_wave_amplitudes': [],
            'avg_h_wave_amplitudes': [],
        }

        for channel_index in range(self.emg_object.num_channels):
            if channel_index not in channel_indices:
                continue
            # Get average H-wave amplitudes
            h_response_means, _ = self.emg_object.get_avg_h_wave_amplitudes(method, channel_index)
            stimulus_voltages = self.emg_object.stimulus_voltages

            # Find the voltage with the maximum average H-reflex amplitude
            max_h_reflex_amplitude = max(h_response_means)
            max_h_reflex_voltage = stimulus_voltages[h_response_means.index(max_h_reflex_amplitude)]
            
            # Define the range of voltages around the max H-reflex voltage
            voltage_indices = range(max(0, h_response_means.index(max_h_reflex_amplitude) - bin_margin),
                                min(len(stimulus_voltages), h_response_means.index(max_h_reflex_amplitude) + bin_margin + 1))
            marginal_voltages = [stimulus_voltages[i] for i in voltage_indices]

            # Collect M-wave and H-response amplitudes for the marginal bins
            stimulus_voltages = []
            m_wave_amplitudes = []
            h_response_amplitudes = []
            for voltage in marginal_voltages:
                m_wave_avgs = self.emg_object.get_m_wave_amplitude_avgs_at_voltage(method, channel_index, voltage)
                h_response_avgs = self.emg_object.get_h_wave_amplitude_avgs_at_voltage(method, channel_index, voltage)
                m_wave_amplitudes.extend(m_wave_avgs)
                h_response_amplitudes.extend(h_response_avgs)
                stimulus_voltages.extend([voltage] * len(m_wave_avgs))

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            if relative_to_mmax:
                if manual_mmax is not None:
                    m_max = manual_mmax[channel_index]
                else:
                    m_max = self.emg_object.get_avg_m_max(method=method, channel_index=channel_index)
                try: 
                    m_wave_amplitudes = [amplitude / m_max for amplitude in m_wave_amplitudes]
                    h_response_amplitudes = [amplitude / m_max for amplitude in h_response_amplitudes]
                except TypeError:
                    raise UnableToPlotError(f'M-max could not be calculated for channel {channel_index}.')
                except ZeroDivisionError:
                    raise UnableToPlotError(f'M-max is zero for channel {channel_index}. Cannot divide by zero.')

            # Append data to raw data dictionary
            raw_data_dict['channel_index'].extend([channel_index] * len(m_wave_amplitudes))
            raw_data_dict['stimulus_v'].extend(stimulus_voltages)
            raw_data_dict['avg_m_wave_amplitudes'].extend(m_wave_amplitudes)
            raw_data_dict['avg_h_wave_amplitudes'].extend(h_response_amplitudes)

            # Plot the M-wave and H-response amplitudes for the maximum H-reflex voltage.
            m_x = 1
            h_x = 2.5
            if num_channels == 1:
                ax.plot(m_x, [m_wave_amplitudes], color=self.emg_object.m_color, marker='o', markersize=5)
                ax.annotate(f'n={len(m_wave_amplitudes)}', xy=(m_x + 0.4, np.mean(m_wave_amplitudes)), ha='center', color=self.emg_object.m_color)
                ax.errorbar(m_x, np.mean(m_wave_amplitudes), yerr=np.std(m_wave_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                ax.plot(h_x, [h_response_amplitudes], color=self.emg_object.h_color, marker='o', markersize=5)
                ax.annotate(f'n={len(h_response_amplitudes)}', xy=(h_x + 0.4, np.mean(h_response_amplitudes)), ha='center', color=self.emg_object.h_color)
                ax.errorbar(h_x, np.mean(h_response_amplitudes), yerr=np.std(h_response_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                ax.set_xticks([m_x, h_x])
                ax.set_xticklabels(['M-response', 'H-reflex'])
                ax.set_title(f'{channel_names[0]} ({round(max_h_reflex_voltage, 2)} ± {round((self.emg_object.bin_size/2)+(self.emg_object.bin_size * bin_margin),2)}V)')
                ax.set_xlim(m_x-1, h_x+1) # Set x-axis limits for each subplot to better center data points.
            else:
                axes[channel_index].plot(m_x, [m_wave_amplitudes], color=self.emg_object.m_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(m_wave_amplitudes)}', xy=(m_x + 0.4, np.mean(m_wave_amplitudes)), ha='center', color=self.emg_object.m_color)
                axes[channel_index].errorbar(m_x, np.mean(m_wave_amplitudes), yerr=np.std(m_wave_amplitudes), color='black', marker='+', markersize=10, capsize=10)

                axes[channel_index].plot(h_x, [h_response_amplitudes], color=self.emg_object.h_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(h_response_amplitudes)}', xy=(h_x + 0.4, np.mean(h_response_amplitudes)), ha='center', color=self.emg_object.h_color)
                axes[channel_index].errorbar(h_x, np.mean(h_response_amplitudes), yerr=np.std(h_response_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                axes[channel_index].set_title(f'{channel_names[channel_index]} ({round(max_h_reflex_voltage, 2)} ± {round((self.emg_object.bin_size/2)+(self.emg_object.bin_size * bin_margin),2)}V)')
                axes[channel_index].set_xticks([m_x, h_x])
                axes[channel_index].set_xticklabels(['M-response', 'H-reflex'])
                axes[channel_index].set_xlim(m_x-1, h_x+1) # Set x-axis limits for each subplot to better center data points.

        # Set labels and title
        fig.suptitle('EMG Responses at Max H-reflex Stimulation')
        if num_channels == 1:
            ax.set_xlabel('Response Type')
            ax.set_ylabel(f'Avg. EMG Amp. (mV, {method})')
            if relative_to_mmax:
                ax.set_ylabel(f'Avg. EMG Amp. (M-max, {method})')
        else:
            fig.supxlabel('Response Type')
            fig.supylabel(f'Avg. EMG Amp. (mV, {method})')
            if relative_to_mmax:
                fig.supylabel(f'Avg. EMG Amp. (M-max, {method})')

        # Show the plot
        self.display_plot(canvas)

        # Create DataFrame with multi-level index
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.set_index(['channel_index', 'stimulus_v'], inplace=True)
        return raw_data_df

class UnableToPlotError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)