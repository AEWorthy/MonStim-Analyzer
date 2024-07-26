import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
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
        plt.switch_backend('QtAgg')
        
    def set_plot_defaults(self):
        """
        Set plot font/style defaults for returned graphs.
        """
        plt.rcParams.update({'figure.titlesize': self.emg_object.title_font_size})
        plt.rcParams.update({'figure.labelsize': self.emg_object.axis_label_font_size, 'figure.labelweight': 'bold'})
        plt.rcParams.update({'axes.titlesize': self.emg_object.axis_label_font_size, 'axes.titleweight': 'bold'})
        plt.rcParams.update({'axes.labelsize': self.emg_object.axis_label_font_size, 'axes.labelweight': 'bold'})
        plt.rcParams.update({'xtick.labelsize': self.emg_object.tick_font_size, 'ytick.labelsize': self.emg_object.tick_font_size})

    def create_fig_and_axes(self, canvas: FigureCanvas = None, figsizes = 'large'):
        fig, ax, axes = None, None, None
        if figsizes == 'large':
            single_channel_size_tuple = (4, 2)
            multi_channel_size_tuple = (7, 2)
        elif figsizes == 'small':
            single_channel_size_tuple = (3, 2)
            multi_channel_size_tuple = (5, 2)
        
        if canvas:
            fig = canvas.figure # Type: matplotlib.figure.Figure
            fig.clear()
            fig.set_tight_layout(True)
            
            # Use predefined size tuples
            if self.emg_object.num_channels == 1:
                fig.set_size_inches(*single_channel_size_tuple)
                ax = fig.add_subplot(111)
            else:
                fig.set_size_inches(*multi_channel_size_tuple)
                axes = fig.subplots(nrows=1, ncols=self.emg_object.num_channels, sharey=True)
            
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
            if self.emg_object.num_channels == 1:
                fig, ax = plt.subplots(figsize=tuple([item * scale for item in single_channel_size_tuple])) # Type: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
            else:
                fig, axes = plt.subplots(nrows=1, ncols=self.emg_object.num_channels, figsize=tuple([item * scale for item in multi_channel_size_tuple]), sharey=True) # Type: matplotlib.figure.Figure, numpy.ndarray

        return fig, ax, axes # Type: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot, numpy.ndarray
    
    def display_plot(self, canvas):
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

    def get_emg_recordings(self, data_type):
        """
        Get the EMG recordings based on the specified data type.

        Parameters:
        - data_type (str): The type of EMG data to retrieve. Valid options are 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered'.

        Returns:
        - list: The EMG recordings based on the specified data type.

        Raises:
        - ValueError: If the specified data type is not supported.

        """
        if data_type == 'filtered':
            return self.emg_object.recordings_processed
        elif data_type == 'raw':
            return self.emg_object.recordings_raw
        elif data_type in ['rectified_raw', 'rectified_filtered']:
            attribute_name = f'recordings_{data_type}'
            if not hasattr(self.emg_object, attribute_name):
                setattr(self.emg_object, attribute_name, self.emg_object._process_emg_data(apply_filter=(data_type == 'rectified_filtered'), rectify=True))
            return getattr(self.emg_object, attribute_name)
        else:
            raise ValueError(f"Data type '{data_type}' is not supported. Please use 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered'.")

    def plot_channel_data(self, ax, time_axis, channel_data, start, end, stimulus_v, channel_index):
        ax.plot(time_axis, channel_data[start:end], label=f"Stimulus Voltage: {stimulus_v}")
        ax.set_title(f'{self.emg_object.channel_names[channel_index]}')
        ax.grid(True)
   
    def plot_latency_windows(self, ax, all_flags, channel_index):
        if all_flags:
            for window in self.emg_object.latency_windows:
                window.plot(ax=ax, channel_index=channel_index)
        # else:
        #     for window in self.emg_object.latency_windows:
        #         if window.should_plot:
        #             window.plot(ax=ax, channel_index=channel_index)

    def set_fig_labels_and_legends(self, fig, sup_title : str, x_title : str, y_title: str, plot_legend : bool, legend_elements : list):
        fig.suptitle(sup_title)
        if self.emg_object.num_channels == 1:
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
    def plot_emg(self, all_flags : bool = True, plot_legend : bool = True, data_type : str = 'filtered', canvas: FigureCanvas = None):
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

        time_axis, window_start_sample, window_end_sample = self.get_time_axis()
        fig, ax, axes = self.create_fig_and_axes(canvas=canvas)
        legend_elements = [window.get_legend_element() for window in self.emg_object.latency_windows] if plot_latency_windows else []
        emg_recordings = self.get_emg_recordings(data_type)

        for recording in emg_recordings:
            for channel_index, channel_data in enumerate(recording['channel_data']):
                current_ax = ax if self.emg_object.num_channels == 1 else axes[channel_index]
                
                self.plot_channel_data(current_ax, time_axis, channel_data, window_start_sample, window_end_sample, recording['stimulus_v'], channel_index)
                self.plot_latency_windows(current_ax, all_flags, channel_index)


        # Set labels and title, and display plot
        if self.emg_object.num_channels == 1:
            sup_title = 'EMG Overlay for Channel 0 (all recordings)'
        else:
            sup_title = 'EMG Overlay for All Channels (all recordings)'
        x_title = 'Time (ms)'
        y_title = 'EMG (mV)'

        self.set_fig_labels_and_legends(fig, sup_title, x_title, y_title, plot_legend, legend_elements)
        self.display_plot(canvas)

    def plot_emg_thresholded (self, latency_window_oi_name : str, emg_threshold_v : float = 0.3, method : str = None, all_flags : bool = False, plot_legend : bool = False, canvas : FigureCanvas = None):
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

        time_axis, window_start_sample, window_end_sample = self.get_time_axis()
        fig, ax, axes = self.create_fig_and_axes(canvas=canvas)
        # legend_elements = [window.get_legend_element() for window in self.emg_object.latency_windows] if plot_latency_windows else []
        emg_recordings = self.get_emg_recordings('filtered')

        latency_window_oi = next((window for window in self.emg_object.latency_windows 
                                  if window.name == latency_window_oi_name), None)
        if latency_window_oi is None:
            raise UnableToPlotError(f"No latency window named '{latency_window_oi_name}' found in the session.")

        # Plot the EMG arrays for each channel, only for the first 10ms
        for recording in emg_recordings:
            for channel_index, channel_data in enumerate(recording['channel_data']):
                current_ax = ax if self.emg_object.num_channels == 1 else axes[channel_index]
                latency_window_oi_amplitude = Transform_EMG.calculate_emg_amplitude(channel_data, 
                                                                             latency_window_oi.start_times[channel_index] + self.emg_object.stim_start, 
                                                                             latency_window_oi.end_times[channel_index] + self.emg_object.stim_start, 
                                                                             self.emg_object.scan_rate,  
                                                                             method=method)
                if latency_window_oi_amplitude > emg_threshold_v:  # Check EMG amplitude within H-reflex window
                    self.plot_channel_data(current_ax, time_axis, channel_data, window_start_sample, window_end_sample, recording['stimulus_v'], channel_index)
                    self.plot_latency_windows(current_ax, all_flags=False, channel_index=channel_index)
                    if plot_legend:
                        current_ax.legend()

        # Set labels and title, and display plot
        sup_title = f'EMG Recordings Over Threshold ({emg_threshold_v} V) for {latency_window_oi_name}'
        x_title = 'Time (ms)'
        y_title = 'EMG (mV)'

        self.set_fig_labels_and_legends(fig, sup_title, x_title, y_title, plot_legend=False)
        self.display_plot(canvas)
    
    def plot_suspectedH(self, h_threshold : float = 0.3, method : str = None, all_flags : bool = False, plot_legend : bool = False, canvas : FigureCanvas = None):
        has_h_reflex_latency_window = False
        for window in self.emg_object.latency_windows:
            if window.name.lower() in ('h-reflex', 'h_reflex', 'h reflex', 'hreflex'):
                has_h_reflex_latency_window = True
                break
        if not has_h_reflex_latency_window:
            raise UnableToPlotError("No H-reflex latency window detected. Please add an H-reflex latency window to the session.")
        self.plot_emg_thresholded('H-reflex', emg_threshold_v=h_threshold, method=method, 
                                  all_flags=all_flags, plot_legend=plot_legend, canvas=canvas)

    def plot_mmax(self, method : str = None, canvas : FigureCanvas = None):
        """
        Plots overlayed M-response and H-reflex curves for each recorded channel.

        Args:
            channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
            method (str, optional): The method used to calculate the mean and standard deviation. Options are 'rms', 'avg_rectified', or 'peak_to_trough'. Default is 'rms'.
        """
        # Set method to default if not specified.
        if method is None:
            method = self.emg_object.default_method

        channel_names = self.emg_object.channel_names

        fig, ax, axes = self.create_fig_and_axes(canvas=canvas, figsizes = 'small')

        emg_recordings = self.get_emg_recordings('filtered')

        # Create a superlist to store all M-wave amplitudes for each channel for later y-axis adjustment.
        all_m_max_amplitudes = []
        
        # Plot the M-wave and H-response amplitudes for each channel
        for channel_index in range(self.emg_object.num_channels):
            m_wave_amplitudes = []
            stimulus_voltages = []

            # Append the M-wave and H-response amplitudes for each recording into the superlist.
            for recording in emg_recordings:
                channel_data = recording['channel_data'][channel_index]
                stimulus_v = recording['stimulus_v']
                             
                try:
                    m_wave_amplitude = Transform_EMG.calculate_emg_amplitude(channel_data, self.emg_object.m_start[channel_index] + self.emg_object.stim_start, self.emg_object.m_end[channel_index] + self.emg_object.stim_start, self.emg_object.scan_rate, method=method)
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
            if self.emg_object.num_channels == 1:
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

        # Set labels and title
        fig.suptitle('Average M-response values at M-max for each channel')
        if self.emg_object.num_channels == 1:
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

    def plot_reflexCurves (self, method=None, plot_legend=True, relative_to_mmax=False, manual_mmax=None, canvas=None):
        """
        Plots overlayed M-response and H-reflex curves for each recorded channel.

        Args:
            channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the session.
            method (str, optional): The method used to calculate the mean and standard deviation. Options are 'rms', 'average_rectified', 'average_unrectified', or 'peak_to_trough'. Default is 'rms'.
        """
        # Set method to default if not specified.
        if method is None:
            method = self.emg_object.default_method

        channel_names = self.emg_object.channel_names

        fig, ax, axes = self.create_fig_and_axes(canvas=canvas)

        # Plot the M-wave and H-response amplitudes for each channel
        for channel_index in range(self.emg_object.num_channels):
            m_wave_amplitudes = []
            h_response_amplitudes = []
            stimulus_voltages = []

            # Append the M-wave and H-response amplitudes for each recording into the superlist.
            for recording in self.emg_object.recordings_processed:
                channel_data = recording['channel_data'][channel_index]
                stimulus_v = recording['stimulus_v']
                
                try:
                    m_wave_amplitude = Transform_EMG.calculate_emg_amplitude(channel_data, self.emg_object.m_start[channel_index] + self.emg_object.stim_start, self.emg_object.m_end[channel_index] + self.emg_object.stim_start, self.emg_object.scan_rate, method=method)
                    h_response_amplitude = Transform_EMG.calculate_emg_amplitude(channel_data, self.emg_object.h_start[channel_index] + self.emg_object.stim_start, self.emg_object.h_end[channel_index] + self.emg_object.stim_start, self.emg_object.scan_rate, method=method)
                except ValueError:
                    raise UnableToPlotError(f"The method {method} is not supported. Please use 'rms', 'average_rectified', 'average_unrectified', or 'peak_to_trough'.")

                m_wave_amplitudes.append(m_wave_amplitude)
                h_response_amplitudes.append(h_response_amplitude)
                stimulus_voltages.append(stimulus_v)
            
            # Convert superlists to numpy arrays.
            m_wave_amplitudes = np.array(m_wave_amplitudes)
            h_response_amplitudes = np.array(h_response_amplitudes)
            stimulus_voltages = np.array(stimulus_voltages)

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            if relative_to_mmax:
                if manual_mmax is not None:
                    m_max = manual_mmax[channel_index]
                else:
                    try:
                        m_max = self.emg_object.get_m_max(method=method, channel_index=channel_index)
                    except Transform_EMG.NoCalculableMmaxError:
                        raise UnableToPlotError(f'M-max could not be calculated for channel {channel_index}.')
                m_wave_amplitudes = [amplitude / m_max for amplitude in m_wave_amplitudes]
                h_response_amplitudes = [amplitude / m_max for amplitude in h_response_amplitudes]    

            if self.emg_object.num_channels == 1:
                ax.scatter(stimulus_voltages, m_wave_amplitudes, color=self.emg_object.m_color, label='M-wave', marker='o')
                ax.scatter(stimulus_voltages, h_response_amplitudes, color=self.emg_object.h_color, label='H-response', marker='o')
                ax.set_title(f'{channel_names[0]}')
                ax.grid(True)
                if plot_legend:
                    ax.legend()
                    
            else:
                axes[channel_index].scatter(stimulus_voltages, m_wave_amplitudes, color=self.emg_object.m_color, label='M-wave', marker='o')
                axes[channel_index].scatter(stimulus_voltages, h_response_amplitudes, color=self.emg_object.h_color, label='H-response', marker='o')
                axes[channel_index].set_title(f'{channel_names[channel_index]}')
                axes[channel_index].grid(True)
                if plot_legend:
                    axes[channel_index].legend()
                    

        # Set labels and title
        fig.suptitle('M-response and H-reflex Curves')
        if self.emg_object.num_channels == 1:
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

    def plot_m_curves_smoothened (self, method=None, relative_to_mmax=False, manual_mmax=None, canvas=None):
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

        channel_names = self.emg_object.channel_names

        fig, ax, axes = self.create_fig_and_axes(canvas=canvas)

        # Plot the M-wave and H-response amplitudes for each channel
        for channel_index in range(self.emg_object.num_channels):
            m_wave_amplitudes = []
            h_response_amplitudes = []
            stimulus_voltages = []

            # Append the M-wave and H-response amplitudes for each recording into the superlist.
            for recording in self.emg_object.recordings_processed:
                channel_data = recording['channel_data'][channel_index]
                stimulus_v = recording['stimulus_v']
                

                try:
                    m_wave_amplitude = Transform_EMG.calculate_emg_amplitude(channel_data, self.emg_object.m_start[channel_index] + self.emg_object.stim_start, self.emg_object.m_end[channel_index] + self.emg_object.stim_start, self.emg_object.scan_rate, method=method)
                    h_response_amplitude = Transform_EMG.calculate_emg_amplitude(channel_data, self.emg_object.h_start[channel_index] + self.emg_object.stim_start, self.emg_object.h_end[channel_index] + self.emg_object.stim_start, self.emg_object.scan_rate, method=method)
                except ValueError:
                    raise UnableToPlotError(f"The method {method} is not supported. Please use 'rms', 'average_rectified', 'average_unrectified', or 'peak_to_trough'.")

                m_wave_amplitudes.append(m_wave_amplitude)
                h_response_amplitudes.append(h_response_amplitude)
                stimulus_voltages.append(stimulus_v)
            
            # Convert superlists to numpy arrays.
            m_wave_amplitudes = np.array(m_wave_amplitudes)
            h_response_amplitudes = np.array(h_response_amplitudes)
            stimulus_voltages = np.array(stimulus_voltages)

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            if relative_to_mmax:
                if manual_mmax is not None:
                    m_max = manual_mmax[channel_index]
                else:
                    m_max = self.emg_object.get_m_max(method=method, channel_index=channel_index)
                m_wave_amplitudes = [amplitude / m_max for amplitude in m_wave_amplitudes]
                h_response_amplitudes = [amplitude / m_max for amplitude in h_response_amplitudes]    

            # Smoothen the data
            m_wave_amplitudes = Transform_EMG.savgol_filter_y(m_wave_amplitudes)
            # h_response_amplitudes = np.gradient(m_wave_amplitudes, stimulus_voltages)

            if self.emg_object.num_channels == 1:
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
        if self.emg_object.num_channels == 1:
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
    def plot_reflexCurves(self, method=None, plot_legend=True, relative_to_mmax=False, manual_mmax=None, canvas=None):
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

        # Unpack processed session recordings.
        recordings = []
        for session in self.emg_object.emg_sessions:
            recordings.extend(session.recordings_processed)
        sorted_recordings = sorted(recordings, key=lambda x: x['stimulus_v'])

        fig, ax, axes = self.create_fig_and_axes(canvas=canvas)

        # Get unique binned stimulus voltages
        stimulus_voltages = sorted(list(set([round(recording['stimulus_v'] / self.emg_object.bin_size) * self.emg_object.bin_size for recording in sorted_recordings])))

        # Plot the M-wave and H-response amplitudes for each channel
        for channel_index in range(self.emg_object.num_channels):
            m_wave_means = []
            m_wave_stds = []
            h_response_means = []
            h_response_stds = []
            for stimulus_v in stimulus_voltages:
                m_wave_amplitudes = []
                h_response_amplitudes = []
                
                # Append the M-wave and H-response amplitudes for the binned voltage into a list.
                for recording in recordings:
                    binned_stimulus_v = round(recording['stimulus_v'] / self.emg_object.bin_size) * self.emg_object.bin_size
                    if binned_stimulus_v == stimulus_v:
                        channel_data = recording['channel_data'][channel_index]

                        try:
                            m_wave_amplitude = Transform_EMG.calculate_emg_amplitude(channel_data, self.emg_object.m_start[channel_index] + self.emg_object.stim_start, self.emg_object.m_end[channel_index] + self.emg_object.stim_start, self.emg_object.scan_rate, method=method)
                            h_response_amplitude = Transform_EMG.calculate_emg_amplitude(channel_data, self.emg_object.h_start[channel_index] + self.emg_object.stim_start, self.emg_object.h_end[channel_index] + self.emg_object.stim_start, self.emg_object.scan_rate, method=method)
                        except ValueError:
                            raise UnableToPlotError(f"The method {method} is not supported. Please use 'rms', 'average_rectified', 'average_unrectified', or 'peak_to_trough'.")

                        m_wave_amplitudes.append(m_wave_amplitude)
                        h_response_amplitudes.append(h_response_amplitude)
                
                # Calculate the mean and standard deviation of the M-wave and H-response amplitudes for the binned voltage.
                m_wave_mean = np.mean(m_wave_amplitudes)
                m_wave_std = np.std(m_wave_amplitudes)
                h_response_mean = np.mean(h_response_amplitudes)
                h_response_std = np.std(h_response_amplitudes)

                # Append the mean and standard deviation to the superlist.
                m_wave_means.append(m_wave_mean)
                m_wave_stds.append(m_wave_std)
                h_response_means.append(h_response_mean)
                h_response_stds.append(h_response_std)

            # Convert superlists to numpy arrays.
            m_wave_means = np.array(m_wave_means)
            m_wave_stds = np.array(m_wave_stds)
            h_response_means = np.array(h_response_means)
            h_response_stds = np.array(h_response_stds)

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            if relative_to_mmax:
                if manual_mmax is not None:
                    channel_m_max = manual_mmax[channel_index]
                else:
                    channel_m_max = self.emg_object.get_avg_m_max(method=method, channel_index=channel_index)
                m_wave_means = [(amplitude / channel_m_max) for amplitude in m_wave_means]
                m_wave_stds = [(amplitude / channel_m_max) for amplitude in m_wave_stds]
                h_response_means = [(amplitude / channel_m_max) for amplitude in h_response_means]
                h_response_stds = [(amplitude / channel_m_max) for amplitude in h_response_stds]

            if self.emg_object.num_channels == 1:
                ax.plot(stimulus_voltages, m_wave_means, color=self.emg_object.m_color, label='M-wave')
                ax.fill_between(stimulus_voltages, np.array(m_wave_means) - np.array(m_wave_stds), np.array(m_wave_means) + np.array(m_wave_stds), color='r', alpha=0.2)
                ax.plot(stimulus_voltages, h_response_means, color=self.emg_object.h_color, label='H-response')
                ax.fill_between(stimulus_voltages, np.array(h_response_means) - np.array(h_response_stds), np.array(h_response_means) + np.array(h_response_stds), color='b', alpha=0.2)
                ax.set_title(f'{channel_names[0]}')
                ax.grid(True)
                if plot_legend:
                    ax.legend()
            else:
                axes[channel_index].plot(stimulus_voltages, m_wave_means, color=self.emg_object.m_color, label='M-wave')
                axes[channel_index].fill_between(stimulus_voltages, np.array(m_wave_means) - np.array(m_wave_stds), np.array(m_wave_means) + np.array(m_wave_stds), color='r', alpha=0.2)
                axes[channel_index].plot(stimulus_voltages, h_response_means, color=self.emg_object.h_color, label='H-response')
                axes[channel_index].fill_between(stimulus_voltages, np.array(h_response_means) - np.array(h_response_stds), np.array(h_response_means) + np.array(h_response_stds), color='b', alpha=0.2)
                axes[channel_index].set_title(f'{channel_names[channel_index]}')
                axes[channel_index].grid(True)
                if plot_legend:
                    axes[channel_index].legend()

        # Set labels and title
        fig.suptitle('M-response and H-reflex Curves')
        if self.emg_object.num_channels == 1:
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

    def plot_maxH(self, method=None, relative_to_mmax=False, manual_mmax=None, canvas=None):
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

        # Unpack processed session recordings.
        recordings = []
        for session in self.emg_object.emg_sessions:
            recordings.extend(session.recordings_processed)
        sorted_recordings = sorted(recordings, key=lambda x: x['stimulus_v'])

        fig, ax, axes = self.create_fig_and_axes(canvas=canvas, figsizes='small')

        # Get unique binned stimulus voltages
        stimulus_voltages = sorted(list(set([round(recording['stimulus_v'] / self.emg_object.bin_size) * self.emg_object.bin_size for recording in sorted_recordings])))

        for channel_index in range(self.emg_object.num_channels):
            if relative_to_mmax:
                m_wave_means = []
            max_h_reflex_voltage = None
            max_h_reflex_amplitude = -float('inf')
            
            # Find the binned voltage where the average H-reflex amplitude is maximal and calculate the mean M-wave responses for M-max correction if relative_to_mmax is True.
            for stimulus_v in stimulus_voltages:
                if relative_to_mmax:
                    m_wave_amplitudes = []
                h_response_amplitudes = []
                
                # Append the M-wave and H-response amplitudes for the binned voltage into a list.
                for recording in recordings:
                    binned_stimulus_v = round(recording['stimulus_v'] / self.emg_object.bin_size) * self.emg_object.bin_size
                    if binned_stimulus_v == stimulus_v:
                        channel_data = recording['channel_data'][channel_index]
                        
                        try:
                            m_wave_amplitude = Transform_EMG.calculate_emg_amplitude(channel_data, self.emg_object.m_start[channel_index] + self.emg_object.stim_start, self.emg_object.m_end[channel_index] + self.emg_object.stim_start, self.emg_object.scan_rate, method=method)
                            h_response_amplitude = Transform_EMG.calculate_emg_amplitude(channel_data, self.emg_object.h_start[channel_index] + self.emg_object.stim_start, self.emg_object.h_end[channel_index] + self.emg_object.stim_start, self.emg_object.scan_rate, method=method)
                        except ValueError:
                            raise UnableToPlotError(f"The method {method} is not supported. Please use 'rms', 'avg_rectified', or 'peak_to_trough'.")

                        if relative_to_mmax:
                            m_wave_amplitudes.append(m_wave_amplitude)
                        h_response_amplitudes.append(h_response_amplitude)
                
                if relative_to_mmax:
                    # Append the M-wave mean to the superlist.
                    m_wave_means.append(np.mean(m_wave_amplitudes))

                # Calculate the mean H-response amplitude for the binned voltage.
                h_response_mean = np.mean(h_response_amplitudes)

                # Update maximum H-reflex amplitude and voltage if applicable
                if h_response_mean > max_h_reflex_amplitude:
                    max_h_reflex_amplitude = h_response_mean
                    max_h_reflex_voltage = stimulus_v

            # Get data to plot in whisker plot.

            m_wave_amplitudes_max_h = []
            h_response_amplitudes_max_h = []

            for recording in recordings:
                binned_stimulus_v = round(recording['stimulus_v'] / self.emg_object.bin_size) * self.emg_object.bin_size
                if binned_stimulus_v == max_h_reflex_voltage:
                    channel_data = recording['channel_data'][channel_index]
                    
                    try:
                        m_wave_amplitude_max_h = Transform_EMG.calculate_emg_amplitude(channel_data, self.emg_object.m_start[channel_index] + self.emg_object.stim_start, self.emg_object.m_end[channel_index] + self.emg_object.stim_start, self.emg_object.scan_rate, method=method)
                        h_response_amplitude_max_h = Transform_EMG.calculate_emg_amplitude(channel_data, self.emg_object.h_start[channel_index] + self.emg_object.stim_start, self.emg_object.h_end[channel_index] + self.emg_object.stim_start, self.emg_object.scan_rate, method=method)
                    except ValueError:
                        raise UnableToPlotError(f"The method {method} is not supported. Please use 'rms', 'avg_rectified', or 'peak_to_trough'.")
                
                    m_wave_amplitudes_max_h.append(m_wave_amplitude_max_h)
                    h_response_amplitudes_max_h.append(h_response_amplitude_max_h)

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            if relative_to_mmax:
                m_wave_means = np.array(m_wave_means)
                if manual_mmax is not None:
                    m_max = manual_mmax[channel_index]
                else:     
                    m_max = self.emg_object.get_avg_m_max(method=method, channel_index=channel_index)
                m_wave_amplitudes_max_h = [amplitude / m_max for amplitude in m_wave_amplitudes_max_h]
                h_response_amplitudes_max_h = [amplitude / m_max for amplitude in h_response_amplitudes_max_h]

            # Plot the M-wave and H-response amplitudes for the maximum H-reflex voltage.
            m_x = 1
            h_x = 2.5
            if self.emg_object.num_channels == 1:
                ax.plot(m_x, [m_wave_amplitudes_max_h], color=self.emg_object.m_color, marker='o', markersize=5)
                ax.annotate(f'n={len(m_wave_amplitudes_max_h)}', xy=(m_x + 0.4, np.mean(m_wave_amplitudes_max_h)), ha='center', color=self.emg_object.m_color)
                ax.errorbar(m_x, np.mean(m_wave_amplitudes_max_h), yerr=np.std(m_wave_amplitudes_max_h), color='black', marker='+', markersize=10, capsize=10)
                
                ax.plot(h_x, [h_response_amplitudes_max_h], color=self.emg_object.h_color, marker='o', markersize=5)
                ax.annotate(f'n={len(h_response_amplitudes_max_h)}', xy=(h_x + 0.4, np.mean(h_response_amplitudes_max_h)), ha='center', color=self.emg_object.h_color)
                ax.errorbar(h_x, np.mean(h_response_amplitudes_max_h), yerr=np.std(h_response_amplitudes_max_h), color='black', marker='+', markersize=10, capsize=10)
                
                ax.set_xticks([m_x, h_x])
                ax.set_xticklabels(['M-response', 'H-reflex'])
                ax.set_title(f'{channel_names[0]} ({round(max_h_reflex_voltage + self.emg_object.bin_size / 2, 2)} ± {self.emg_object.bin_size/2}V)')
                ax.set_xlim(m_x-1, h_x+1) # Set x-axis limits for each subplot to better center data points.
            else:
                axes[channel_index].plot(m_x, [m_wave_amplitudes_max_h], color=self.emg_object.m_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(m_wave_amplitudes_max_h)}', xy=(m_x + 0.4, np.mean(m_wave_amplitudes_max_h)), ha='center', color=self.emg_object.m_color)
                axes[channel_index].errorbar(m_x, np.mean(m_wave_amplitudes_max_h), yerr=np.std(m_wave_amplitudes_max_h), color='black', marker='+', markersize=10, capsize=10)

                axes[channel_index].plot(h_x, [h_response_amplitudes_max_h], color=self.emg_object.h_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(h_response_amplitudes_max_h)}', xy=(h_x + 0.4, np.mean(h_response_amplitudes_max_h)), ha='center', color=self.emg_object.h_color)
                axes[channel_index].errorbar(h_x, np.mean(h_response_amplitudes_max_h), yerr=np.std(h_response_amplitudes_max_h), color='black', marker='+', markersize=10, capsize=10)
                
                axes[channel_index].set_title(f'{channel_names[channel_index]} ({round(max_h_reflex_voltage + self.emg_object.bin_size / 2, 2)} ± {self.emg_object.bin_size/2}V)')
                axes[channel_index].set_xticks([m_x, h_x])
                axes[channel_index].set_xticklabels(['M-response', 'H-reflex'])
                axes[channel_index].set_xlim(m_x-1, h_x+1) # Set x-axis limits for each subplot to better center data points.
        
        # Set labels and title
        fig.suptitle('EMG Responses at Max H-reflex Stimulation')
        if self.emg_object.num_channels == 1:
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

    # def plot_mmax(self, channel_names=[], method='rms'):
    #     """
    #     Plots the average M-max for each session in the dataset.

    #     Args:
    #         channel_names (list): List of custom channel names. Default is an empty list.
    #         method (str): Method for calculating the amplitude. Options are 'rms', 'avg_rectified', or 'peak_to_trough'. Default is 'rms'.
    #         relative_to_mmax (bool): Flag indicating whether to make the M-wave amplitudes relative to the maximum M-wave amplitude. Default is False.
    #         manual_mmax (float): Manual value for the maximum M-wave amplitude. Default is None.

    #     Returns:
    #         None
    #     """
    #     # Handle custom channel names parameter if specified.
    #     customNames = False
    #     if len(channel_names) == 0:
    #         pass
    #     elif len(channel_names) != self.dataset.num_channels:
    #         print(f">! Error: list of custom channel names does not match the number of recorded channels. The entered list is {len(channel_names)} names long, but {self.dataset.num_channels} channels were recorded.")
    #     elif len(channel_names) == self.dataset.num_channels:
    #         customNames = True

    #     # Unpack processed session recordings.
    #     recordings = []
    #     for session in self.dataset.emg_sessions:
    #         recordings.extend(session.recordings_processed)
    #     sorted_recordings = sorted(recordings, key=lambda x: x['stimulus_v'])

    #     # Create a figure and axis
    #     if self.dataset.num_channels == 1:
    #         fig, ax = plt.subplots(figsize=(6, 4))
    #         axes = [ax]
    #     else:
    #         fig, axes = plt.subplots(nrows=1, ncols=self.dataset.num_channels, figsize=(8, 4), sharey=True)

    #     # Get unique binned stimulus voltages
    #     stimulus_voltages = sorted(list(set([round(recording['stimulus_v'] / self.dataset.bin_size) * self.dataset.bin_size for recording in sorted_recordings])))

    #     for channel_index in range(self.dataset.num_channels):
    #         m_wave_means = []
    #         all_m_max_amplitudes = []

    #         # Find the binned voltage where the average H-reflex amplitude is maximal and calculate the mean M-wave responses for M-max correction if relative_to_mmax is True.
    #         for stimulus_v in stimulus_voltages:
    #             m_wave_amplitudes = []
    
    #             # Append the M-wave and H-response amplitudes for the binned voltage into a list.
    #             for recording in recordings:
    #                 binned_stimulus_v = round(recording['stimulus_v'] / self.dataset.bin_size) * self.dataset.bin_size
    #                 if binned_stimulus_v == stimulus_v:
    #                     channel_data = recording['channel_data'][channel_index]
    #                     if method == 'rms':
    #                         m_wave_amplitude = EMG_Transformer.calculate_rms_amplitude(channel_data, self.dataset.m_start[channel_index] + self.dataset.stim_start, self.dataset.m_end[channel_index] + self.dataset.stim_start, self.dataset.scan_rate)
    #                     elif method == 'avg_rectified':
    #                         m_wave_amplitude = EMG_Transformer.calculate_average_amplitude_rectified(channel_data, self.dataset.m_start[channel_index] + self.dataset.stim_start, self.dataset.m_end[channel_index] + self.dataset.stim_start, self.dataset.scan_rate)
    #                     elif method == 'peak_to_trough':
    #                         m_wave_amplitude = EMG_Transformer.calculate_peak_to_trough_amplitude(channel_data, self.dataset.m_start[channel_index] + self.dataset.stim_start, self.dataset.m_end[channel_index] + self.dataset.stim_start, self.dataset.scan_rate)
    #                     else:
    #                         print(f">! Error: method {method} is not supported. Please use 'rms', 'avg_rectified', or 'peak_to_trough'.")
    #                         return

    #                     m_wave_amplitudes.append(m_wave_amplitude)
                
    #             # Append the M-wave mean to the superlist.
    #             m_wave_means.append(np.mean(m_wave_amplitudes))

    #         # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
    #         m_wave_means = np.array(m_wave_means)
    #         m_max, mmax_low_stim, _ = EMG_Transformer.get_avg_mmax(stimulus_voltages, m_wave_means, **self.dataset.m_max_args, return_mmax_stim_range=True)
            
    #         # Filter out M-wave amplitudes below the M-max stimulus threshold.
    #         mask = (stimulus_voltages >= mmax_low_stim)
    #         m_max_amplitudes = m_wave_amplitudes[mask]

    #         # Append M-wave amplitudes to superlist for y-axis adjustment.
    #         all_m_max_amplitudes.extend(m_max_amplitudes)

    #         m_x = 1 # Set x-axis position of the M-wave data.
    #         if self.session.num_channels == 1:
    #             ax.plot(m_x, [m_max_amplitudes], color=self.session.m_color, marker='o', markersize=5)
    #             ax.annotate(f'n={len(m_max_amplitudes)}\nM-max: {m_max:.2f}mV\nStim.: above {mmax_low_stim:.2f}V', xy=(m_x + 0.2, np.mean(m_max_amplitudes) - 0.2), ha='left', va='center', color='black')
    #             ax.errorbar(m_x, np.mean(m_max_amplitudes), yerr=np.std(m_max_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
    #             ax.set_xticklabels(['M-response'])
    #             ax.set_title('Channel 0')
    #             if customNames:
    #                 ax.set_title(f'{channel_names[0]}')
    #             ax.set_xlim(m_x-1, m_x+1.5) # Set x-axis limits for each subplot to better center data points.
    #             ax.set_ylim(0, 1.1 * max(all_m_max_amplitudes))
    #         else:
    #             axes[channel_index].plot(m_x, [m_max_amplitudes], color=self.session.m_color, marker='o', markersize=5)
    #             axes[channel_index].annotate(f'n={len(m_max_amplitudes)}\nM-max: {m_max:.2f}mV\nStim.: above {mmax_low_stim:.2f}V', xy=(m_x + 0.2, np.mean(m_max_amplitudes)), ha='left', va='center', color='black')
    #             axes[channel_index].errorbar(m_x, np.mean(m_max_amplitudes), yerr=np.std(m_max_amplitudes), color='black', marker='+', markersize=10, capsize=10)

    #             axes[channel_index].set_title(f'Channel {channel_index}' if not channel_names else channel_names[channel_index])
    #             axes[channel_index].set_xticks([m_x])
    #             axes[channel_index].set_xticklabels(['M-response'])
    #             if customNames:
    #                 axes[channel_index].set_title(f'{channel_names[channel_index]}')
    #             axes[channel_index].set_xlim(m_x-1, m_x+1.5) # Set x-axis limits for each subplot to better center data points.
    #             axes[channel_index].set_ylim(0, 1.1 * max(all_m_max_amplitudes))
    

    #     # Set labels and title
    #     fig.suptitle(f'Average M-response values at M-max for each channel')
    #     if self.session.num_channels == 1:
    #         # ax.set_xlabel('Stimulus Voltage (V)')
    #         ax.set_ylabel(f'Reflex Ampl. (mV, {method})')

    #     else:
    #         # fig.supxlabel('Stimulus Voltage (V)')
    #         fig.supylabel(f'Reflex Ampl. (mV, {method})')

    #     # Adjust subplot spacing
    #     plt.subplots_adjust(**self.session.subplot_adjust_args)

    #     # Show the plot
    #     plt.show()

class UnableToPlotError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)