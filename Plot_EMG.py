import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import EMG_Transformer


class EMGPlotter:
    """
    A parent class for plotting EMG data.
    
    Attributes:
        data: The EMG data object to be imported.
    """
    def __init__(self, data):
        self.data = data
        # plt.switch_backend('QtAgg')
        
    def set_plot_defaults(self):
        """
        Set plot font/style defaults for returned graphs.
        """
        plt.rcParams.update({'figure.titlesize': self.data.title_font_size})
        plt.rcParams.update({'figure.labelsize': self.data.axis_label_font_size, 'figure.labelweight': 'bold'})
        plt.rcParams.update({'axes.titlesize': self.data.axis_label_font_size, 'axes.titleweight': 'bold'})
        plt.rcParams.update({'xtick.labelsize': self.data.tick_font_size, 'ytick.labelsize': self.data.tick_font_size})

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

    def __init__(self, data):
        """
        Initializes an instance of the EMGSessionPlotter class.

        Args:
            data (EMGSession): The EMGSession object to be imported containing the EMG data.

        Returns:
            None
        """
        super().__init__(data)

        from Analyze_EMG import EMGSession
        if isinstance(data, EMGSession):
            self.session = data
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

    # EMGSession plotting functions
    def plot_emg (self, all_flags=False, m_flags=False, h_flags=False, data_type='filtered', canvas=None):
        """
        Plots EMG data from a Pickle file for a specified time window.

        Args:
            channel_names (list, optional): List of custom channel names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
            m_flags (bool, optional): Flag to indicate whether to plot markers for muscle onset and offset. Default is False.
            h_flags (bool, optional): Flag to indicate whether to plot markers for hand onset and offset. Default is False.
            data_type (str, optional): Type of EMG data to plot. Options are 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered'. Default is 'filtered'.

        Returns:
            None
        """
        if all_flags:
            m_flags = True
            h_flags = True

        channel_names = self.session.channel_names

        # Calculate time values based on the scan rate
        time_values_ms = np.arange(self.session.num_samples) * 1000 / self.session.scan_rate  # Time values in milliseconds

        # Determine the number of samples for the desired time window in ms
        num_samples_time_window = int(self.session.time_window_ms * self.session.scan_rate / 1000)  # Convert time window to number of samples

        # Slice the time array for the time window
        time_axis = time_values_ms[:num_samples_time_window] - self.session.stim_delay

        # Create a figure and axis
        if self.session.num_channels == 1:
            fig, ax = plt.subplots(figsize=(8, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(nrows=1, ncols=self.session.num_channels, figsize=(12, 4), sharey=True)

        # Establish type of EMG data to plot
        if data_type == 'filtered':
            emg_recordings = self.session.recordings_processed
        elif data_type == 'raw':
            emg_recordings = self.session.recordings_raw
        elif data_type == 'rectified_raw':
            if not hasattr(self.session, 'recordings_rectified_raw'):
                self.session.recordings_rectified_raw = self.session._process_emg_data(apply_filter=False, rectify=True)
            emg_recordings = self.session.recordings_rectified_raw
        elif data_type == 'rectified_filtered':
            if not hasattr(self.session, 'recordings_rectified_filtered'):
                self.session.recordings_rectified_filtered = self.session._process_emg_data(apply_filter=True, rectify=True)
            emg_recordings = self.session.recordings_rectified_filtered
        else:
            raise UnableToPlotError(f"data type {data_type} is not supported. Please use 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered'.")

        
        # Plot the EMG arrays for each channel, only for the first 10ms
        for recording in emg_recordings:
            for channel_index, channel_data in enumerate(recording['channel_data']):
                if self.session.num_channels == 1:
                    ax.plot(time_axis, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                    ax.set_title(f'{channel_names[0]}')
                    ax.grid(True)
                    #ax.legend()
                    if m_flags:
                        ax.axvline(self.session.m_start[channel_index], color=self.session.m_color, linestyle=self.session.flag_style)
                        ax.axvline(self.session.m_end[channel_index], color=self.session.m_color, linestyle=self.session.flag_style)                         
                    if h_flags:
                        ax.axvline(self.session.h_start[channel_index], color=self.session.h_color, linestyle=self.session.flag_style)
                        ax.axvline(self.session.h_end[channel_index], color=self.session.h_color, linestyle=self.session.flag_style)                       
                else:
                    axes[channel_index].plot(time_axis, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                    axes[channel_index].set_title(f'{channel_names[channel_index]}')
                    axes[channel_index].grid(True)
                    #axes[channel_index].legend()
                    if m_flags:
                        axes[channel_index].axvline(self.session.m_start[channel_index], color=self.session.m_color, linestyle=self.session.flag_style)
                        axes[channel_index].axvline(self.session.m_end[channel_index], color=self.session.m_color, linestyle=self.session.flag_style)
                    if h_flags:
                        axes[channel_index].axvline(self.session.h_start[channel_index], color=self.session.h_color, linestyle=self.session.flag_style)
                        axes[channel_index].axvline(self.session.h_end[channel_index], color=self.session.h_color, linestyle=self.session.flag_style)

        # Set labels and title
        if self.session.num_channels == 1:
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('EMG (mV)')
            fig.suptitle('EMG Overlay for Channel 0 (all recordings)')
        else:
            fig.suptitle('EMG Overlay for All Channels (all recordings)')
            fig.supxlabel('Time (ms)')
            fig.supylabel('EMG (mV)')

        # Adjust subplot spacing
        plt.subplots_adjust(**self.session.subplot_adjust_args)

        # Show the plot
        if canvas:
            canvas.draw()
        else:
            plt.show()

    def plot_suspectedH (self, h_threshold=0.3, method=None, plot_legend=False, canvas= None):
        """
        Detects session recordings with potential H-reflexes and plots them.

        Args:
            channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
            h_threshold (float, optional): Detection threshold of the average rectified EMG response in millivolts in the H-relfex window. Defaults to 0.3mV.
            plot_legend (bool, optional): Whether to plot legends. Defaults to False.
        """

        if method is None:
            method = self.session.default_method

        channel_names = self.session.channel_names

        # Calculate time values based on the scan rate
        time_values_ms = np.arange(self.session.num_samples) * 1000 / self.session.scan_rate  # Time values in milliseconds

        # Determine the number of samples for the first 10ms
        num_samples_time_window = int(self.session.time_window_ms * self.session.scan_rate / 1000)  # Convert time window to number of samples

        # Slice the time array for the time window
        time_axis = time_values_ms[:num_samples_time_window] - self.session.stim_delay

        # Create a figure and axis
        if self.session.num_channels == 1:
            fig, ax = plt.subplots(figsize=(8, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(nrows=1, ncols=self.session.num_channels, figsize=(12, 4), sharey=True)

        # Plot the EMG arrays for each channel, only for the first 10ms
        for recording in self.session.recordings_processed:
            for channel_index, channel_data in enumerate(recording['channel_data']):
                h_reflex_amplitude = EMG_Transformer.calculate_emg_amplitude(channel_data, 
                                                                             self.session.h_start[channel_index] + self.session.stim_delay, 
                                                                             self.session.h_end[channel_index] + self.session.stim_delay, 
                                                                             self.session.scan_rate,  
                                                                             method=method)
                if h_reflex_amplitude > h_threshold:  # Check EMG amplitude within H-reflex window
                    if self.session.num_channels == 1:
                        ax.plot(time_axis, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                        ax.set_title(f'{channel_names[0]}')
                        ax.grid(True)
                        if plot_legend:
                            ax.legend()
                    else:
                        axes[channel_index].plot(time_axis, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                        axes[channel_index].set_title(f'{channel_names[channel_index]}')
                        axes[channel_index].grid(True)
                        if plot_legend:
                            axes[channel_index].legend()

        # Set labels and title
        if self.session.num_channels == 1:
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('EMG (mV)')
            fig.suptitle(f'EMG Overlay for Channel 0 (H-reflex Amplitude > {h_threshold} mV)')
        else:
            fig.suptitle(f'EMG Overlay for All Channels (H-reflex Amplitude > {h_threshold} mV)')
            fig.supxlabel('Time (ms)')
            fig.supylabel('EMG (mV)')

        # Adjust subplot spacing
        plt.subplots_adjust(**self.session.subplot_adjust_args)

        # Show the plot
        if canvas:
            canvas.draw()
        else:
            plt.show()

    def plot_mmax(self, method=None, mmax_report=True, canvas=None):
        """
        Plots overlayed M-response and H-reflex curves for each recorded channel.

        Args:
            channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
            method (str, optional): The method used to calculate the mean and standard deviation. Options are 'rms', 'avg_rectified', or 'peak_to_trough'. Default is 'rms'.
        """
        # Set method to default if not specified.
        if method is None:
            method = self.session.default_method

        channel_names = self.session.channel_names

        # Create a figure and axis
        if self.session.num_channels == 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(nrows=1, ncols=self.session.num_channels, figsize=(8, 4), sharey=True)

        # Create a superlist to store all M-wave amplitudes for each channel for later y-axis adjustment.
        all_m_max_amplitudes = []
        
        # Plot the M-wave and H-response amplitudes for each channel
        for channel_index in range(self.session.num_channels):
            m_wave_amplitudes = []
            stimulus_voltages = []

            # Append the M-wave and H-response amplitudes for each recording into the superlist.
            for recording in self.session.recordings_processed:
                channel_data = recording['channel_data'][channel_index]
                stimulus_v = recording['stimulus_v']
                             
                try:
                    m_wave_amplitude = EMG_Transformer.calculate_emg_amplitude(channel_data, self.session.m_start[channel_index] + self.session.stim_delay, self.session.m_end[channel_index] + self.session.stim_delay, self.session.scan_rate, method=method)
                except ValueError:
                    raise UnableToPlotError(f"The method {method} is not supported. Please use 'rms', 'average_rectified', 'average_unrectified', or 'peak_to_trough'.")

                m_wave_amplitudes.append(m_wave_amplitude)
                stimulus_voltages.append(stimulus_v)
            
            # Convert superlists to numpy arrays.
            m_wave_amplitudes = np.array(m_wave_amplitudes)
            stimulus_voltages = np.array(stimulus_voltages)

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            m_max, mmax_low_stim, _ = EMG_Transformer.get_avg_mmax(stimulus_voltages, m_wave_amplitudes, mmax_report=mmax_report, return_mmax_stim_range=True, **self.session.m_max_args)
            # print(f'M-max for channel {channel_index}: {m_max:.2f}mV; between {mmax_low_stim:.2f}V and {mmax_high_stim:.2f}V')
            
            # Filter out M-wave amplitudes below the M-max stimulus threshold.
            mask = (stimulus_voltages >= mmax_low_stim)
            m_max_amplitudes = m_wave_amplitudes[mask]

            # Append M-wave amplitudes to superlist for y-axis adjustment.
            all_m_max_amplitudes.extend(m_max_amplitudes)

            m_x = 1 # Set x-axis position of the M-wave data.
            if self.session.num_channels == 1:
                ax.plot(m_x, [m_max_amplitudes], color=self.session.m_color, marker='o', markersize=5)
                ax.annotate(f'n={len(m_max_amplitudes)}\nM-max: {m_max:.2f}mV\nStim.: above {mmax_low_stim:.2f}V', xy=(m_x + 0.2, np.mean(m_max_amplitudes) - 0.2), ha='left', va='center', color='black')
                ax.errorbar(m_x, np.mean(m_max_amplitudes), yerr=np.std(m_max_amplitudes), color='black', marker='+', markersize=10, capsize=10)
                
                ax.set_xticklabels(['M-response'])
                ax.set_title(f'{channel_names[0]}')                    
                ax.set_xlim(m_x-1, m_x+1.5) # Set x-axis limits for each subplot to better center data points.
                ax.set_ylim(0, 1.1 * max(all_m_max_amplitudes))
            else:
                axes[channel_index].plot(m_x, [m_max_amplitudes], color=self.session.m_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(m_max_amplitudes)}\nM-max: {m_max:.2f}mV\nStim.: above {mmax_low_stim:.2f}V', xy=(m_x + 0.2, np.mean(m_max_amplitudes)), ha='left', va='center', color='black')
                axes[channel_index].errorbar(m_x, np.mean(m_max_amplitudes), yerr=np.std(m_max_amplitudes), color='black', marker='+', markersize=10, capsize=10)

                axes[channel_index].set_title(f'{channel_names[channel_index]}')
                axes[channel_index].set_xticks([m_x])
                axes[channel_index].set_xticklabels(['M-response'])                    
                axes[channel_index].set_xlim(m_x-1, m_x+1.5) # Set x-axis limits for each subplot to better center data points.
                axes[channel_index].set_ylim(0, 1.1 * max(all_m_max_amplitudes))

        # Set labels and title
        fig.suptitle('Average M-response values at M-max for each channel')
        if self.session.num_channels == 1:
            # ax.set_xlabel('Stimulus Voltage (V)')
            ax.set_ylabel(f'Reflex Ampl. (mV, {method})')

        else:
            # fig.supxlabel('Stimulus Voltage (V)')
            fig.supylabel(f'Reflex Ampl. (mV, {method})')

        # Adjust subplot spacing
        plt.subplots_adjust(**self.session.subplot_adjust_args)

        # Show the plot
        if canvas:
            canvas.draw()
        else:
            plt.show()

    def plot_reflexCurves (self, method=None, relative_to_mmax=False, manual_mmax=None, mmax_report=True, canvas=None):
        """
        Plots overlayed M-response and H-reflex curves for each recorded channel.

        Args:
            channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the session.
            method (str, optional): The method used to calculate the mean and standard deviation. Options are 'rms', 'average_rectified', 'average_unrectified', or 'peak_to_trough'. Default is 'rms'.
        """
        # Set method to default if not specified.
        if method is None:
            method = self.session.default_method

        channel_names = self.session.channel_names

        # Create a figure and axis
        if self.session.num_channels == 1:
            fig, ax = plt.subplots(figsize=(8, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(nrows=1, ncols=self.session.num_channels, figsize=(12, 4), sharey=True)

        # Plot the M-wave and H-response amplitudes for each channel
        for channel_index in range(self.session.num_channels):
            m_wave_amplitudes = []
            h_response_amplitudes = []
            stimulus_voltages = []

            # Append the M-wave and H-response amplitudes for each recording into the superlist.
            for recording in self.session.recordings_processed:
                channel_data = recording['channel_data'][channel_index]
                stimulus_v = recording['stimulus_v']
                
                try:
                    m_wave_amplitude = EMG_Transformer.calculate_emg_amplitude(channel_data, self.session.m_start[channel_index] + self.session.stim_delay, self.session.m_end[channel_index] + self.session.stim_delay, self.session.scan_rate, method=method)
                    h_response_amplitude = EMG_Transformer.calculate_emg_amplitude(channel_data, self.session.h_start[channel_index] + self.session.stim_delay, self.session.h_end[channel_index] + self.session.stim_delay, self.session.scan_rate, method=method)
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
                    m_max = manual_mmax
                else:
                    try:
                        m_max = EMG_Transformer.get_avg_mmax(stimulus_voltages, m_wave_amplitudes, mmax_report=mmax_report, **self.session.m_max_args)
                    except EMG_Transformer.NoCalculableMmaxError:
                        raise UnableToPlotError(f'M-max could not be calculated for channel {channel_index}.')
                m_wave_amplitudes = [amplitude / m_max for amplitude in m_wave_amplitudes]
                h_response_amplitudes = [amplitude / m_max for amplitude in h_response_amplitudes]    

            if self.session.num_channels == 1:
                ax.scatter(stimulus_voltages, m_wave_amplitudes, color=self.session.m_color, label='M-wave', marker='o')
                ax.scatter(stimulus_voltages, h_response_amplitudes, color=self.session.h_color, label='H-response', marker='o')
                ax.set_title(f'{channel_names[0]}')
                ax.grid(True)
                ax.legend()
                    
            else:
                axes[channel_index].scatter(stimulus_voltages, m_wave_amplitudes, color=self.session.m_color, label='M-wave', marker='o')
                axes[channel_index].scatter(stimulus_voltages, h_response_amplitudes, color=self.session.h_color, label='H-response', marker='o')
                axes[channel_index].set_title(f'{channel_names[channel_index]}')
                axes[channel_index].grid(True)
                axes[channel_index].legend()
                    

        # Set labels and title
        fig.suptitle('M-response and H-reflex Curves')
        if self.session.num_channels == 1:
            ax.set_xlabel('Stimulus Voltage (V)')
            ax.set_ylabel(f'Reflex Ampl. (mV, {method})')
            if relative_to_mmax:
                ax.set_ylabel(f'Reflex Ampl. (M-max, {method})')
        else:
            fig.supxlabel('Stimulus Voltage (V)')
            fig.supylabel(f'Reflex Ampl. (mV, {method})')
            if relative_to_mmax:
                fig.supylabel(f'Reflex Ampl. (M-max, {method})')
        
        # Adjust subplot spacing
        plt.subplots_adjust(**self.session.subplot_adjust_args)

        # Show the plot
        if canvas:
            canvas.draw()
        else:
            plt.show()

    def plot_m_curves_smoothened (self, method=None, relative_to_mmax=False, manual_mmax=None, mmax_report=True, canvas=None):
        """
        Plots overlayed M-response and H-reflex curves for each recorded channel.
        This plot is smoothened using a Savitzky-Golay filter, which therefore emulates the transformation used before calculating M-max in the EMG analysis.

        Args:
            channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
            method (str, optional): The method used to calculate the mean and standard deviation. Options are 'rms', 'avg_rectified', or 'peak_to_trough'. Default is 'rms'.
        """
        # Set method to default if not specified.
        if method is None:
            method = self.session.default_method

        channel_names = self.session.channel_names

        # Create a figure and axis
        if self.session.num_channels == 1:
            fig, ax = plt.subplots(figsize=(8, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(nrows=1, ncols=self.session.num_channels, figsize=(12, 4), sharey=True)

        # Plot the M-wave and H-response amplitudes for each channel
        for channel_index in range(self.session.num_channels):
            m_wave_amplitudes = []
            h_response_amplitudes = []
            stimulus_voltages = []

            # Append the M-wave and H-response amplitudes for each recording into the superlist.
            for recording in self.session.recordings_processed:
                channel_data = recording['channel_data'][channel_index]
                stimulus_v = recording['stimulus_v']
                

                try:
                    m_wave_amplitude = EMG_Transformer.calculate_emg_amplitude(channel_data, self.session.m_start[channel_index] + self.session.stim_delay, self.session.m_end[channel_index] + self.session.stim_delay, self.session.scan_rate, method=method)
                    h_response_amplitude = EMG_Transformer.calculate_emg_amplitude(channel_data, self.session.h_start[channel_index] + self.session.stim_delay, self.session.h_end[channel_index] + self.session.stim_delay, self.session.scan_rate, method=method)
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
                    m_max = manual_mmax
                else:
                    m_max = EMG_Transformer.get_avg_mmax(stimulus_voltages, m_wave_amplitudes, mmax_report=mmax_report **self.session.m_max_args)
                m_wave_amplitudes = [amplitude / m_max for amplitude in m_wave_amplitudes]
                h_response_amplitudes = [amplitude / m_max for amplitude in h_response_amplitudes]    

            # Smoothen the data
            m_wave_amplitudes = EMG_Transformer.savgol_filter_y(m_wave_amplitudes)
            # h_response_amplitudes = np.gradient(m_wave_amplitudes, stimulus_voltages)

            if self.session.num_channels == 1:
                ax.scatter(stimulus_voltages, m_wave_amplitudes, color=self.session.m_color, label='M-wave', marker='o')
                ax.scatter(stimulus_voltages, h_response_amplitudes, color=self.session.h_color, label='H-response', marker='o')
                ax.set_title(f'{channel_names[0]}')
                ax.grid(True)
                ax.legend()    
            else:
                axes[channel_index].scatter(stimulus_voltages, m_wave_amplitudes, color=self.session.m_color, label='M-wave', marker='o')
                axes[channel_index].scatter(stimulus_voltages, h_response_amplitudes, color=self.session.h_color, label='H-response', marker='o')
                axes[channel_index].set_title(f'{channel_names[channel_index]}')
                axes[channel_index].grid(True)
                axes[channel_index].legend()
                    
        # Set labels and title
        fig.suptitle('M-response and H-reflex Curves')
        if self.session.num_channels == 1:
            ax.set_xlabel('Stimulus Voltage (V)')
            ax.set_ylabel(f'Reflex Ampl. (mV, {method})')
            if relative_to_mmax:
                ax.set_ylabel(f'Reflex Ampl. (M-max, {method})')
        else:
            fig.supxlabel('Stimulus Voltage (V)')
            fig.supylabel(f'Reflex Ampl. (mV, {method})')
            if relative_to_mmax:
                fig.supylabel(f'Reflex Ampl. (M-max, {method})')
        
        # Adjust subplot spacing
        plt.subplots_adjust(**self.session.subplot_adjust_args)

        # Show the plot
        if canvas:
            canvas.draw()
        else:
            plt.show()

class EMGDatasetPlotter(EMGPlotter):
    """
    A class for EMG data from an EMGDataset object.

    Args:
        data (EMGDataset): The EMG dataset object to be imported.

    Attributes:
        dataset (EMGDataset): The imported EMG dataset object.

    Methods:
        plot_reflex_curves: Plots the average M-response and H-reflex curves for each channel.
        plot_max_h_reflex: Plots the M-wave and H-response amplitudes at the stimulation voltage where the average H-reflex is maximal.
        help: Displays the help text for the class.
    """
    def __init__(self, data):

        super().__init__(data)

        from Analyze_EMG import EMGDataset
        if isinstance(data, EMGDataset):
            self.dataset = data
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
    def plot_reflexCurves(self, method=None, relative_to_mmax=False, manual_mmax=None, mmax_report=True):
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
            method = self.dataset.default_method
        channel_names = self.dataset.channel_names


        # Unpack processed session recordings.
        recordings = []
        for session in self.dataset.emg_sessions:
            recordings.extend(session.recordings_processed)
        sorted_recordings = sorted(recordings, key=lambda x: x['stimulus_v'])

        # Create a figure and axis
        if self.dataset.num_channels == 1:
            fig, ax = plt.subplots(figsize=(8, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(nrows=1, ncols=self.dataset.num_channels, figsize=(12, 4), sharey=True)

        # Get unique binned stimulus voltages
        stimulus_voltages = sorted(list(set([round(recording['stimulus_v'] / self.dataset.bin_size) * self.dataset.bin_size for recording in sorted_recordings])))

        # Plot the M-wave and H-response amplitudes for each channel
        for channel_index in range(self.dataset.num_channels):
            m_wave_means = []
            m_wave_stds = []
            h_response_means = []
            h_response_stds = []
            for stimulus_v in stimulus_voltages:
                m_wave_amplitudes = []
                h_response_amplitudes = []
                
                # Append the M-wave and H-response amplitudes for the binned voltage into a list.
                for recording in recordings:
                    binned_stimulus_v = round(recording['stimulus_v'] / self.dataset.bin_size) * self.dataset.bin_size
                    if binned_stimulus_v == stimulus_v:
                        channel_data = recording['channel_data'][channel_index]

                        try:
                            m_wave_amplitude = EMG_Transformer.calculate_emg_amplitude(channel_data, self.dataset.m_start[channel_index] + self.dataset.stim_delay, self.dataset.m_end[channel_index] + self.dataset.stim_delay, self.dataset.scan_rate, method=method)
                            h_response_amplitude = EMG_Transformer.calculate_emg_amplitude(channel_data, self.dataset.h_start[channel_index] + self.dataset.stim_delay, self.dataset.h_end[channel_index] + self.dataset.stim_delay, self.dataset.scan_rate, method=method)
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
                    m_max = manual_mmax
                else:
                    try:
                        m_max = EMG_Transformer.get_avg_mmax(stimulus_voltages, m_wave_means, mmax_report=mmax_report, **self.dataset.m_max_args)
                    except EMG_Transformer.NoCalculableMmaxError:
                        raise UnableToPlotError(f'M-max could not be calculated for channel {channel_index}.')
                m_wave_means = [(amplitude / m_max) for amplitude in m_wave_means]
                m_wave_stds = [(amplitude / m_max) for amplitude in m_wave_stds]
                h_response_means = [(amplitude / m_max) for amplitude in h_response_means]
                h_response_stds = [(amplitude / m_max) for amplitude in h_response_stds]

            if self.dataset.num_channels == 1:
                ax.plot(stimulus_voltages, m_wave_means, color=self.dataset.m_color, label='M-wave')
                ax.fill_between(stimulus_voltages, np.array(m_wave_means) - np.array(m_wave_stds), np.array(m_wave_means) + np.array(m_wave_stds), color='r', alpha=0.2)
                ax.plot(stimulus_voltages, h_response_means, color=self.dataset.h_color, label='H-response')
                ax.fill_between(stimulus_voltages, np.array(h_response_means) - np.array(h_response_stds), np.array(h_response_means) + np.array(h_response_stds), color='b', alpha=0.2)
                ax.set_title(f'{channel_names[0]}')
                ax.grid(True)
                ax.legend()
            else:
                axes[channel_index].plot(stimulus_voltages, m_wave_means, color=self.dataset.m_color, label='M-wave')
                axes[channel_index].fill_between(stimulus_voltages, np.array(m_wave_means) - np.array(m_wave_stds), np.array(m_wave_means) + np.array(m_wave_stds), color='r', alpha=0.2)
                axes[channel_index].plot(stimulus_voltages, h_response_means, color=self.dataset.h_color, label='H-response')
                axes[channel_index].fill_between(stimulus_voltages, np.array(h_response_means) - np.array(h_response_stds), np.array(h_response_means) + np.array(h_response_stds), color='b', alpha=0.2)
                axes[channel_index].set_title(f'{channel_names[channel_index]}')
                axes[channel_index].grid(True)
                axes[channel_index].legend()

        # Set labels and title
        fig.suptitle('M-response and H-reflex Curves')
        if self.dataset.num_channels == 1:
            ax.set_xlabel('Stimulus Voltage (V)')
            ax.set_ylabel(f'Reflex Ampl. (mV, {method})')
            if relative_to_mmax:
                ax.set_ylabel(f'Reflex Ampl. (M-max, {method})')
        else:
            fig.supxlabel('Stimulus Voltage (V)')
            fig.supylabel(f'Reflex Ampl. (mV, {method})')
            if relative_to_mmax:
                fig.supylabel(f'Reflex Ampl. (M-max, {method})')

        # Adjust subplot spacing
        plt.subplots_adjust(**self.dataset.subplot_adjust_args)

        # Show the plot
        plt.show()

    def plot_maxH(self, method=None, relative_to_mmax=False, manual_mmax=None, mmax_report=True):
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
            method = self.dataset.default_method
        channel_names = self.dataset.channel_names

        # Unpack processed session recordings.
        recordings = []
        for session in self.dataset.emg_sessions:
            recordings.extend(session.recordings_processed)
        sorted_recordings = sorted(recordings, key=lambda x: x['stimulus_v'])

        # Create a figure and axis
        if self.dataset.num_channels == 1:
            fig, ax = plt.subplots(figsize=(5, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(nrows=1, ncols=self.dataset.num_channels, figsize=(8, 4), sharey=True)

        # Get unique binned stimulus voltages
        stimulus_voltages = sorted(list(set([round(recording['stimulus_v'] / self.dataset.bin_size) * self.dataset.bin_size for recording in sorted_recordings])))

        for channel_index in range(self.dataset.num_channels):
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
                    binned_stimulus_v = round(recording['stimulus_v'] / self.dataset.bin_size) * self.dataset.bin_size
                    if binned_stimulus_v == stimulus_v:
                        channel_data = recording['channel_data'][channel_index]
                        
                        try:
                            m_wave_amplitude = EMG_Transformer.calculate_emg_amplitude(channel_data, self.dataset.m_start[channel_index] + self.dataset.stim_delay, self.dataset.m_end[channel_index] + self.dataset.stim_delay, self.dataset.scan_rate, method=method)
                            h_response_amplitude = EMG_Transformer.calculate_emg_amplitude(channel_data, self.dataset.h_start[channel_index] + self.dataset.stim_delay, self.dataset.h_end[channel_index] + self.dataset.stim_delay, self.dataset.scan_rate, method=method)
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
                binned_stimulus_v = round(recording['stimulus_v'] / self.dataset.bin_size) * self.dataset.bin_size
                if binned_stimulus_v == max_h_reflex_voltage:
                    channel_data = recording['channel_data'][channel_index]
                    
                    try:
                        m_wave_amplitude_max_h = EMG_Transformer.calculate_emg_amplitude(channel_data, self.dataset.m_start[channel_index] + self.dataset.stim_delay, self.dataset.m_end[channel_index] + self.dataset.stim_delay, self.dataset.scan_rate, method=method)
                        h_response_amplitude_max_h = EMG_Transformer.calculate_emg_amplitude(channel_data, self.dataset.h_start[channel_index] + self.dataset.stim_delay, self.dataset.h_end[channel_index] + self.dataset.stim_delay, self.dataset.scan_rate, method=method)
                    except ValueError:
                        raise UnableToPlotError(f"The method {method} is not supported. Please use 'rms', 'avg_rectified', or 'peak_to_trough'.")
                
                    m_wave_amplitudes_max_h.append(m_wave_amplitude_max_h)
                    h_response_amplitudes_max_h.append(h_response_amplitude_max_h)

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            if relative_to_mmax:
                m_wave_means = np.array(m_wave_means)
                if manual_mmax is not None:
                    m_max = manual_mmax
                else:     
                    m_max = EMG_Transformer.get_avg_mmax(stimulus_voltages, m_wave_means, mmax_report=mmax_report, **self.dataset.m_max_args)
                m_wave_amplitudes_max_h = [amplitude / m_max for amplitude in m_wave_amplitudes_max_h]
                h_response_amplitudes_max_h = [amplitude / m_max for amplitude in h_response_amplitudes_max_h]

            # Plot the M-wave and H-response amplitudes for the maximum H-reflex voltage.
            m_x = 1
            h_x = 2.5
            if self.dataset.num_channels == 1:
                ax.plot(m_x, [m_wave_amplitudes_max_h], color=self.dataset.m_color, marker='o', markersize=5)
                ax.annotate(f'n={len(m_wave_amplitudes_max_h)}', xy=(m_x + 0.4, np.mean(m_wave_amplitudes_max_h)), ha='center', color=self.dataset.m_color)
                ax.errorbar(m_x, np.mean(m_wave_amplitudes_max_h), yerr=np.std(m_wave_amplitudes_max_h), color='black', marker='+', markersize=10, capsize=10)
                
                ax.plot(h_x, [h_response_amplitudes_max_h], color=self.dataset.h_color, marker='o', markersize=5)
                ax.annotate(f'n={len(h_response_amplitudes_max_h)}', xy=(h_x + 0.4, np.mean(h_response_amplitudes_max_h)), ha='center', color=self.dataset.h_color)
                ax.errorbar(h_x, np.mean(h_response_amplitudes_max_h), yerr=np.std(h_response_amplitudes_max_h), color='black', marker='+', markersize=10, capsize=10)
                
                ax.set_xticks([m_x, h_x])
                ax.set_xticklabels(['M-response', 'H-reflex'])
                ax.set_title(f'{channel_names[0]} ({round(max_h_reflex_voltage + self.dataset.bin_size / 2, 2)} ± {self.dataset.bin_size/2}V)')
                ax.set_xlim(m_x-1, h_x+1) # Set x-axis limits for each subplot to better center data points.
            else:
                axes[channel_index].plot(m_x, [m_wave_amplitudes_max_h], color=self.dataset.m_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(m_wave_amplitudes_max_h)}', xy=(m_x + 0.4, np.mean(m_wave_amplitudes_max_h)), ha='center', color=self.dataset.m_color)
                axes[channel_index].errorbar(m_x, np.mean(m_wave_amplitudes_max_h), yerr=np.std(m_wave_amplitudes_max_h), color='black', marker='+', markersize=10, capsize=10)

                axes[channel_index].plot(h_x, [h_response_amplitudes_max_h], color=self.dataset.h_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(h_response_amplitudes_max_h)}', xy=(h_x + 0.4, np.mean(h_response_amplitudes_max_h)), ha='center', color=self.dataset.h_color)
                axes[channel_index].errorbar(h_x, np.mean(h_response_amplitudes_max_h), yerr=np.std(h_response_amplitudes_max_h), color='black', marker='+', markersize=10, capsize=10)
                
                axes[channel_index].set_title(f'{channel_names[channel_index]} ({round(max_h_reflex_voltage + self.dataset.bin_size / 2, 2)} ± {self.dataset.bin_size/2}V)')
                axes[channel_index].set_xticks([m_x, h_x])
                axes[channel_index].set_xticklabels(['M-response', 'H-reflex'])
                axes[channel_index].set_xlim(m_x-1, h_x+1) # Set x-axis limits for each subplot to better center data points.
        
        # Set labels and title
        fig.suptitle('EMG Responses at Max H-reflex Stimulation')
        if self.dataset.num_channels == 1:
            ax.set_xlabel('Response Type')
            ax.set_ylabel(f'EMG Amp. (mV, {method})')
            if relative_to_mmax:
                ax.set_ylabel(f'EMG Amp. (M-max, {method})')
        else:
            fig.supxlabel('Response Type')
            fig.supylabel(f'EMG Amp. (mV, {method})')
            if relative_to_mmax:
                fig.supylabel(f'EMG Amp. (M-max, {method})')


        # Adjust subplot spacing
        plt.subplots_adjust(**self.dataset.subplot_adjust_args)

        # Show the plot
        plt.show()

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
    #                         m_wave_amplitude = EMG_Transformer.calculate_rms_amplitude(channel_data, self.dataset.m_start[channel_index] + self.dataset.stim_delay, self.dataset.m_end[channel_index] + self.dataset.stim_delay, self.dataset.scan_rate)
    #                     elif method == 'avg_rectified':
    #                         m_wave_amplitude = EMG_Transformer.calculate_average_amplitude_rectified(channel_data, self.dataset.m_start[channel_index] + self.dataset.stim_delay, self.dataset.m_end[channel_index] + self.dataset.stim_delay, self.dataset.scan_rate)
    #                     elif method == 'peak_to_trough':
    #                         m_wave_amplitude = EMG_Transformer.calculate_peak_to_trough_amplitude(channel_data, self.dataset.m_start[channel_index] + self.dataset.stim_delay, self.dataset.m_end[channel_index] + self.dataset.stim_delay, self.dataset.scan_rate)
    #                     else:
    #                         print(f">! Error: method {method} is not supported. Please use 'rms', 'avg_rectified', or 'peak_to_trough'.")
    #                         return

    #                     m_wave_amplitudes.append(m_wave_amplitude)
                
    #             # Append the M-wave mean to the superlist.
    #             m_wave_means.append(np.mean(m_wave_amplitudes))

    #         # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
    #         m_wave_means = np.array(m_wave_means)
    #         m_max, mmax_low_stim, _ = EMG_Transformer.get_avg_mmax(stimulus_voltages, m_wave_means, mmax_report=mmax_report **self.dataset.m_max_args, return_mmax_stim_range=True)
            
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


# Custom Matplotlib canvas class for embedding plots in PyQt5 applications. Not currently used.
class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)

class UnableToPlotError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)