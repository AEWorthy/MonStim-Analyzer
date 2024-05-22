
"""
Classes to analyze and plot EMG data from individual sessions or an entire dataset of sessions.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import emg_transform as emg_transform
import yaml_config as yaml_config
import utils as utils
import copy

config = yaml_config.load_config('config.yml')

class EMGSession:
    """
    Class for analyzing and plotting data from a single recording session of variable channel numbers for within-session analyses and plotting.

    This module provides functions for analyzing data stored in Pickle files from a single EMG recording session.
    It includes functions to extract session parameters, plot all EMG data, and plot EMG data from suspected H-reflex recordings.
    Class must be instatiated with the Pickled session data file.

    Attributes:
            time_window (float): Time window to plot in milliseconds. Defaults to first 10ms.
            m_start_ms (float): Start time of the M-response window in milliseconds. Defaults to 2.0 ms.
            m_end_ms (float): End time of the M-response window in milliseconds. Defaults to 4.0 ms.
            h_start_ms (float): Start time of the suspected H-reflex window in milliseconds. Defaults to 4.0 ms.
            h_end_ms (float): End time of the suspected H-reflex window in milliseconds. Defaults to 7.0 ms.
    """
    def __init__(self, pickled_session_data):
        """
        Initialize an EMGSession instance.

        Args:
            pickled_session_data (str): filepath of .pickle session data file for this session.
        """
        self.load_session_data(pickled_session_data)
        self.m_start = config['m_start']
        self.m_end = config['m_end']
        self.h_start = config['h_start']
        self.h_end = config['h_end']
        self.time_window_ms = config['time_window']

        self.flag_style = config['flag_style']
        self.m_color = config['m_color']
        self.h_color = config['h_color']
        self.title_font_size = config['title_font_size']
        self.axis_label_font_size = config['axis_label_font_size']
        self.tick_font_size = config['tick_font_size']
        self.subplot_adjust_args = config['subplot_adjust_args']

        #Set plot font/style defaults for returned graphs
        plt.rcParams.update({'figure.titlesize': self.title_font_size})
        plt.rcParams.update({'figure.labelsize': self.axis_label_font_size, 'figure.labelweight': 'bold'})
        plt.rcParams.update({'axes.titlesize': self.axis_label_font_size, 'axes.titleweight': 'bold'})
        plt.rcParams.update({'xtick.labelsize': self.tick_font_size, 'ytick.labelsize': self.tick_font_size})
        
    def load_session_data(self, pickled_data):
        # Load the session data from the pickle file
        with open(pickled_data, 'rb') as pickle_file:
            session_data = pickle.load(pickle_file)

        # Access session-wide information
        session_info = session_data['session_info']
        self.session_name = session_info['session_name']
        self.num_channels = session_info['num_channels']
        self.scan_rate = session_info['scan_rate']
        self.num_samples = session_info['num_samples']
        self.stim_delay = session_info['stim_delay']
        self.stim_duration = session_info['stim_duration']
        self.stim_interval = session_info['stim_interval']
        self.emg_amp_gains = session_info['emg_amp_gains']

        # Access the raw EMG recordings. Sort by stimulus voltage.
        self.recordings_raw = sorted(session_data['recordings'], key=lambda x: x['stimulus_v']).copy()
        
        # Apply bandpass filter.
        self.recordings_filtered = copy.deepcopy(self.recordings_raw)
        for recording in self.recordings_filtered:
                for i, channel_emg in enumerate(recording['channel_data']):
                    filtered_emg = emg_transform.butter_bandpass_filter(channel_emg, self.scan_rate)
                    recording['channel_data'][i] = filtered_emg
                # recording['channel_data'] = emg_transform.correct_emg_to_baseline(recording['channel_data'], self.scan_rate, self.stim_delay)

        # Rectify the raw EMG data
        self.recordings_rectified_raw = copy.deepcopy(self.recordings_raw)
        for recording in self.recordings_rectified_raw:
                for i, channel_emg in enumerate(recording['channel_data']):
                    rectified_emg = emg_transform.rectify_emg(channel_emg)
                    recording['channel_data'][i] = rectified_emg
        
        # Rectify the processed EMG data
        self.recordings_rectified_processed = copy.deepcopy(self.recordings_filtered)
        for recording in self.recordings_rectified_processed:
                for i, channel_emg in enumerate(recording['channel_data']):
                    rectified_emg = emg_transform.rectify_emg(channel_emg)
                    recording['channel_data'][i] = rectified_emg

    def session_parameters (self):
        """
        Prints EMG recording session parameters from a Pickle file.
        """
        print(f"Session Name: {self.session_name}")
        print(f"# of Channels: {self.num_channels}")
        print(f"Scan rate (Hz): {self.scan_rate}")
        print(f"Samples/Channel: {self.num_samples}")
        print(f"Stimulus delay (ms): {self.stim_delay}")
        print(f"Stimulus duration (ms): {self.stim_duration}")
        print(f"Stimulus interval (s): {self.stim_interval}")
        print(f"EMG amp gains: {self.emg_amp_gains}")

    def plot_emg (self, channel_names=[], m_flags = False, h_flags = False, data_type='filtered'):
        """
        Plots EMG data from a Pickle file for a specified time window.

        Args:
            channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
        """

        # Handle custom channel names parameter if specified.
        customNames = False
        if len(channel_names) == 0:
            pass
        elif len(channel_names) != self.num_channels:
            print(f">! Error: list of custom channel names does not match the number of recorded channels. The entered list is {len(channel_names)} names long, but {self.num_channels} channels were recorded.")
        elif len(channel_names) == self.num_channels:
            customNames = True

        # Calculate time values based on the scan rate
        time_values_ms = np.arange(self.num_samples) * 1000 / self.scan_rate  # Time values in milliseconds

        # Determine the number of samples for the desired time window in ms
        num_samples_time_window = int(self.time_window_ms * self.scan_rate / 1000)  # Convert time window to number of samples

        # Slice the time array for the time window
        time_axis = time_values_ms[:num_samples_time_window] - self.stim_delay

        # Create a figure and axis
        if self.num_channels == 1:
            fig, ax = plt.subplots(figsize=(8, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(nrows=1, ncols=self.num_channels, figsize=(12, 4), sharey=True)

        # Establish type of EMG data to plot
        if data_type == 'filtered':
            emg_recordings = self.recordings_filtered
        elif data_type == 'raw':
            emg_recordings = self.recordings_raw
        elif data_type == 'rectified_raw':
            emg_recordings = self.recordings_rectified_raw
        elif data_type == 'rectified_filtered':
            emg_recordings = self.recordings_rectified_processed
        else:
            print(f">! Error: data type {data_type} is not supported. Please use 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered'.")
            return
        
        # Plot the EMG arrays for each channel, only for the first 10ms
        if customNames:
            for recording in emg_recordings:
                for channel_index, channel_data in enumerate(recording['channel_data']):
                    if self.num_channels == 1:
                        ax.plot(time_axis, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                        ax.set_title(f'{channel_names[0]}')
                        ax.grid(True)
                        #ax.legend()
                        if m_flags:
                            ax.axvline(self.m_start[channel_index], color=self.m_color, linestyle=self.flag_style)
                            ax.axvline(self.m_end[channel_index], color=self.m_color, linestyle=self.flag_style)                         
                        if h_flags:
                            ax.axvline(self.h_start[channel_index], color=self.h_color, linestyle=self.flag_style)
                            ax.axvline(self.h_end[channel_index], color=self.h_color, linestyle=self.flag_style)                       
                    else:
                        axes[channel_index].plot(time_axis, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                        axes[channel_index].set_title(f'{channel_names[channel_index]}')
                        axes[channel_index].grid(True)
                        #axes[channel_index].legend()
                        if m_flags:
                            axes[channel_index].axvline(self.m_start[channel_index], color=self.m_color, linestyle=self.flag_style)
                            axes[channel_index].axvline(self.m_end[channel_index], color=self.m_color, linestyle=self.flag_style)
                        if h_flags:
                            axes[channel_index].axvline(self.h_start[channel_index], color=self.h_color, linestyle=self.flag_style)
                            axes[channel_index].axvline(self.h_end[channel_index], color=self.h_color, linestyle=self.flag_style)
        else:
            for recording in self.recordings_filtered:
                for channel_index, channel_data in enumerate(recording['channel_data']):
                    if self.num_channels == 1:
                        ax.plot(time_axis, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                        ax.set_title('Channel 0')
                        ax.grid(True)
                        #ax.legend()
                        if m_flags:
                            ax.axvline(self.m_start[channel_index], color=self.m_color, linestyle=self.flag_style)
                            ax.axvline(self.m_end[channel_index], color=self.m_color, linestyle=self.flag_style)                         
                        if h_flags:
                            ax.axvline(self.h_start[channel_index], color=self.h_color, linestyle=self.flag_style)
                            ax.axvline(self.h_end[channel_index], color=self.h_color, linestyle=self.flag_style)  
                    else:
                        axes[channel_index].plot(time_axis, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                        axes[channel_index].set_title(f'Channel {channel_index}')
                        axes[channel_index].grid(True)
                        #axes[channel_index].legend()
                        if m_flags:
                            axes[channel_index].axvline(self.m_start[channel_index], color=self.m_color, linestyle=self.flag_style)
                            axes[channel_index].axvline(self.m_end[channel_index], color=self.m_color, linestyle=self.flag_style)
                        if h_flags:
                            axes[channel_index].axvline(self.h_start[channel_index], color=self.h_color, linestyle=self.flag_style)
                            axes[channel_index].axvline(self.h_end[channel_index], color=self.h_color, linestyle=self.flag_style)

        # Set labels and title
        if self.num_channels == 1:
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('EMG (mV)')
            fig.suptitle('EMG Overlay for Channel 0 (all recordings)')
        else:
            fig.suptitle('EMG Overlay for All Channels (all recordings)')
            fig.supxlabel('Time (ms)')
            fig.supylabel('EMG (mV)')

        # Adjust subplot spacing
        plt.subplots_adjust(**self.subplot_adjust_args)

        # Show the plot
        plt.show()

    def plot_emg_suspectedH (self, channel_names=[], h_threshold=0.3, plot_legend=False):
        """
        Detects session recordings with potential H-reflexes and plots them.

        Args:
            channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
            h_threshold (float, optional): Detection threshold of the average rectified EMG response in millivolts in the H-relfex window. Defaults to 0.3mV.
            plot_legend (bool, optional): Whether to plot legends. Defaults to False.
        """

        # Handle custom channel names parameter if specified.
        customNames = False
        if len(channel_names) == 0:
            pass
        elif len(channel_names) != self.num_channels:
            print(f">! Error: list of custom channel names does not match the number of recorded channels. The entered list is {len(channel_names)} names long, but {self.num_channels} channels were recorded.")
        elif len(channel_names) == self.num_channels:
            customNames = True

        # Calculate time values based on the scan rate
        time_values_ms = np.arange(self.num_samples) * 1000 / self.scan_rate  # Time values in milliseconds

        # Determine the number of samples for the first 10ms
        num_samples_time_window = int(self.time_window_ms * self.scan_rate / 1000)  # Convert time window to number of samples

        # Slice the time array for the time window
        time_axis = time_values_ms[:num_samples_time_window] - self.stim_delay

        # Create a figure and axis
        if self.num_channels == 1:
            fig, ax = plt.subplots(figsize=(8, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(nrows=1, ncols=self.num_channels, figsize=(12, 4), sharey=True)

        # Plot the EMG arrays for each channel, only for the first 10ms
        if customNames:
            for recording in self.recordings_filtered:
                for channel_index, channel_data in enumerate(recording['channel_data']):
                    h_window = channel_data[int(self.h_start[channel_index] * self.scan_rate / 1000):int(self.h_end[channel_index] * self.scan_rate / 1000)]
                    if max(h_window) - min(h_window) > h_threshold:  # Check amplitude variation within H-reflex window
                        if self.num_channels == 1:
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
        else:
            for recording in self.recordings_filtered:
                for channel_index, channel_data in enumerate(recording['channel_data']):
                    h_window = channel_data[int(self.h_start[channel_index] * self.scan_rate / 1000):int(self.h_end[channel_index] * self.scan_rate / 1000)]
                    if max(h_window) - min(h_window) > h_threshold:  # Check amplitude variation within 5-10ms window
                        if self.num_channels == 1:
                            ax.plot(time_axis, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                            ax.set_title('Channel 0')
                            ax.grid(True)
                            if plot_legend:
                                ax.legend()
                        else:
                            axes[channel_index].plot(time_axis, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                            axes[channel_index].set_title(f'Channel {channel_index}')
                            axes[channel_index].grid(True)
                            if plot_legend:
                                axes[channel_index].legend()

        # Set labels and title
        if self.num_channels == 1:
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('EMG (mV)')
            fig.suptitle(f'EMG Overlay for Channel 0 (H-reflex Amplitude Variability > {h_threshold} mV)')
        else:
            fig.suptitle(f'EMG Overlay for All Channels (H-reflex Amplitude Variability > {h_threshold} mV)')
            fig.supxlabel('Time (ms)')
            fig.supylabel('EMG (mV)')

        # Adjust subplot spacing
        plt.subplots_adjust(**self.subplot_adjust_args)

        # Show the plot
        plt.show()

    def plot_reflex_curves (self, channel_names=[], method='rms'):
        """
        Plots overlayed M-response and H-reflex curves for each recorded channel.

        Args:
            channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
        """

        # Handle custom channel names parameter if specified.
        customNames = False
        if len(channel_names) == 0:
            pass
        elif len(channel_names) != self.num_channels:
            print(f">! Error: list of custom channel names does not match the number of recorded channels. The entered list is {len(channel_names)} names long, but {self.num_channels} channels were recorded.")
        elif len(channel_names) == self.num_channels:
            customNames = True

        # Create a figure and axis
        if self.num_channels == 1:
            fig, ax = plt.subplots(figsize=(8, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(nrows=1, ncols=self.num_channels, figsize=(12, 4), sharey=True)

        # Plot the M-wave and H-response amplitudes for each channel
        for channel_index in range(self.num_channels):
            m_wave_amplitudes = []
            h_response_amplitudes = []
            stimulus_voltages = []

            for recording in self.recordings_filtered:
                channel_data = recording['channel_data'][channel_index]
                stimulus_v = recording['stimulus_v']
                
                if method == 'rms':
                    m_wave_amplitude = emg_transform.calculate_rms_amplitude(channel_data, self.m_start[channel_index] + self.stim_delay, self.m_end[channel_index] + self.stim_delay, self.scan_rate)
                    h_response_amplitude = emg_transform.calculate_rms_amplitude(channel_data, self.h_start[channel_index] + self.stim_delay, self.h_end[channel_index] + self.stim_delay, self.scan_rate)
                elif method == 'avg_rectified':
                    m_wave_amplitude = emg_transform.calculate_average_amplitude_rectified(channel_data, self.m_start[channel_index] + self.stim_delay, self.m_end[channel_index] + self.stim_delay, self.scan_rate)
                    h_response_amplitude = emg_transform.calculate_average_amplitude_rectified(channel_data, self.h_start[channel_index] + self.stim_delay, self.h_end[channel_index] + self.stim_delay, self.scan_rate)
                elif method == 'peak_to_trough':
                    m_wave_amplitude = emg_transform.calculate_peak_to_trough_amplitude(channel_data, self.m_start[channel_index] + self.stim_delay, self.m_end[channel_index] + self.stim_delay, self.scan_rate)
                    h_response_amplitude = emg_transform.calculate_peak_to_trough_amplitude(channel_data, self.h_start[channel_index] + self.stim_delay, self.h_end[channel_index] + self.stim_delay, self.scan_rate)
                else:
                    print(f">! Error: method {method} is not supported. Please use 'rms', 'avg_rectified', or 'peak_to_trough'.")
                    return

                m_wave_amplitudes.append(m_wave_amplitude)
                h_response_amplitudes.append(h_response_amplitude)
                stimulus_voltages.append(stimulus_v)

            if self.num_channels == 1:
                ax.scatter(stimulus_voltages, m_wave_amplitudes, color=self.m_color, label='M-wave', marker='o')
                ax.scatter(stimulus_voltages, h_response_amplitudes, color=self.h_color, label='H-response', marker='o')
                ax.set_title('Channel 0')
                #ax.set_xlabel('Stimulus Voltage (V)')
                #ax.set_ylabel('Amplitude (mV)')
                ax.grid(True)
                ax.legend()
                if customNames:
                    ax.set_title(f'{channel_names[0]}')
            else:
                axes[channel_index].scatter(stimulus_voltages, m_wave_amplitudes, color=self.m_color, label='M-wave', marker='o')
                axes[channel_index].scatter(stimulus_voltages, h_response_amplitudes, color=self.h_color, label='H-response', marker='o')
                axes[channel_index].set_title(f'Channel {channel_index}')
                #axes[channel_index].set_xlabel('Stimulus Voltage (V)')
                #axes[0].set_ylabel('Amplitude (mV)')
                axes[channel_index].grid(True)
                axes[channel_index].legend()
                if customNames:
                    axes[channel_index].set_title(f'{channel_names[channel_index]}')
        
        # Set labels and title
        if self.num_channels == 1:
            ax.set_xlabel('Stimulus Voltage (V)')
            ax.set_ylabel('Avg. Rect. Reflex EMG (mV)')
            fig.suptitle(f'M-response and H-reflex Curves')
            # Adjust subplot spacing
        else:
            fig.suptitle(f'M-response and H-reflex Curves')
            fig.supxlabel('Stimulus Voltage (V)')
            fig.supylabel('Avg. Rect. Reflex EMG (mV)')
        
        # Adjust subplot spacing
        plt.subplots_adjust(**self.subplot_adjust_args)

        # Show the plot
        plt.show()

class EMGDataset:
    """
    Class for a dataset of EMGSession instances for multi-session analyses and plotting.

    This module provides functions for analyzing a full dataset of EMGSessions. This code assumes all session have the same recording parameters and number of channels.
    The class must be instantiated with a list of EMGSession instances.

    Attributes:
        m_start_ms (float, optional): Start time of the M-response window in milliseconds. Defaults to 2.0 ms.
        m_end_ms (float, optional): End time of the M-response window in milliseconds. Defaults to 4.0 ms.
        h_start_ms (float, optional): Start time of the suspected H-reflex window in milliseconds. Defaults to 4.0 ms.
        h_end_ms (float, optional): End time of the suspected H-reflex window in milliseconds. Defaults to 7.0 ms.
        bin_size (float, optional): Bin size for the x-axis (in V). Defaults to 0.05V.
    """
    
    def __init__(self, emg_sessions):
        """
        Initialize an EMGDataset instance from a list of EMGSession instances for multi-session analyses and plotting.

        Args:
            emg_sessions (list): A list of instances of the class EMGSession, or a list of Pickle file locations that you want to use for the dataset.
        """
        self.emg_sessions = utils.unpackEMGSessions(emg_sessions) # Convert file location strings into a list of EMGSession instances.
        self.scan_rate = self.emg_sessions[0].scan_rate
        self.num_channels = self.emg_sessions[0].num_channels
        self.stim_delay = self.emg_sessions[0].stim_delay

        self.m_start = config['m_start']
        self.m_end = config['m_end']
        self.h_start = config['h_start']
        self.h_end = config['h_end']
        self.bin_size = config['bin_size']

        self.m_color = config['m_color']
        self.h_color = config['h_color']
        self.title_font_size = config['title_font_size']
        self.axis_label_font_size = config['axis_label_font_size']
        self.tick_font_size = config['tick_font_size']
        self.subplot_adjust_args = config['subplot_adjust_args']

        # Set plot font/style defaults for returned graphs
        plt.rcParams.update({'figure.titlesize': self.title_font_size})
        plt.rcParams.update({'figure.labelsize': self.axis_label_font_size, 'figure.labelweight': 'bold'})
        plt.rcParams.update({'axes.titlesize': self.axis_label_font_size, 'axes.titleweight': 'bold'})
        plt.rcParams.update({'xtick.labelsize': self.tick_font_size, 'ytick.labelsize': self.tick_font_size})

    def dataset_parameters(self):
        """
        Prints EMG dataset parameters.
        """
        print(f"# EMG Sessions: {len(self.emg_sessions)}")

    def plot_reflex_curves(self, channel_names=[], method='rms'):
        """
        Plots average M-response and H-reflex curves for a dataset of EMG sessions along with the standard deviation. 
        Because the stimulus voltages for subsequent trials/session vary slightly in their intensity, slight binning is required to plot a smooth curve.

        Args:
            channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
            method (string, optional): Method for calculating the amplitude of the M-wave and H-reflex. Options are 'rms', 'avg_rectified', or 'peak_to_trough'. Defaults to 'rms'.
        """
        # Handle custom channel names parameter if specified.
        customNames = False
        if len(channel_names) == 0:
            pass
        elif len(channel_names) != self.num_channels:
            print(f">! Error: list of custom channel names does not match the number of recorded channels. The entered list is {len(channel_names)} names long, but {self.num_channels} channels were recorded.")
        elif len(channel_names) == self.num_channels:
            customNames = True
        
        # Unpack session recordings.
        recordings = []
        for session in self.emg_sessions:
            recordings.extend(session.recordings_processed)
        sorted_recordings = sorted(recordings, key=lambda x: x['stimulus_v'])

        # Create a figure and axis
        if self.num_channels == 1:
            fig, ax = plt.subplots(figsize=(8, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(nrows=1, ncols=self.num_channels, figsize=(12, 4), sharey=True)

        # Get unique binned stimulus voltages
        stimulus_voltages = sorted(list(set([round(recording['stimulus_v'] / self.bin_size) * self.bin_size for recording in sorted_recordings])))

        # Plot the M-wave and H-response amplitudes for each channel
        for channel_index in range(self.num_channels):
            m_wave_means = []
            m_wave_stds = []
            h_response_means = []
            h_response_stds = []
            for stimulus_v in stimulus_voltages:
                m_wave_mean, m_wave_std, h_response_mean, h_response_std = emg_transform.calculate_mean_std(sorted_recordings, stimulus_v, channel_index, self.m_start[channel_index] + self.stim_delay, self.m_end[channel_index] + self.stim_delay, self.h_start[channel_index] + self.stim_delay, self.h_end[channel_index] + self.stim_delay, self.bin_size, self.scan_rate, method=method)
                m_wave_means.append(m_wave_mean)
                m_wave_stds.append(m_wave_std)
                h_response_means.append(h_response_mean)
                h_response_stds.append(h_response_std)

            if self.num_channels == 1:
                ax.plot(stimulus_voltages, m_wave_means, color=self.m_color, label='M-wave')
                ax.fill_between(stimulus_voltages, np.array(m_wave_means) - np.array(m_wave_stds), np.array(m_wave_means) + np.array(m_wave_stds), color='r', alpha=0.2)
                ax.plot(stimulus_voltages, h_response_means, color=self.h_color, label='H-response')
                ax.fill_between(stimulus_voltages, np.array(h_response_means) - np.array(h_response_stds), np.array(h_response_means) + np.array(h_response_stds), color='b', alpha=0.2)
                ax.set_title('Channel 0')
                if customNames:
                    ax.set_title(f'{channel_names[0]}')
                ax.grid(True)
                ax.legend()
            else:
                axes[channel_index].plot(stimulus_voltages, m_wave_means, color=self.m_color, label='M-wave')
                axes[channel_index].fill_between(stimulus_voltages, np.array(m_wave_means) - np.array(m_wave_stds), np.array(m_wave_means) + np.array(m_wave_stds), color='r', alpha=0.2)
                axes[channel_index].plot(stimulus_voltages, h_response_means, color=self.h_color, label='H-response')
                axes[channel_index].fill_between(stimulus_voltages, np.array(h_response_means) - np.array(h_response_stds), np.array(h_response_means) + np.array(h_response_stds), color='b', alpha=0.2)
                axes[channel_index].set_title(f'Channel {channel_index}' if not channel_names else channel_names[channel_index])
                if customNames:
                    axes[channel_index].set_title(f'{channel_names[channel_index]}')
                axes[channel_index].grid(True)
                axes[channel_index].legend()

        # Set labels and title
        if self.num_channels == 1:
            ax.set_xlabel('Stimulus Voltage (V)')
            ax.set_ylabel('Avg. Rect. Reflex EMG (mV)')
            fig.suptitle(f'M-response and H-reflex Curves')
        else:
            fig.suptitle(f'M-response and H-reflex Curves')
            fig.supxlabel('Stimulus Voltage (V)')
            fig.supylabel('Avg. Rect. Reflex EMG (mV)')

        # Adjust subplot spacing
        plt.subplots_adjust(**self.subplot_adjust_args)

        # Show the plot
        plt.show()
