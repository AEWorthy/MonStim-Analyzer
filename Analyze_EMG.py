
"""
Classes to analyze and plot EMG data from individual sessions or an entire dataset of sessions.
"""

import os
import pickle
import copy
from concurrent.futures import ThreadPoolExecutor

import yaml
import numpy as np
import matplotlib.pyplot as plt

import emg_transform as emg_transform



# Load configuration settings
def load_config(config_file):
    """
    Loads the config.yaml file into a YAML object that can be used to reference hard-coded configurable constants.

    Args:
        config_file (str): location of the 'config.yaml' file.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config('config.yml')



# Main classes for EMG analysis.
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
        self.m_max_args = config['m_max_args']

        self.butter_filter_args = config['butter_filter_args']

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
        self.recordings_raw = sorted(session_data['recordings'], key=lambda x: x['stimulus_v'])

    def process_emg_data(self, apply_filter=False, rectify=False):
        """
        Process EMG data by applying a bandpass filter and/or rectifying the data.

        Args:
            apply_filter (bool, optional): Whether to apply a bandpass filter to the data. Defaults to False.
            rectify (bool, optional): Whether to rectify the data. Defaults to False.

        Returns:
            list: List of processed recordings.

        Raises:
            None

        Examples:
            # Process EMG data without applying any filters or rectification
            processed_data = process_emg_data()

            # Process EMG data with bandpass filter applied
            processed_data = process_emg_data(apply_filter=True)

            # Process EMG data with rectification applied
            processed_data = process_emg_data(rectify=True)

            # Process EMG data with both bandpass filter and rectification applied
            processed_data = process_emg_data(apply_filter=True, rectify=True)
        """
        def process_single_recording(recording):
            for i, channel_emg in enumerate(recording['channel_data']):
                if apply_filter:
                    filtered_emg = emg_transform.butter_bandpass_filter(channel_emg, self.scan_rate, **self.butter_filter_args)
                    if rectify:
                        recording['channel_data'][i] = emg_transform.rectify_emg(filtered_emg)
                    else:
                        recording['channel_data'][i] = filtered_emg
                elif rectify:
                    rectified_emg = emg_transform.rectify_emg(channel_emg)
                    recording['channel_data'][i] = rectified_emg
                
                # Apply baseline correction to the processed data if a filter was applied.
                if apply_filter:
                    recording['channel_data'] = emg_transform.correct_emg_to_baseline(recording['channel_data'], self.scan_rate, self.stim_delay)
            return recording
        
        # Copy recordings if deep copy is needed.
        processed_recordings = copy.deepcopy(self.recordings_raw) if apply_filter or rectify else self.recordings_raw

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            processed_recordings = list(executor.map(process_single_recording, processed_recordings))

        return processed_recordings

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
            channel_names (list, optional): List of custom channel names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
            m_flags (bool, optional): Flag to indicate whether to plot markers for muscle onset and offset. Default is False.
            h_flags (bool, optional): Flag to indicate whether to plot markers for hand onset and offset. Default is False.
            data_type (str, optional): Type of EMG data to plot. Options are 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered'. Default is 'filtered'.

        Returns:
            None
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
            if not hasattr(self, 'recordings_processed'):
                self.recordings_processed = self.process_emg_data(apply_filter=True, rectify=False)
            emg_recordings = self.recordings_processed
        elif data_type == 'raw':
            emg_recordings = self.recordings_raw
        elif data_type == 'rectified_raw':
            if not hasattr(self, 'recordings_rectified_raw'):
                self.recordings_rectified_raw = self.process_emg_data(apply_filter=False, rectify=True)
            emg_recordings = self.recordings_rectified_raw
        elif data_type == 'rectified_filtered':
            if not hasattr(self, 'recordings_rectified_filtered'):
                self.recordings_rectified_filtered = self.process_emg_data(apply_filter=True, rectify=True)
            emg_recordings = self.recordings_rectified_filtered
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
            for recording in emg_recordings:
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

    def plot_reflex_curves (self, channel_names=[], method='rms', relative_to_mmax=False, manual_mmax=None):
        """
        Plots overlayed M-response and H-reflex curves for each recorded channel.

        Args:
            channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
            method (str, optional): The method used to calculate the mean and standard deviation. Options are 'rms', 'avg_rectified', or 'peak_to_trough'. Default is 'rms'.
        """
        # Process EMG data if not already done.
        if not hasattr(self, 'recordings_processed'):
            self.recordings_processed = self.process_emg_data(apply_filter=True, rectify=False)

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

            # Append the M-wave and H-response amplitudes for each recording into the superlist.
            for recording in self.recordings_processed:
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
            
            # Convert superlists to numpy arrays.
            m_wave_amplitudes = np.array(m_wave_amplitudes)
            h_response_amplitudes = np.array(h_response_amplitudes)
            stimulus_voltages = np.array(stimulus_voltages)

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            if relative_to_mmax:
                if manual_mmax is not None:
                    m_max = manual_mmax
                else:
                    m_max = emg_transform.get_avg_mmax(stimulus_voltages, m_wave_amplitudes, **self.m_max_args)
                m_wave_amplitudes = [amplitude / m_max for amplitude in m_wave_amplitudes]
                h_response_amplitudes = [amplitude / m_max for amplitude in h_response_amplitudes]    

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
        fig.suptitle(f'M-response and H-reflex Curves')
        if self.num_channels == 1:
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
        plt.subplots_adjust(**self.subplot_adjust_args)

        # Show the plot
        plt.show()

    def plot_m_curves_smoothened (self, channel_names=[], method='rms', relative_to_mmax=False, manual_mmax=None):
        """
        Plots overlayed M-response and H-reflex curves for each recorded channel.
        This plot is smoothened using a Savitzky-Golay filter, which therefore emulates the transformation used before calculating M-max in the EMG analysis.

        Args:
            channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
            method (str, optional): The method used to calculate the mean and standard deviation. Options are 'rms', 'avg_rectified', or 'peak_to_trough'. Default is 'rms'.
        """
        # Process EMG data if not already done.
        if not hasattr(self, 'recordings_processed'):
            self.recordings_processed = self.process_emg_data(apply_filter=True, rectify=False)

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

            # Append the M-wave and H-response amplitudes for each recording into the superlist.
            for recording in self.recordings_processed:
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
            
            # Convert superlists to numpy arrays.
            m_wave_amplitudes = np.array(m_wave_amplitudes)
            h_response_amplitudes = np.array(h_response_amplitudes)
            stimulus_voltages = np.array(stimulus_voltages)

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            if relative_to_mmax:
                if manual_mmax is not None:
                    m_max = manual_mmax
                else:
                    m_max = emg_transform.get_avg_mmax(stimulus_voltages, m_wave_amplitudes, **self.m_max_args)
                m_wave_amplitudes = [amplitude / m_max for amplitude in m_wave_amplitudes]
                h_response_amplitudes = [amplitude / m_max for amplitude in h_response_amplitudes]    

            # Smoothen the data
            m_wave_amplitudes = emg_transform.savgol_filter_y(m_wave_amplitudes)
            # h_response_amplitudes = np.gradient(m_wave_amplitudes, stimulus_voltages)

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
        fig.suptitle(f'M-response and H-reflex Curves')
        if self.num_channels == 1:
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
    
    def __init__(self, emg_sessions, emg_sessions_to_exclude=[]):
        """
        Initialize an EMGDataset instance from a list of EMGSession instances for multi-session analyses and plotting.

        Args:
            emg_sessions (list): A list of instances of the class EMGSession, or a list of Pickle file locations that you want to use for the dataset.
            emg_sessions_to_exclude (list, optional): A list of session names to exclude from the dataset. Defaults to an empty list.
        """
        self.emg_sessions = self.unpackEMGSessions(emg_sessions) # Convert file location strings into a list of EMGSession instances.
        if len(emg_sessions_to_exclude) > 0:
            print(f"Excluding the following sessions from the dataset: {emg_sessions_to_exclude}")
            self.emg_sessions = [session for session in self.emg_sessions if session.session_name not in emg_sessions_to_exclude]
            self.num_sessions_excluded = len(emg_sessions) - len(self.emg_sessions)
        else:
            self.num_sessions_excluded = 0
        
        # Generate processed recordings for each session if not already done.
        for session in self.emg_sessions:
            if not hasattr(session, 'recordings_processed'):
                session.recordings_processed = session.process_emg_data(apply_filter=True, rectify=False)

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
        self.m_max_args = config['m_max_args']

        # Set plot font/style defaults for returned graphs
        plt.rcParams.update({'figure.titlesize': self.title_font_size})
        plt.rcParams.update({'figure.labelsize': self.axis_label_font_size, 'figure.labelweight': 'bold'})
        plt.rcParams.update({'axes.titlesize': self.axis_label_font_size, 'axes.titleweight': 'bold'})
        plt.rcParams.update({'xtick.labelsize': self.tick_font_size, 'ytick.labelsize': self.tick_font_size})

    def unpackEMGSessions(self, emg_sessions):
        """
        Unpacks a list of EMG session Pickle files and outputs a list of EMGSession instances for those pickles.
        If a list of EMGSession instances is passed, will return that same list.

        Args:
            emg_sessions (list): a list of instances of the class EMGSession, or a list of Pickle file locations that you want to use for the dataset.

        Returns:
            list: a list of EMGSession instances.

        Raises:
            TypeError: if an object in the 'emg_sessions' list was not properly converted to an EMGSession.
        """
        # Check if list dtype is EMGSession. If it is, convert it to a new EMGSession instance and replace the string in the list.
        pickled_sessions = []
        for session in emg_sessions:
            if isinstance(session, str): # If list object is dtype(string), then convert to an EMGSession.
                session = EMGSession(session) # replace the string with an actual session object.
                pickled_sessions.append(session)
            elif isinstance(session, EMGSession):
                pickled_sessions.append(session)
                print(session)
            else:
                raise TypeError(f"An object in the 'emg_sessions' list was not properly converted to an EMGSession. Object: {session}, {type(session)}")

        return pickled_sessions

    def dataset_parameters(self):
        """
        Prints EMG dataset parameters.
        """
        print(f"# EMG Sessions: {len(self.emg_sessions)} of {len(self.emg_sessions) + self.num_sessions_excluded}.")

    def plot_reflex_curves(self, channel_names=[], method='rms', relative_to_mmax=False, manual_mmax=None):
        """
        Plots the M-response and H-reflex curves for each channel.

        Args:
            channel_names (list): A list of custom channel names. If specified, the channel names will be used in the plot titles.
            method (str): The method used to calculate the mean and standard deviation. Options are 'rms', 'avg_rectified', or 'peak_to_trough'. Default is 'rms'.

        Returns:
            None
        """

        # Handle custom channel names parameter if specified.
        customNames = False
        if len(channel_names) == 0:
            pass
        elif len(channel_names) != self.num_channels:
            print(f">! Error: list of custom channel names does not match the number of recorded channels. The entered list is {len(channel_names)} names long, but {self.num_channels} channels were recorded.")
        elif len(channel_names) == self.num_channels:
            customNames = True

        # Unpack processed session recordings.
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
                m_wave_amplitudes = []
                h_response_amplitudes = []
                
                # Append the M-wave and H-response amplitudes for the binned voltage into a list.
                for recording in recordings:
                    binned_stimulus_v = round(recording['stimulus_v'] / self.bin_size) * self.bin_size
                    if binned_stimulus_v == stimulus_v:
                        channel_data = recording['channel_data'][channel_index]
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
                    m_max = emg_transform.get_avg_mmax(stimulus_voltages, m_wave_means, **self.m_max_args)
                m_wave_means = [amplitude / m_max for amplitude in m_wave_means]
                m_wave_stds = [amplitude / m_max for amplitude in m_wave_stds]
                h_response_means = [amplitude / m_max for amplitude in h_response_means]
                h_response_stds = [amplitude / m_max for amplitude in h_response_stds]

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
        fig.suptitle(f'M-response and H-reflex Curves')
        if self.num_channels == 1:
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
        plt.subplots_adjust(**self.subplot_adjust_args)

        # Show the plot
        plt.show()

    def plot_max_h_reflex(self, channel_names=[], method='rms', relative_to_mmax=False, manual_mmax=None):
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
        # Handle custom channel names parameter if specified.
        customNames = False
        if len(channel_names) == 0:
            pass
        elif len(channel_names) != self.num_channels:
            print(f">! Error: list of custom channel names does not match the number of recorded channels. The entered list is {len(channel_names)} names long, but {self.num_channels} channels were recorded.")
        elif len(channel_names) == self.num_channels:
            customNames = True

        # Unpack processed session recordings.
        recordings = []
        for session in self.emg_sessions:
            recordings.extend(session.recordings_processed)
        sorted_recordings = sorted(recordings, key=lambda x: x['stimulus_v'])

        # Create a figure and axis
        if self.num_channels == 1:
            fig, ax = plt.subplots(figsize=(5, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(nrows=1, ncols=self.num_channels, figsize=(8, 4), sharey=True)

        # Get unique binned stimulus voltages
        stimulus_voltages = sorted(list(set([round(recording['stimulus_v'] / self.bin_size) * self.bin_size for recording in sorted_recordings])))

        for channel_index in range(self.num_channels):
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
                    binned_stimulus_v = round(recording['stimulus_v'] / self.bin_size) * self.bin_size
                    if binned_stimulus_v == stimulus_v:
                        channel_data = recording['channel_data'][channel_index]
                        if method == 'rms':
                            if relative_to_mmax:
                                m_wave_amplitude = emg_transform.calculate_rms_amplitude(channel_data, self.m_start[channel_index] + self.stim_delay, self.m_end[channel_index] + self.stim_delay, self.scan_rate)
                            h_response_amplitude = emg_transform.calculate_rms_amplitude(channel_data, self.h_start[channel_index] + self.stim_delay, self.h_end[channel_index] + self.stim_delay, self.scan_rate)
                        elif method == 'avg_rectified':
                            if relative_to_mmax:
                                m_wave_amplitude = emg_transform.calculate_average_amplitude_rectified(channel_data, self.m_start[channel_index] + self.stim_delay, self.m_end[channel_index] + self.stim_delay, self.scan_rate)
                            h_response_amplitude = emg_transform.calculate_average_amplitude_rectified(channel_data, self.h_start[channel_index] + self.stim_delay, self.h_end[channel_index] + self.stim_delay, self.scan_rate)
                        elif method == 'peak_to_trough':
                            if relative_to_mmax:
                                m_wave_amplitude = emg_transform.calculate_peak_to_trough_amplitude(channel_data, self.m_start[channel_index] + self.stim_delay, self.m_end[channel_index] + self.stim_delay, self.scan_rate)
                            h_response_amplitude = emg_transform.calculate_peak_to_trough_amplitude(channel_data, self.h_start[channel_index] + self.stim_delay, self.h_end[channel_index] + self.stim_delay, self.scan_rate)
                        else:
                            print(f">! Error: method {method} is not supported. Please use 'rms', 'avg_rectified', or 'peak_to_trough'.")
                            return

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
                binned_stimulus_v = round(recording['stimulus_v'] / self.bin_size) * self.bin_size
                if binned_stimulus_v == max_h_reflex_voltage:
                    channel_data = recording['channel_data'][channel_index]
                    if method == 'rms':
                        m_wave_amplitude_max_h = emg_transform.calculate_rms_amplitude(channel_data, self.m_start[channel_index] + self.stim_delay, self.m_end[channel_index] + self.stim_delay, self.scan_rate)
                        h_response_amplitude_max_h = emg_transform.calculate_rms_amplitude(channel_data, self.h_start[channel_index] + self.stim_delay, self.h_end[channel_index] + self.stim_delay, self.scan_rate)
                    elif method == 'avg_rectified':
                        m_wave_amplitude_max_h = emg_transform.calculate_average_amplitude_rectified(channel_data, self.m_start[channel_index] + self.stim_delay, self.m_end[channel_index] + self.stim_delay, self.scan_rate)
                        h_response_amplitude_max_h = emg_transform.calculate_average_amplitude_rectified(channel_data, self.h_start[channel_index] + self.stim_delay, self.h_end[channel_index] + self.stim_delay, self.scan_rate)
                    elif method == 'peak_to_trough':
                        m_wave_amplitude_max_h = emg_transform.calculate_peak_to_trough_amplitude(channel_data, self.m_start[channel_index] + self.stim_delay, self.m_end[channel_index] + self.stim_delay, self.scan_rate)
                        h_response_amplitude_max_h = emg_transform.calculate_peak_to_trough_amplitude(channel_data, self.h_start[channel_index] + self.stim_delay, self.h_end[channel_index] + self.stim_delay, self.scan_rate)
                    else:
                        print(f">! Error: method {method} is not supported. Please use 'rms', 'avg_rectified', or 'peak_to_trough'.")
                        return
                
                    m_wave_amplitudes_max_h.append(m_wave_amplitude_max_h)
                    h_response_amplitudes_max_h.append(h_response_amplitude_max_h)

            # Make the M-wave amplitudes relative to the maximum M-wave amplitude if specified.
            if relative_to_mmax:
                m_wave_means = np.array(m_wave_means)
                if manual_mmax is not None:
                    m_max = manual_mmax
                else:     
                    m_max = emg_transform.get_avg_mmax(stimulus_voltages, m_wave_means, **self.m_max_args)
                m_wave_amplitudes_max_h = [amplitude / m_max for amplitude in m_wave_amplitudes_max_h]
                h_response_amplitudes_max_h = [amplitude / m_max for amplitude in h_response_amplitudes_max_h]

            # Plot the M-wave and H-response amplitudes for the maximum H-reflex voltage.
            m_x = 1
            h_x = 2.5
            if self.num_channels == 1:
                ax.plot(m_x, [m_wave_amplitudes_max_h], color=self.m_color, marker='o', markersize=5)
                ax.annotate(f'n={len(m_wave_amplitudes_max_h)}', xy=(m_x + 0.4, np.mean(m_wave_amplitudes_max_h)), ha='center', color=self.m_color)
                ax.errorbar(m_x, np.mean(m_wave_amplitudes_max_h), yerr=np.std(m_wave_amplitudes_max_h), color='black', marker='+', markersize=10, capsize=10)
                
                ax.plot(h_x, [h_response_amplitudes_max_h], color=self.h_color, marker='o', markersize=5)
                ax.annotate(f'n={len(h_response_amplitudes_max_h)}', xy=(h_x + 0.4, np.mean(h_response_amplitudes_max_h)), ha='center', color=self.h_color)
                ax.errorbar(h_x, np.mean(h_response_amplitudes_max_h), yerr=np.std(h_response_amplitudes_max_h), color='black', marker='+', markersize=10, capsize=10)
                
                ax.set_xticks([m_x, h_x])
                ax.set_xticklabels(['M-response', 'H-reflex'])
                ax.set_title('Channel 0')
                if customNames:
                    ax.set_title(f'{channel_names[0]}')
                ax.set_xlim(m_x-1, h_x+1) # Set x-axis limits for each subplot to better center data points.
            else:
                axes[channel_index].plot(m_x, [m_wave_amplitudes_max_h], color=self.m_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(m_wave_amplitudes_max_h)}', xy=(m_x + 0.4, np.mean(m_wave_amplitudes_max_h)), ha='center', color=self.m_color)
                axes[channel_index].errorbar(m_x, np.mean(m_wave_amplitudes_max_h), yerr=np.std(m_wave_amplitudes_max_h), color='black', marker='+', markersize=10, capsize=10)

                axes[channel_index].plot(h_x, [h_response_amplitudes_max_h], color=self.h_color, marker='o', markersize=5)
                axes[channel_index].annotate(f'n={len(h_response_amplitudes_max_h)}', xy=(h_x + 0.4, np.mean(h_response_amplitudes_max_h)), ha='center', color=self.h_color)
                axes[channel_index].errorbar(h_x, np.mean(h_response_amplitudes_max_h), yerr=np.std(h_response_amplitudes_max_h), color='black', marker='+', markersize=10, capsize=10)
                
                axes[channel_index].set_title(f'Channel {channel_index}' if not channel_names else channel_names[channel_index])
                axes[channel_index].set_xticks([m_x, h_x])
                axes[channel_index].set_xticklabels(['M-response', 'H-reflex'])
                if customNames:
                    axes[channel_index].set_title(f'{channel_names[channel_index]} ({round(max_h_reflex_voltage + self.bin_size / 2, 2)} ± {self.bin_size/2}V)')
                axes[channel_index].set_xlim(m_x-1, h_x+1) # Set x-axis limits for each subplot to better center data points.
        
        # Set labels and title
        fig.suptitle(f'EMG Responses at Max H-reflex Stimulation')
        if self.num_channels == 1:
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
        plt.subplots_adjust(**self.subplot_adjust_args)

        # Show the plot
        plt.show()



# Helper functions for EMG analysis.
def unpackPickleOutput (output_path):
    """
    Unpacks a list of EMG session Pickle files and outputs a dictionary with k/v pairs of session names and the session Pickle location.

    Args:
        output_path (str): location of the output folder containing dataset directories/Pickle files.
    """
    dataset_pickles_dict = {} #k=datasets, v=pickle_filepath(s)

    for dataset in os.listdir(output_path):
        if os.path.isdir(os.path.join(output_path, dataset)):
            pickles = os.listdir(os.path.join(output_path, dataset))
            pickle_paths = [os.path.join(output_path, dataset, pickle).replace('\\', '/') for pickle in pickles]
            dataset_pickles_dict[dataset] = pickle_paths
        else: # if this is a single session instead...
            split_parts = dataset.split('-') # Split the string at the hyphens
            session_name = '-'.join(split_parts[:-1]) # Select the portion before the last hyphen to drop the "-SessionData.pickle" portion.
            dataset_pickles_dict[session_name] = os.path.join(output_path, dataset).replace('\\', '/')
    # Get dict keys
    dataset_dict_keys = list(dataset_pickles_dict.keys())
    return dataset_pickles_dict, dataset_dict_keys

def dataset_oi(dataset_dict, datasets, dataset_idx, emg_sessions_to_exclude=[]):
    """
    Defines a session of interest for downstream analysis.

    Args:
        dataset_dict (dict): A dictionary containing dataset information (keys = dataset names, values = dataset filepaths).
        datasets (list): A list of dataset names (keys).
        dataset_idx (int): The index of the dataset to be used.
        emg_sessions_to_exclude (list, optional): A list of EMG sessions to exclude. Defaults to an empty list.

    Returns:
        EMGDataset: The session of interest for downstream analysis.
    """
    dataset_oi = EMGDataset(dataset_dict[datasets[dataset_idx]], emg_sessions_to_exclude=emg_sessions_to_exclude)
    return dataset_oi

def session_oi (dataset_dict, datasets, dataset_idx, session_idx):
    """
    Defines a dataset of interest for downstream analysis.

    Args:
        output_path (str): location of the output folder containing dataset directories/Pickle files.
    """
    dataset_oi = dataset_dict[datasets[dataset_idx]]
    session_oi = EMGSession(dataset_oi[session_idx])
    return session_oi