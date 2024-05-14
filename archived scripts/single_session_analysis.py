# single_session_analysis.py

"""Module for analyzing data from a single session.

This module provides functions for analyzing data stored in Pickle files from a single EMG recording session.
It includes functions to extract session parameters, plot all EMG data, and plot EMG data from suspected H-reflex recordings.

Functions:
    - session_parameters(pickled_data): Extracts and prints session parameters from an EMG session Pickle file.
    - plot_EMG(pickled_data, time_window_ms=10): Plots EMG data from a Pickle file for a specified time window.
    - plot_EMG_suspectedH(pickled_data, h_start=5.5, h_end=10, h_threshold=0.3, plot_legend=False): Plots EMG data of detected H-reflexes.

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import scripts.emg_transform as emg_transform

def session_parameters (pickled_data):
    """Extracts and prints EMG recording session parameters from a Pickle file.

    Args:
        pickled_data (str): Path to the Pickle file containing session data and recordings.

    """

    # Load the session data from the pickle file
    with open(pickled_data, 'rb') as pickle_file:
        session_data = pickle.load(pickle_file)

    # Access session-wide information
    session_info = session_data['session_info']
    session_name = session_info['session_name']
    num_channels = session_info['num_channels']
    scan_rate = session_info['scan_rate']
    num_samples = session_info['num_samples']
    stim_duration = session_info['stim_duration']
    stim_interval = session_info['stim_interval']
    emg_amp_gains = session_info['emg_amp_gains']

    print(f"Session Name: {session_name}")
    print(f"# of Channels: {num_channels}")
    print(f"Scan rate (Hz): {scan_rate}")
    print(f"Samples/Channel: {num_samples}")
    print(f"Stimulus duration (ms): {stim_duration}")
    print(f"Stimulus interval (s): {stim_interval}")
    print(f"EMG amp gains: {emg_amp_gains}")

def plot_EMG (pickled_data, time_window_ms=10, channel_names=[]):
    """Plots EMG data from a Pickle file for a specified time window.

    Args:
        pickled_data (str): Path to the Pickle file containing session data.
        time_window_ms (float, optional): Time window to plot in milliseconds. Defaults to first 10ms.
        channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.

    """

    # Load the session data from the pickle file
    with open(pickled_data, 'rb') as pickle_file:
        session_data = pickle.load(pickle_file)

    # Access session-wide information
    session_info = session_data['session_info']
    scan_rate = session_info['scan_rate']
    num_samples = session_info['num_samples']
    num_channels = session_info['num_channels']

    # Handle custom channel names parameter if specified.
    customNames = False
    if len(channel_names) == 0:
        pass
    elif len(channel_names) != num_channels:
        print(f">! Error: list of custom channel names does not match the number of recorded channels. The entered list is {len(channel_names)} names long, but {num_channels} channels were recorded.")
    elif len(channel_names) == num_channels:
        customNames = True

    # Access recordings and sort by stimulus_value
    recordings = sorted(session_data['recordings'], key=lambda x: x['stimulus_v'])

    # Calculate time values based on the scan rate
    time_values_ms = np.arange(num_samples) * 1000 / scan_rate  # Time values in milliseconds

    # Determine the number of samples for the desired time window in ms
    num_samples_time_window = int(time_window_ms * scan_rate / 1000)  # Convert time window to number of samples

    # Slice the time array for the time window
    time_window_ms = time_values_ms[:num_samples_time_window]

    # Create a figure and axis
    if num_channels == 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        axes = [ax]
    else:
        fig, axes = plt.subplots(nrows=1, ncols=num_channels, figsize=(12, 4), sharey=True)

    # Plot the EMG arrays for each channel, only for the first 10ms
    if customNames:
        for recording in recordings:
            for channel_index, channel_data in enumerate(recording['channel_data']):
                if num_channels == 1:
                    ax.plot(time_window_ms, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                    ax.set_title(f'{channel_names[0]}')
                    ax.grid(True)
                    #ax.legend()
                else:
                    axes[channel_index].plot(time_window_ms, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                    axes[channel_index].set_title(f'{channel_names[channel_index]}')
                    axes[channel_index].grid(True)
                    #axes[channel_index].legend()
    else:
        for recording in recordings:
            for channel_index, channel_data in enumerate(recording['channel_data']):
                if num_channels == 1:
                    ax.plot(time_window_ms, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                    ax.set_title('Channel 0')
                    ax.grid(True)
                    #ax.legend()
                else:
                    axes[channel_index].plot(time_window_ms, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                    axes[channel_index].set_title(f'Channel {channel_index}')
                    axes[channel_index].grid(True)
                    #axes[channel_index].legend()

    # Set labels and title
    if num_channels == 1:
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('EMG (mV)')
        fig.suptitle('EMG Overlay for Channel 0 (all recordings)', fontsize=16)
    else:
        fig.suptitle('EMG Overlay for All Channels (all recordings)', fontsize=16)
        fig.supxlabel('Time (ms)')
        fig.supylabel('EMG (mV)')

        # Adjust subplot spacing
        plt.subplots_adjust(wspace=0.1,left=0.1, right=0.9, top=0.85, bottom=0.15)

    # Show the plot
    plt.show()

def plot_emg_rectified (pickled_data, time_window_ms=10, channel_names=[]):
    """Plots rectified EMG data from a Pickle file for a specified time window.

    Args:
        pickled_data (str): Path to the Pickle file containing session data.
        time_window_ms (float, optional): Time window to plot in milliseconds. Defaults to first 10ms.
        channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.

    """

    # Load the session data from the pickle file
    with open(pickled_data, 'rb') as pickle_file:
        session_data = pickle.load(pickle_file)

    # Access session-wide information
    session_info = session_data['session_info']
    scan_rate = session_info['scan_rate']
    num_samples = session_info['num_samples']
    num_channels = session_info['num_channels']

    # Handle custom channel names parameter if specified.
    customNames = False
    if len(channel_names) == 0:
        pass
    elif len(channel_names) != num_channels:
        print(f">! Error: list of custom channel names does not match the number of recorded channels. The entered list is {len(channel_names)} names long, but {num_channels} channels were recorded.")
    elif len(channel_names) == num_channels:
        customNames = True

    # Access recordings and sort by stimulus_value
    recordings = sorted(session_data['recordings'], key=lambda x: x['stimulus_v'])

    # Calculate time values based on the scan rate
    time_values_ms = np.arange(num_samples) * 1000 / scan_rate  # Time values in milliseconds

    # Determine the number of samples for the first 10ms
    time_window_ms = 10  # Time window in milliseconds
    num_samples_time_window = int(time_window_ms * scan_rate / 1000)  # Convert time window to number of samples

    # Slice the time array for the time window
    time_window_ms = time_values_ms[:num_samples_time_window]

    # Create a figure and axis
    if num_channels == 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        axes = [ax]
    else:
        fig, axes = plt.subplots(nrows=1, ncols=num_channels, figsize=(12, 4), sharey=True)

    # Plot the rectified EMG arrays for each channel, only for the first 10ms
    if customNames:
        for recording in recordings:
            for channel_index, channel_data in enumerate(recording['channel_data']):
                rectified_channel_data = emg_transform.rectify_emg(channel_data)
                if num_channels == 1:
                    ax.plot(time_window_ms, rectified_channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                    ax.set_title(f'{channel_names[0]} (Rectified)')
                    ax.grid(True)
                    #ax.legend()
                else:
                    axes[channel_index].plot(time_window_ms, rectified_channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                    axes[channel_index].set_title(f'{channel_names[channel_index]} (Rectified)')
                    axes[channel_index].grid(True)
                    #axes[channel_index].legend()
    else:
        for recording in recordings:
            for channel_index, channel_data in enumerate(recording['channel_data']):
                rectified_channel_data = emg_transform.rectify_emg(channel_data)
                if num_channels == 1:
                    ax.plot(time_window_ms, rectified_channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                    ax.set_title('Channel 0 (Rectified)')
                    ax.grid(True)
                    #ax.legend()
                else:
                    axes[channel_index].plot(time_window_ms, rectified_channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                    axes[channel_index].set_title(f'Channel {channel_index} (Rectified)')
                    axes[channel_index].grid(True)
                    #axes[channel_index].legend()

    # Set labels and title
    if num_channels == 1:
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Rectified EMG (mV)')
        fig.suptitle('Rectified EMG Overlay for Channel 0 (all recordings)', fontsize=16)
    else:
        fig.suptitle('Rectified EMG Overlay for All Channels (all recordings)', fontsize=16)
        fig.supxlabel('Time (ms)')
        fig.supylabel('Rectified EMG (mV)')

        # Adjust subplot spacing
        plt.subplots_adjust(wspace=0.1,left=0.1, right=0.9, top=0.85, bottom=0.15)
    # Show the plot
    plt.show()

def plot_EMG_suspectedH (pickled_data, time_window_ms=10, channel_names=[], h_start=5.5, h_end=10, h_threshold=0.3, plot_legend=False):
    """Detects session recordings with potential H-reflexes and plots them.

    Args:
        pickled_data (str): Path to the Pickle file containing session data.
        time_window_ms (float, optional): Time window to plot in milliseconds. Defaults to first 10ms.
        channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
        h_start (float, optional): Start time of the suspected H-reflex in milliseconds. Defaults to 5.5ms.
        h_end (float, optional): End time of the suspected H-reflex in milliseconds. Defaults to 10ms.
        h_threshold (float, optional): Detection threshold of the average rectified EMG response in millivolts in the H-relfex window. Defaults to 0.3mV.
        plot_legend (bool, optional): Whether to plot legends. Defaults to False.

    """
    
    # Plot possible H-reflex EMGs for all channels
    # h_start and h_end are the start and end of the H-relfex in milliseconds, and h_threshold is the threshold of the average rectified EMG response in millivolts.


    # Load the session data from the pickle file
    with open(pickled_data, 'rb') as pickle_file:
        session_data = pickle.load(pickle_file)

    # Access session-wide information
    session_info = session_data['session_info']
    scan_rate = session_info['scan_rate']
    num_samples = session_info['num_samples']
    num_channels = session_info['num_channels']

    # Handle custom channel names parameter if specified.
    customNames = False
    if len(channel_names) == 0:
        pass
    elif len(channel_names) != num_channels:
        print(f">! Error: list of custom channel names does not match the number of recorded channels. The entered list is {len(channel_names)} names long, but {num_channels} channels were recorded.")
    elif len(channel_names) == num_channels:
        customNames = True

    # Access recordings and sort by stimulus_value
    recordings = sorted(session_data['recordings'], key=lambda x: x['stimulus_v'])

    # Calculate time values based on the scan rate
    time_values_ms = np.arange(num_samples) * 1000 / scan_rate  # Time values in milliseconds

    # Determine the number of samples for the first 10ms
    time_window_ms = 10  # Time window in milliseconds
    num_samples_time_window = int(time_window_ms * scan_rate / 1000)  # Convert time window to number of samples

    # Slice the time array for the time window
    time_window_ms = time_values_ms[:num_samples_time_window]

    # Create a figure and axis
    if num_channels == 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        axes = [ax]
    else:
        fig, axes = plt.subplots(nrows=1, ncols=num_channels, figsize=(12, 4), sharey=True)

    # Plot the EMG arrays for each channel, only for the first 10ms
    if customNames:
        for recording in recordings:
            for channel_index, channel_data in enumerate(recording['channel_data']):
                h_window = recording['channel_data'][channel_index][int(h_start * scan_rate / 1000):int(h_end * scan_rate / 1000)]
                if max(h_window) - min(h_window) > h_threshold:  # Check amplitude variation within 5-10ms window
                    if num_channels == 1:
                        ax.plot(time_window_ms, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                        ax.set_title(f'{channel_names[0]}')
                        ax.grid(True)
                        if plot_legend:
                            ax.legend()
                    else:
                        axes[channel_index].plot(time_window_ms, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                        axes[channel_index].set_title(f'{channel_names[channel_index]}')
                        axes[channel_index].grid(True)
                        if plot_legend:
                            axes[channel_index].legend()
    else:
        for recording in recordings:
            for channel_index, channel_data in enumerate(recording['channel_data']):
                h_window = recording['channel_data'][channel_index][int(h_start * scan_rate / 1000):int(h_end * scan_rate / 1000)]
                if max(h_window) - min(h_window) > h_threshold:  # Check amplitude variation within 5-10ms window
                    if num_channels == 1:
                        ax.plot(time_window_ms, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                        ax.set_title('Channel 0')
                        ax.grid(True)
                        if plot_legend:
                            ax.legend()
                    else:
                        axes[channel_index].plot(time_window_ms, channel_data[:num_samples_time_window], label=f"Stimulus Voltage: {recording['stimulus_v']}")
                        axes[channel_index].set_title(f'Channel {channel_index}')
                        axes[channel_index].grid(True)
                        if plot_legend:
                            axes[channel_index].legend()

    # Set labels and title
    if num_channels == 1:
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('EMG (mV)')
        fig.suptitle(f'EMG Overlay for Channel 0 (H-reflex Amplitude Variability > {h_threshold} mV)', fontsize=16)
    else:
        fig.suptitle(f'EMG Overlay for All Channels (H-reflex Amplitude Variability > {h_threshold} mV)', fontsize=16)
        fig.supxlabel('Time (ms)')
        fig.supylabel('EMG (mV)')

        # Adjust subplot spacing
        plt.subplots_adjust(wspace=0.1,left=0.1, right=0.9, top=0.85, bottom=0.15)

    # Show the plot
    plt.show()

def plot_reflex_curves (pickled_data, channel_names=[], m_start_ms=2.0, m_end_ms=4.0, h_start_ms=4.0, h_end_ms=7.0):
    """Plots overlayed M-response and H-reflex curves for each recorded channel.

    Args:
        pickled_data (str): Path to the Pickle file containing session data.
        channel_names (string, optional): List of custom channels names to be plotted. Must be the exact same length as the number of recorded channels in the dataset.
        m_start_ms (float, optional): Start time of the M-response window in milliseconds. Defaults to 2.0 ms.
        m_end_ms (float, optional): End time of the M-response window in milliseconds. Defaults to 4.0 ms.
        h_start_ms (float, optional): Start time of the suspected H-reflex window in milliseconds. Defaults to 4.0 ms.
        h_end_ms (float, optional): End time of the suspected H-reflex window in milliseconds. Defaults to 7.0 ms.
    """
    # Load the session data from the pickle file
    with open(pickled_data, 'rb') as pickle_file:
        session_data = pickle.load(pickle_file)

    # Access session-wide information
    session_info = session_data['session_info']
    scan_rate = session_info['scan_rate']
    num_channels = session_info['num_channels']

    # Handle custom channel names parameter if specified.
    customNames = False
    if len(channel_names) == 0:
        pass
    elif len(channel_names) != num_channels:
        print(f">! Error: list of custom channel names does not match the number of recorded channels. The entered list is {len(channel_names)} names long, but {num_channels} channels were recorded.")
    elif len(channel_names) == num_channels:
        customNames = True

    # Access recordings and sort by stimulus_value
    recordings = sorted(session_data['recordings'], key=lambda x: x['stimulus_v'])

    # Create a figure and axis
    if num_channels == 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        axes = [ax]
    else:
        fig, axes = plt.subplots(nrows=1, ncols=num_channels, figsize=(12, 4), sharey=True)

    # Plot the M-wave and H-response amplitudes for each channel
    for channel_index in range(num_channels):
        m_wave_amplitudes = []
        h_response_amplitudes = []
        stimulus_voltages = []

        for recording in recordings:
            channel_data = recording['channel_data'][channel_index]
            stimulus_v = recording['stimulus_v']

            m_wave_amplitude = emg_transform.calculate_average_amplitude(channel_data, m_start_ms, m_end_ms, scan_rate)
            h_response_amplitude = emg_transform.calculate_average_amplitude(channel_data, h_start_ms, h_end_ms, scan_rate)

            m_wave_amplitudes.append(m_wave_amplitude)
            h_response_amplitudes.append(h_response_amplitude)
            stimulus_voltages.append(stimulus_v)

        if num_channels == 1:
            ax.scatter(stimulus_voltages, m_wave_amplitudes, color='r', label='M-wave', marker='o')
            ax.scatter(stimulus_voltages, h_response_amplitudes, color='b', label='H-response', marker='o')
            ax.set_title('Channel 0')
            #ax.set_xlabel('Stimulus Voltage (V)')
            #ax.set_ylabel('Amplitude (mV)')
            ax.grid(True)
            ax.legend()
            if customNames:
                ax.set_title(f'{channel_names[0]}')
        else:
            axes[channel_index].scatter(stimulus_voltages, m_wave_amplitudes, color='r', label='M-wave', marker='o')
            axes[channel_index].scatter(stimulus_voltages, h_response_amplitudes, color='b', label='H-response', marker='o')
            axes[channel_index].set_title(f'Channel {channel_index}')
            #axes[channel_index].set_xlabel('Stimulus Voltage (V)')
            #axes[0].set_ylabel('Amplitude (mV)')
            axes[channel_index].grid(True)
            axes[channel_index].legend()
            if customNames:
                axes[channel_index].set_title(f'{channel_names[channel_index]}')
    
    # Set labels and title
    if num_channels == 1:
        ax.set_xlabel('Stimulus Voltage (V)')
        ax.set_ylabel('EMG Amplitude (mV)')
        fig.suptitle(f'M-response and H-reflex Curves', fontsize=16)
    else:
        fig.suptitle(f'M-response and H-reflex Curves', fontsize=16)
        fig.supxlabel('Stimulus Voltage (V)')
        fig.supylabel('EMG Amplitude (mV)')

        # Adjust subplot spacing
        plt.subplots_adjust(wspace=0.1,left=0.1, right=0.9, top=0.85, bottom=0.15)

    # Show the plot
    plt.show()