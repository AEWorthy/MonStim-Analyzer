"""
Classes to analyze and plot EMG data from individual sessions or an entire dataset of sessions.
"""

import os
import sys
import pickle
import copy
import re
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
from dataclasses import dataclass

from PyQt6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget
import numpy as np
from matplotlib.lines import Line2D

from monstim_analysis.Plot_EMG import EMGSessionPlotter, EMGDatasetPlotter, EMGExperimentPlotter
import monstim_analysis.Transform_EMG as Transform_EMG
from monstim_utils import load_config, get_output_bin_path

# To do: Add a method to create dataset latency window objects for each session in the dataset. Make the default windows be the m-wave and h-reflex windows.

@dataclass
class LatencyWindow:
    name: str
    color: str
    start_times: List[float]
    durations: List[float]
    linestyle: str = '--'

    @property
    def end_times(self):
        return [start + dur for start, dur in zip(self.start_times, self.durations)]

    def plot(self, ax, channel_index):
        start_exists = end_exists = False
        
        for line in ax.lines:
            if isinstance(line, Line2D):
                if line.get_xdata()[0] == self.start_times[channel_index] and line.get_color() == self.color:
                    start_exists = True
                elif line.get_xdata()[0] == self.end_times[channel_index] and line.get_color() == self.color:
                    end_exists = True
                
                if start_exists and end_exists:
                    break
        
        if not start_exists:
            ax.axvline(self.start_times[channel_index], color=self.color, linestyle=self.linestyle)
        
        if not end_exists:
            ax.axvline(self.end_times[channel_index], color=self.color, linestyle=self.linestyle)

    def get_legend_element(self, stylized=True):
        if stylized:
            return Line2D([0], [0], color=self.color, linestyle=self.linestyle, label=self.name)
        else:
            return Line2D([0], [0], color=self.color, linestyle='-', label=self.name)
    
# Parent EMG data class. Mainly for loading config settings.
class EMGData:
    def __init__(self):
        self.latency_windows: List[LatencyWindow] = []
        _config = load_config()
        
        self.m_start : List[float] = _config['m_start']
        self.m_end : List[float] = [(time + _config['m_duration']) for time in _config['m_start']]
        self.h_start : List[float] = _config['h_start']
        self.h_end : List[float] = [(time + _config['h_duration']) for time in _config['h_start']]
        self.time_window_ms : float = _config['time_window']
        self.bin_size : float = _config['bin_size']

        self.latency_window_style : str = _config['latency_window_style']
        self.m_color : str = _config['m_color']
        self.h_color : str = _config['h_color']
        self.title_font_size : int = _config['title_font_size']
        self.axis_label_font_size : int = _config['axis_label_font_size']
        self.tick_font_size : int = _config['tick_font_size']
        
        self.subplot_adjust_args = _config['subplot_adjust_args']
        self.m_max_args = _config['m_max_args']
        self.butter_filter_args = _config['butter_filter_args']
        self.default_method : str = _config['default_method']
        self.default_channel_names : List[str] = _config['default_channel_names']
    
    def add_latency_window(self, name: str, color: str, start_times: List[float], durations: List[float], linestyle: str = '--'):
        """
        Add a new latency window.
        
        Args:
            name (str): Name of the latency window.
            color (str): Color of the latency window markers.
            start_times (list): List of start times for each channel.
            durations (list): List of durations for each channel.
            linestyle (str, optional): Line style for the markers. Defaults to '--'.
        """
        new_window = LatencyWindow(name, color, start_times, durations, linestyle)
        self.latency_windows.append(new_window)
            
    @staticmethod
    def unpackPickleOutput (output_path):
        """
        Unpacks a list of EMG session Pickle files and outputs a dictionary with k/v pairs of session names and the session Pickle location.

        Args:
            output_path (str): location of the output folder containing dataset directories/Pickle files.
        """
        expt_pickles_dict = {} #k=experiment_name, v=(dataset_pickles_dict, dataset_names)

        # get every experiment folder. Ignores a folder if the folder is named 'bin'
        expts = [file for file in os.listdir(output_path) if file != 'bin']

        for expt in expts:
            dataset_pickles_dict = {} #k=dataset_name, v=pickle_filepath(s)
            for dataset in os.listdir(os.path.join(output_path, expt)):
                if os.path.isdir(os.path.join(output_path, expt, dataset)):
                    pickles = os.listdir(os.path.join(output_path, expt, dataset))
                    pickle_paths = [os.path.join(output_path, expt, dataset, pickle).replace('\\', '/') for pickle in pickles]
                    dataset_pickles_dict[dataset] = pickle_paths
                else: # if this is a single session instead...
                    split_parts = dataset.split('-') # Split the string at the hyphens
                    session_name = '-'.join(split_parts[:-1]) # Select the portion before the last hyphen to drop the "-SessionData.pickle" portion.
                    dataset_pickles_dict[session_name] = os.path.join(output_path, expt, dataset).replace('\\', '/')
            # Add the dataset data to the experiment dictionary.
            dataset_names = list(dataset_pickles_dict.keys())
            expt_pickles_dict[expt] = (dataset_pickles_dict, dataset_names)
        return expt_pickles_dict

# Three main classes for EMG analysis. Children of the EMGData class.
class EMGSession(EMGData):
    """
    Class for analyzing and plotting data from a single recording session of variable channel numbers for within-session analyses and plotting.
    One session contains multiple recordings that will make up, for example, a single M-curve.

    This module provides functions for analyzing data stored in Pickle files from a single EMG recording session.
    It includes functions to extract session parameters, plot all EMG data, and plot EMG data from suspected H-reflex recordings.
    Class must be instantiated with the Pickled session data file.

    Attributes:
        - plotter (EMGSessionPlotter): An instance of the EMGSessionPlotter class for plotting EMG data. 
            - Types of plotting commands include: plot_emg, plot_susectedH, and plot_reflexCurves, and plot_mmax.
            - See the EMGSessionPlotter class in Plot_EMG.py for more details.
        - session_name (str): The name of the recording session.
        - num_channels (int): The number of channels in the EMG recordings.
        - scan_rate (int): The scan rate of the EMG recordings (Hz).
        - num_samples (int): The number of samples per channel in the EMG recordings.
        - stim_start (float): The stimulus delay from recording start (in ms) in the EMG recordings.
        - stim_duration (float): The stimulus duration (in ms) in the EMG recordings.
        - stim_interval (float): The stimulus interval (in s) in the EMG recordings.
        - emg_amp_gains (list): The amplifier gains for each channel in the EMG recordings.
        - recordings_raw (list): A list of dictionaries containing the raw EMG recordings.
        - recordings_processed (list): A list of dictionaries containing the processed EMG recordings.
        - m_max (list): A list of M-wave amplitudes for each channel in the session.

    Methods:
        load_session_data(pickled_data): Loads the session data from the pickle file.
        update_window_settings(): Opens a GUI to manually update the M-wave and H-reflex window settings for each channel.
        process_emg_data(apply_filter=False, rectify=False): Processes EMG data by applying a bandpass filter and/or rectifying the data.
        session_parameters(): Prints EMG recording session parameters from a Pickle file.
    """
    def __init__(self, pickled_data):
        """
        Initialize an EMGSession instance.

        Args:
            pickled_data (str): Filepath of the .pickle session data file for this session.
        """
        super().__init__()
        self.plotter = EMGSessionPlotter(self)
        self.load_session_data(pickled_data)
        self._recordings_processed = None
        self._m_max = None

        # Add M-wave latency window
        self.add_latency_window(
            name="M-wave",
            color="red",
            start_times=[1 for _ in range(self.num_channels)],  # start times for each channel
            durations=[2 for _ in range(self.num_channels)]  # end times for each channel
        )

        # Add H-reflex latency window
        self.add_latency_window(
            name="H-reflex",
            color="blue",
            start_times=[5 for _ in range(self.num_channels)],  # start times for each channel
            durations=[1 for _ in range(self.num_channels)]  # end times for each channel
        )

    def load_session_data(self, pickled_data):
        # Load the session data from the pickle file
        logging.info(f"Loading session data from {pickled_data}")
        with open(pickled_data, 'rb') as pickle_file:
            session_data = pickle.load(pickle_file)

        # Access session-wide information
        session_info : dict = session_data['session_info']
        self.session_id : str = session_info['session_id']
        self.num_channels : int = int(session_info['num_channels'])
        self.channel_names : List[str] = [self.default_channel_names[i] if i < len(self.default_channel_names) 
                                          else 'Channel ' + str(i) for i in range(self.num_channels)]
        self.scan_rate : int = int(session_info['scan_rate'])
        self.num_samples : int = int(session_info['num_samples'])
        self.pre_stim_acquired : float = session_info['pre_stim_acquired']
        self.post_stim_acquired : float = session_info['post_stim_acquired']
        self.stim_delay : float = session_info['stim_delay']
        self.stim_start : float = self.stim_delay + self.pre_stim_acquired
        self.stim_duration : float = session_info['stim_duration']
        self.stim_interval : float = session_info['stim_interval']
        self.emg_amp_gains : List[int] = [int(gain) for index, gain in enumerate(session_info['emg_amp_gains']) 
                                          if index < self.num_channels] # only include gains for the number of recorded channels.
        
        # Access the raw EMG recordings. Sort by stimulus voltage.
        self.recordings_raw = sorted(session_data['recordings'], key=lambda x: x['stimulus_v'])
    
    def _process_emg_data(self, apply_filter=False, rectify=False):
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
        def __process_single_recording(recording):
            for i, channel_emg in enumerate(recording['channel_data']):
                if apply_filter:
                    filtered_emg = Transform_EMG.butter_bandpass_filter(channel_emg, self.scan_rate, **self.butter_filter_args)
                    if rectify:
                        recording['channel_data'][i] = Transform_EMG.rectify_emg(filtered_emg)
                    else:
                        recording['channel_data'][i] = filtered_emg
                elif rectify:
                    rectified_emg = Transform_EMG.rectify_emg(channel_emg)
                    recording['channel_data'][i] = rectified_emg
                
                #!# I decided to remove the baseline correction code for now. It's best not to transform the data more than necessary.
                
                # # Code to apply baseline correction to the processed data if a filter was applied.
                # if apply_filter:
                #     recording['channel_data'] = EMG_Transformer.correct_emg_to_baseline(recording['channel_data'], self.scan_rate, self.stim_start)
            
            return recording
        
        # Copy recordings if deep copy is needed.
        processed_recordings = copy.deepcopy(self.recordings_raw) if apply_filter or rectify else self.recordings_raw

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            processed_recordings = list(executor.map(__process_single_recording, processed_recordings))

        return processed_recordings

    def rename_channels(self, new_names : dict[str]):
        """
        Renames a channel in the session.
    
        Args:
            new_names (dict[str]): A dictionary mapping old channel names to new channel names.
        """
        try:
            # Rename the channels in the session.
            for old_name, new_name in new_names.items():
                channel_idx = self.channel_names.index(old_name)
                self.channel_names[channel_idx] = new_name
        except IndexError:
            print("Error: The number of new names does not match the number of channels in the session.")
        except ValueError:
            print("Error: The channel name to be replaced does not exist in the session.")

    def apply_preferences(self):
        """
        Applies the preferences set in the config file to the dataset.
        """
        config = load_config()

        # Apply the preferences to the dataset.
        self.bin_size = config['bin_size']
        self.time_window_ms = config['time_window']
        self.butter_filter_args = config['butter_filter_args']
        self.default_method = config['default_method']
        self.m_max_args = config['m_max_args']
        self.default_channel_names = config['default_channel_names']

        self.latency_window_style = config['latency_window_style']
        self.m_color = config['m_color']
        self.h_color = config['h_color']
        self.title_font_size = config['title_font_size']
        self.axis_label_font_size = config['axis_label_font_size']
        self.tick_font_size = config['tick_font_size']
        self.subplot_adjust_args = config['subplot_adjust_args']

        # Re-create the plotter object with the new preferences.
        self.plotter = EMGSessionPlotter(self)
        for latency_window in self.latency_windows:
            latency_window.linestyle = config['latency_window_style']


    @property
    def recordings_processed (self):
        if self._recordings_processed is None:
            self._recordings_processed = self._process_emg_data(apply_filter=True, rectify=False)
        return self._recordings_processed

    @property
    def m_max(self):
        # uses default method to calculate m_max if not already calculated.
        if self._m_max is None:
            m_max = []
            
            for channel_idx in range(self.num_channels):
                stimulus_voltages = [recording['stimulus_v'] for recording in self.recordings_processed]
                m_wave_amplitudes = [Transform_EMG.calculate_emg_amplitude(recording['channel_data'][channel_idx], 
                                                                             self.m_start[channel_idx] + self.stim_start,
                                                                             self.m_end[channel_idx] + self.stim_start, 
                                                                             self.scan_rate, 
                                                                             method=self.default_method) 
                                                                             for recording in self.recordings_processed]                
                try: # Check if the channel has a valid M-max amplitude.
                    channel_mmax = Transform_EMG.get_avg_mmax(stimulus_voltages, m_wave_amplitudes, **self.m_max_args)
                    m_max.append(channel_mmax)
                except Transform_EMG.NoCalculableMmaxError:
                    logging.info(f"Channel {channel_idx} does not have a valid M-max amplitude.")
                    m_max.append(None)
            
            self._m_max = m_max
        return self._m_max

    # User methods for EMGSession class
    def m_max_report(self):
        """
        Prints the M-wave amplitudes for each channel in the session.
        """
        report = []
        for i, channel_m_max in enumerate(self.m_max):
            try:
                line = f"- {self.channel_names[i]}: M-max amplitude = {channel_m_max:.2f} V"
                report.append(line)
            except TypeError:
                line = f"- Channel {i} does not have a valid M-max amplitude."
                report.append(line)

        for line in report:
            print(line)
        return report

    def get_m_max(self, method, channel_index):
        """
        Calculates the M-wave amplitude for a specific channel in the session.

        Args:
            method (str): The method to use for calculating the M-wave amplitude. Options include 'average_rectified', 'rms', 'peak_to_trough', and 'average_unrectified'.
            channel_index (int): The index of the channel to calculate the M-wave amplitude for.

        Returns:
            float: The M-wave amplitude for the specified channel.
        """
        stimulus_voltages = [recording['stimulus_v'] for recording in self.recordings_processed]
        m_wave_amplitudes = [Transform_EMG.calculate_emg_amplitude(recording['channel_data'][channel_index], 
                                                                    self.m_start[channel_index] + self.stim_start,
                                                                    self.m_end[channel_index] + self.stim_start, 
                                                                    self.scan_rate, 
                                                                    method=method) 
                                                                    for recording in self.recordings_processed]
        return Transform_EMG.get_avg_mmax(stimulus_voltages, m_wave_amplitudes, **self.m_max_args)

    def update_window_settings(self):
        """
        Opens a GUI to manually update the M-wave and H-reflex window settings for each channel using PyQt.

        This function should only be used if you are working in a Jupyter notebook or an interactive Python environment. Do not call this function in any other GUI environment.
        """
        class ReflexSettings(QWidget):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
                self.initUI()

            def initUI(self):
                self.setWindowTitle(f"Update Reflex Window Settings: Session {self.parent.session_id}")
                layout = QVBoxLayout()

                duration_layout = QHBoxLayout()
                duration_layout.addWidget(QLabel("m_duration:"))
                self.m_duration_entry = QLineEdit(str(self.parent.m_end[0] - self.parent.m_start[0]))
                duration_layout.addWidget(self.m_duration_entry)

                duration_layout.addWidget(QLabel("h_duration:"))
                self.h_duration_entry = QLineEdit(str(self.parent.h_end[0] - self.parent.h_start[0]))
                duration_layout.addWidget(self.h_duration_entry)

                layout.addLayout(duration_layout)

                self.entries = []
                for i in range(self.parent.num_channels):
                    channel_layout = QHBoxLayout()
                    channel_layout.addWidget(QLabel(f"Channel {i}:"))

                    channel_layout.addWidget(QLabel("m_start:"))
                    m_start_entry = QLineEdit(str(self.parent.m_start[i]))
                    channel_layout.addWidget(m_start_entry)

                    channel_layout.addWidget(QLabel("h_start:"))
                    h_start_entry = QLineEdit(str(self.parent.h_start[i]))
                    channel_layout.addWidget(h_start_entry)

                    layout.addLayout(channel_layout)
                    self.entries.append((m_start_entry, h_start_entry))

                save_button = QPushButton("Confirm")
                save_button.clicked.connect(self.save_settings)
                layout.addWidget(save_button)

                self.setLayout(layout)

            def save_settings(self):
                try:
                    m_duration = float(self.m_duration_entry.text())
                    h_duration = float(self.h_duration_entry.text())
                except ValueError:
                    print("Invalid input for durations. Please enter valid numbers.")
                    return

                for i, (m_start_entry, h_start_entry) in enumerate(self.entries):
                    try:
                        m_start = float(m_start_entry.text())
                        h_start = float(h_start_entry.text())

                        self.parent.m_start[i] = m_start
                        self.parent.m_end[i] = m_start + m_duration
                        self.parent.h_start[i] = h_start
                        self.parent.h_end[i] = h_start + h_duration
                    except ValueError:
                        print(f"Invalid input for channel {i}. Skipping.")

                self.close()

        app = QApplication.instance()  # Check if there's an existing QApplication instance
        if not app:
            app = QApplication(sys.argv)
            window = ReflexSettings(self)
            window.show()
            app.exec()
        else:
            window = ReflexSettings(self)
            window.show()

    def session_parameters (self):
        """
        Prints EMG recording session parameters from a Pickle file.
        """
        report = [f"Session ID: {self.session_id}", 
                  f"# of Channels: {self.num_channels}",
                  f"Scan Rate (Hz): {self.scan_rate}",
                  f"Samples/Channel: {self.num_samples}",
                  f"Pre-Stim Acq. time (ms): {self.pre_stim_acquired}",
                  f"Post-Stim Acq. time (ms): {self.post_stim_acquired}",
                  f"Stimulus Delay (ms): {self.stim_delay}",
                  f"Stimulus Duration (ms): {self.stim_duration}",
                  f"Stimulus Interval (s): {self.stim_interval}",
                  f"EMG Amp Gains: {self.emg_amp_gains}"]        
        
        for line in report:
            print(line)
        return report

    def plot(self, plot_type: str = None, **kwargs):
        """
        Plots EMG data from a single session using the specified plot_type.

        Args:
            - plot_type (str): The type of plot to generate. Options include 'emg', 'suspectedH', 'mmax', and 'reflexCurves', and 'mCurvesSmoothened'.
                Plot types are defined in the EMGSessionPlotter class in Plot_EMG.py.
            - channel_names (list): A list of channel names to plot. If None, all channels will be plotted.
            - **kwargs: Additional keyword arguments to pass to the plotting function.
                
                The most common keyword arguments include:
                - 'data_type' (str): The type of data to plot. Options are 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered'. Default is 'filtered'.
                - 'method' (str): The method to use for calculating the M-wave/reflex amplitudes. Options include 'average_rectified', 'rms', 'peak_to_trough', and 'average_unrectified'. Default method is set in config.yml under 'default_method'.
                - 'relative_to_mmax' (bool): Whether to plot the data proportional to the M-wave amplitude (True) or as the actual recorded amplitude (False). Default is False.
                - 'all_flags' (bool): Whether to plot flags at all windows (True) or not (False). Default is False.
                
                Less common keyword arguments include:
                - 'm_flags' (bool): Whether to plot flags at the M-wave window (True) or not (False). Default is False.
                - 'h_flags' (bool): Whether to plot flags at the H-reflex window (True) or not (False). Default is False.
                - 'h_threshold' (float): The threshold for detecting the H-reflex in the suspectedH plot. Default is 0.3.
                - 'mmax_report' (bool): Whether to print the details of the M-max calculations (True) or not (False). Default is False.
                - 'manual_mmax' (float): The manually set M-wave amplitude to use for plotting the reflex curves. Default is None.

        Example Usages:
            # Plot filtered EMG data
                session.plot()

            # Plot raw EMG data with flags at the M-wave and H-reflex windows
            session.plot(data_type='raw', all_flags=True)

            # Plot all EMG data with the M-wave and H-reflex windows highlighted
            session.plot(plot_type='suspectedH')

            # Plot M-wave amplitudes for each channel
            session.plot(plot_type='mmax')

            # Plot the reflex curves for each channel
            session.plot(plot_type='reflexCurves')
        """

        # Call the appropriate plotting method from the plotter object
        getattr(self.plotter, f'plot_{"emg" if not plot_type else plot_type}')(**kwargs)

class EMGDataset(EMGData):
    """
    Class for a dataset of EMGSession instances for multi-session analyses and plotting.

    This module provides functions for analyzing a full dataset of EMGSessions. This code assumes all session have the same recording parameters and number of channels.
    The class must be instantiated with a list of EMGSession instances.

    Attributes:
        - plotter (EMGDatasetPlotter): An instance of the EMGDatasetPlotter class for plotting EMG data. 
            - Types of plotting commands include: plot_reflexCurves, plot_mmax, and plot_maxH.
            - See the EMGDatasetPlotter class in Plot_EMG.py for more details.
        - emg_sessions (list): A list of instances of the class EMGSession that make up the dataset.
        - date (str): The date of the dataset.
        - animal_id (str): The animal ID of the dataset.
        - condition (str): The experimental condition of the dataset.
        - scan_rate (int): The scan rate of the EMG recordings (Hz).
        - num_channels (int): The number of channels in the EMG recordings.
    """
    
    def __init__(self, emg_sessions, date, animal_id, condition, emg_sessions_to_exclude=[], save_path=None, temp=False):
        """
        Initialize an EMGDataset instance from a list of EMGSession instances for multi-session analyses and plotting.

        Args:
            emg_sessions (list): A list of instances of the class EMGSession, or a list of Pickle file locations that you want to use for the dataset.
            emg_sessions_to_exclude (list, optional): A list of session names to exclude from the dataset. Defaults to an empty list.
        """
        # Set dataset parameters
        self.date : str = date
        self.animal_id : str = animal_id
        self.condition : str = condition
        self.save_path = os.path.join(get_output_bin_path(),(save_path or f"{self.date}_{self.animal_id}_{self.condition}.pickle"))
        self._m_max = None

        if os.path.exists(self.save_path) and not temp:
            raise FileExistsError(self.save_path)
        else:
            logging.info(f"Creating new dataset: {self.date} {self.animal_id} {self.condition}.")
            super().__init__()
            self.plotter = EMGDatasetPlotter(self)    
            
            # Unpack the EMG sessions and exclude any sessions if needed.
            if isinstance(emg_sessions, str) or isinstance(emg_sessions, EMGSession):
                emg_sessions = [emg_sessions]
            self.original_emg_sessions = emg_sessions
            self.emg_sessions: List[EMGSession] = []
            logging.info(f"Unpacking EMG sessions: {emg_sessions}")
            self.emg_sessions = self.__unpackEMGSessions(emg_sessions) # Convert file location strings into a list of EMGSession instances.
            if len(emg_sessions_to_exclude) > 0:
                print(f"Excluding the following sessions from the dataset: {emg_sessions_to_exclude}")
                self.emg_sessions = [session for session in self.emg_sessions if session.session_id not in emg_sessions_to_exclude]
                self._num_sessions_excluded = len(emg_sessions) - len(self.emg_sessions)
            else:
                self._num_sessions_excluded = 0

            # Check that all sessions have the same parameters and set dataset parameters.
            consistent, message = self.__check_session_consistency()
            if not consistent:
                print(f"Error: {message}")
            else:
                self.scan_rate : int = self.emg_sessions[0].scan_rate
                self.num_channels : int = self.emg_sessions[0].num_channels
                self.stim_start : float = self.emg_sessions[0].stim_start
                self.channel_names : List[str] = self.emg_sessions[0].channel_names # not checked for consistency, but should be the same for all sessions.
                self.latency_windows : LatencyWindow = self.emg_sessions[0].latency_windows.copy()        
                if not temp:
                    self.save_dataset(self.save_path)          

    def dataset_parameters(self):
        """
        Prints EMG dataset parameters.
        """
        report = [f"EMG Sessions ({len(self.emg_sessions)}): {[session.session_id for session in self.emg_sessions]}.",
                    f"Date: {self.date}",
                    f"Animal ID: {self.animal_id}",
                    f"Condition: {self.condition}"]

        for line in report:
            print(line)
        return report

    def plot(self, plot_type: str = None, **kwargs):
        """
        Plots EMG data from a single session using the specified plot_type.

        Args:
            - plot_type (str): The type of plot to generate. Options include 'reflexCurves', 'mmax', and 'maxH'. Default is 'reflexCurves'.
                Plot types are defined in the EMGDatasetPlotter class in Plot_EMG.py.
            - channel_names (list): A list of channel names to plot. If None, all channels will be plotted.
            - **kwargs: Additional keyword arguments to pass to the plotting function.
                
                The most common keyword arguments include:
                - 'method' (str): The method to use for calculating the M-wave/reflex amplitudes. Options include 'average_rectified', 'rms', 'peak_to_trough', and 'average_unrectified'. Default method is set in config.yml under 'default_method'.
                - 'relative_to_mmax' (bool): Whether to plot the data proportional to the M-wave amplitude (True) or as the actual recorded amplitude (False). Default is False.
                - 'mmax_report' (bool): Whether to print the details of the M-max calculations (True) or not (False). Default is False.
                - 'manual_mmax' (float): The manually set M-wave amplitude to use for plotting the reflex curves. Default is None.

        Example Usages:
            # Plot the reflex curves for each channel.
            dataset.plot()

            # Plot M-wave amplitudes for each channel.
            dataset.plot(plot_type='mmax')

            # Plot the reflex curves for each channel.
            dataset.plot(plot_type='reflexCurves')

            # Plot the reflex curves for each channel and print the M-max details.
            dataset.plot(plot_type='reflexCurves', mmax_report=True)
        """

        # Call the appropriate plotting method from the plotter object
        getattr(self.plotter, f'plot_{"reflexCurves" if not plot_type else plot_type}')(**kwargs)

    def get_avg_m_max(self, method, channel_index):
        """
        Calculates the average M-wave amplitude for a specific channel in the dataset.

        Args:
            method (str): The method to use for calculating the M-wave amplitude. Options include 'average_rectified', 'rms', 'peak_to_trough', and 'average_unrectified'.
            channel_index (int): The index of the channel to calculate the M-wave amplitude for.

        Returns:
            float: The average M-wave amplitude for the specified channel.
        """
        m_wave_amplitudes = [session.get_m_max(method, channel_index) for session in self.emg_sessions if session.m_max[channel_index] is not None]
        return np.mean(m_wave_amplitudes)

    #Properties for the EMGDataset class.
    @property
    def name(self):
        return f"{self.date} {self.animal_id} {self.condition}"
    @property
    def m_max(self):
        if self._m_max is None:
            session_m_maxes = [session.m_max for session in self.emg_sessions]
            m_max = []
            # separate each channel's m_max into separate arrays and average them.
            for channel_index in range(self.num_channels):
                channel_m_maxes = [m_max[channel_index] for m_max in session_m_maxes if m_max[channel_index] is not None]
                if len(channel_m_maxes) == 0:
                    raise ValueError(f"Error: No valid M-max values found for channel {channel_index}.")
                channel_m_max = np.mean(channel_m_maxes)
                m_max.append(float(channel_m_max))
            self._m_max = m_max
            logging.info(f"Average M-max values created for dataset {self.name}: {self._m_max}")
        return self._m_max

    # User methods for manipulating the EMGSession instances in the dataset.
    def add_session(self, session : Union[EMGSession, str]):
        """
        Adds an EMGSession to the emg_sessions list of this EMGDataset.

        Parameters:
            session (EMGSession or str): The session to be added. It can be an instance of EMGSession or a file path to a pickled EMGSession.

        Raises:
            TypeError: If the session is neither an instance of EMGSession nor a valid file path to a pickled EMGSession.
        """
        if session in [session.session_id for session in self.emg_sessions]:
            logging.warn(f"Session {session.session_id} is already in the dataset. It will not be re-added.")
        else:
            if isinstance(session, EMGSession):
                # Add the session to the dataset.
                self.emg_sessions.append(session)
                self.emg_sessions = sorted(self.emg_sessions, key=lambda x: x.session_id)
                
            else:
                try:
                    self.emg_sessions.append(EMGSession(session))
                except:  # noqa: E722
                    raise TypeError("Expected an instance of EMGSession or a file path to a pickled EMGSession.")
            
            # Check that all sessions have the same parameters.
            self.__check_session_consistency()
            consistent, message = self.__check_session_consistency()
            if not consistent:
                print(f"Error: {message}")
            else:
                self.scan_rate = self.emg_sessions[0].scan_rate
                self.num_channels = self.emg_sessions[0].num_channels
                self.stim_start = self.emg_sessions[0].stim_start

    def remove_session(self, session_id : str):
        """
        Removes a session from the dataset.

        Args:
            session_id (str): The session_id of the session to be removed.
        """
        if session_id not in [session.session_id for session in self.emg_sessions]:
            print(f">! Error: session {session_id} not found in the dataset.")
        else:
            self.emg_sessions = [session for session in self.emg_sessions if session.session_id != session_id]
    
    def reload_dataset_sessions(self):
        """
        Reloads the dataset, adding any removed sessions back to the dataset.
        """
        fresh_temp_dataset = EMGDataset(self.original_emg_sessions, self.date, self.animal_id, self.condition, temp=True)
        self.emg_sessions = fresh_temp_dataset.emg_sessions
        channel_name_dict = {fresh_temp_dataset.channel_names[i]: self.channel_names[i] for i in range(self.num_channels)}
        self.rename_channels(channel_name_dict)
        self.set_reflex_settings(self.m_start, self.m_end[0]-self.m_start[0], self.h_start, self.h_end[0]-self.h_start[0])

    def get_session(self, session_idx: int) -> EMGSession:
        """
        Returns the EMGSession object at the specified index.
        """
        return self.emg_sessions[session_idx]

    def set_reflex_settings(self, m_start, m_duration, h_start, h_duration):
        """
        Overwrite the M-wave and H-reflex windows for all sessions in the dataset.

        Args:
            m_start (list): A list of M-wave window start times for each channel.
            m_duration (list): A list of M-wave window durations for each channel.
            h_start (list): A list of H-reflex window start times for each channel.
            h_duration (list): A list of H-reflex window durations for each channel.
        """
        for session in self.emg_sessions:
            session.m_start = m_start
            session.m_end = [(time + m_duration) for time in m_start]
            session.h_start = h_start
            session.h_end = [(time + h_duration) for time in h_start]
        self.m_start = m_start
        self.m_end = [(time + m_duration) for time in m_start]
        self.h_start = h_start
        self.h_end = [(time + h_duration) for time in h_start]

    def rename_channels(self, new_names : dict[str]):
        """
        Renames a channel in the dataset.

        Args:
            old_name (str): The current name of the channel.
            new_name (str): The new name to assign to the channel.
        """
        try:
            # Rename the channels in each session.
            for session in self.emg_sessions:
                session.rename_channels(new_names)
            # Rename the channels in the dataset.
            for old_name, new_name in new_names.items():
                try:
                    channel_idx = self.channel_names.index(old_name)
                    self.channel_names[channel_idx] = new_name
                except ValueError:
                    logging.info(f"The name '{old_name}' does not exist in the dataset. Channel names are still {self.channel_names}.")
        except IndexError:
            print("Error: The number of new names does not match the number of channels in the dataset.")
        
    def apply_preferences(self):
        """
        Applies the preferences set in the config file to the dataset.
        """
        config = load_config()
        # Apply the preferences to the dataset.
        self.bin_size = config['bin_size']
        self.time_window_ms = config['time_window']
        self.butter_filter_args = config['butter_filter_args']
        self.default_method = config['default_method']
        self.m_max_args = config['m_max_args']
        self.default_channel_names = config['default_channel_names']

        self.latency_window_style = config['latency_window_style']
        self.m_color = config['m_color']
        self.h_color = config['h_color']
        self.title_font_size = config['title_font_size']
        self.axis_label_font_size = config['axis_label_font_size']
        self.tick_font_size = config['tick_font_size']
        self.subplot_adjust_args = config['subplot_adjust_args']

        # Apply preferences to the session objects.
        for session in self.emg_sessions:
            session.apply_preferences()

        # Re-create the plotter object with the new preferences.
        self.plotter = EMGDatasetPlotter(self)
        for latency_window in self.latency_windows:
            latency_window.linestyle = config['latency_window_style']
        
    # Save and load the dataset object.
    def save_dataset(self, save_path=None):
        """
        Save the curated dataset object to disk.

        Args:
            save_path (str): The filename/save path to use for saving the dataset object.
        """
        if save_path is None:
            save_path = self.save_path
            
        with open(save_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_dataset(save_path):
        """
        Load a previously saved dataset object from disk.

        Args:
            save_path (str): The filename/save path of the saved dataset object.

        Returns:
            EMGDataset: The loaded dataset object.
        """
        with open(save_path, 'rb') as file:
            dataset = pickle.load(file) # type: EMGDataset
            dataset.apply_preferences()
        return dataset
    
    # Static methods for extracting information from dataset names and dataset dictionaries.
    @staticmethod
    def getDatasetInfo(dataset_name : str) -> tuple:
        """
        Extracts information from a dataset' directory name.

        Args:
            dataset_name (str): The name of the dataset in the format '[YYMMDD] [AnimalID] [Condition]'.

        Returns:
            tuple: A tuple containing the extracted information in the following order: (date, animal_id, condition).
                If the dataset name does not match the expected format, returns (None, None, None).
        """
        # Define the regex pattern
        pattern = r'^(\d{6})\s([A-Z0-9.]+)\s(.+)$'
        
        # Match the pattern
        match = re.match(pattern, dataset_name)
        
        if match:
            date = match.group(1)
            animal_id = match.group(2)
            condition = match.group(3)
            
            # Convert the date to "yyyy-mm-dd"
            date = datetime.strptime(date, '%y%m%d').strftime('%Y-%m-%d')
            
            return date, animal_id, condition
        else:
            logging.error(f"Error: Dataset name '{dataset_name}' does not match the expected format: '[YYMMDD] [AnimalID] [Condition]'.")
            return None, None, None

    @classmethod
    def dataset_from_dataset_dict(cls, dataset_dict: dict, dataset_idx: int, emg_sessions_to_exclude: List[str] = [], temp=True) -> 'EMGDataset':
        """
        Instantiates an EMGDataset from a dataset dictionary for downstream analysis.

        Args:
            dataset_dict (dict): A dictionary containing dataset information (keys = dataset names, values = dataset filepaths).
            datasets (list): A list of dataset names (keys).
            dataset_idx (int): The index of the dataset to be used.
            emg_sessions_to_exclude (list, optional): A list of EMG sessions to exclude. Defaults to an empty list.
            temp (bool, optional): Whether to create a temporary dataset object or save to the bin folder. Defaults to True.

        Returns:
            EMGDataset: The session of interest for downstream analysis.
        """
        try: 
            datasets = list(dataset_dict.keys())
            date, animal_id, condition = cls.getDatasetInfo(datasets[dataset_idx])
            # Future: Add a check to see if the dataset is already saved as a pickle file. If it is, load the pickle file instead of re-creating the dataset.
            dataset_oi = EMGDataset(dataset_dict[datasets[dataset_idx]], date, animal_id, condition, emg_sessions_to_exclude=emg_sessions_to_exclude, temp=temp)
            dataset_oi.apply_preferences()
            return dataset_oi
        except FileExistsError as e:
            save_path = str(e)
            dataset_oi = EMGDataset.load_dataset(save_path)
            return dataset_oi

    # Private methods for the EMGDataset class.
    def __unpackEMGSessions(self, emg_sessions):
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
            else:
                raise TypeError(f"An object in the 'emg_sessions' list was not properly converted to an EMGSession. Object: {session}, {type(session)}")
            
            pickled_sessions = sorted(pickled_sessions, key=lambda x: x.session_id)
        
        return pickled_sessions

    def __check_session_consistency(self):
        """
        Checks if all sessions in the dataset have the same parameters (scan rate, num_channels, stim_start).

        Returns:
            tuple: A tuple containing a boolean value indicating whether all sessions have consistent parameters and a message indicating the result.
        """
        reference_session = self.emg_sessions[0]
        reference_scan_rate = reference_session.scan_rate
        reference_num_channels = reference_session.num_channels
        reference_stim_start = reference_session.stim_start

        for session in self.emg_sessions[1:]:
            if session.scan_rate != reference_scan_rate:
                return False, f"Inconsistent scan_rate for {session.session_id}: {session.scan_rate} != {reference_scan_rate}."
            if session.num_channels != reference_num_channels:
                return False, f"Inconsistent num_channels for {session.session_id}: {session.num_channels} != {reference_num_channels}."
            if session.stim_start != reference_stim_start:
                return False, f"Inconsistent stim_start for {session.session_id}: {session.stim_start} != {reference_stim_start}."

        return True, "All sessions have consistent parameters"

class EMGExperiment(EMGData):
    def __init__(self, expt_name, expts_dict: List[str] = [], save_path: str = None, temp: bool = False):
            self.dataset_dict, self.dataset_names = expts_dict[expt_name]
            self.expt_name = expt_name

            # Create a list of EMGDataset instances from the dataset dictionary.
            emg_datasets = [EMGDataset.dataset_from_dataset_dict(self.dataset_dict, i, temp=temp) for i in range(len(self.dataset_names))] # Type: List[EMGDataset]
            self.save_path = os.path.join(get_output_bin_path(),(save_path or f"{self.expt_name}.pickle")) if save_path is None else save_path

            if os.path.exists(self.save_path) and not temp:
                raise FileExistsError(f"Experiment already exists at {self.save_path}.")
            else:
                logging.info(f"Creating new experiment: {self.expt_name}.")
                super().__init__()
                self.plotter = EMGExperimentPlotter(self)    
                
                # Unpack the EMG datasets and exclude any datasets if needed.
                if isinstance(emg_datasets, str) or isinstance(emg_datasets, EMGDataset):
                    emg_datasets = [emg_datasets]
                self.original_emg_datasets = emg_datasets
                self.emg_datasets: List[EMGDataset] = []
                logging.info(f"Unpacking EMG sessions: {emg_datasets}")
                self.emg_datasets = self.__unpackEMGDatasets(emg_datasets) # Convert file location strings into a list of EMGDataset instances.

                # Check that all sessions have the same parameters and set dataset parameters.
                consistent, message = True, ''
                # consistent, message = self.__check_dataset_consistency()
                if not consistent:
                    print(f"Error: {message}")
                else:
                    # self.scan_rate : int = self.emg_datasets[0].scan_rate
                    # self.num_channels : int = self.emg_datasets[0].num_channels
                    # self.stim_start : float = self.emg_datasets[0].stim_start
                    # self.channel_names : List[str] = self.emg_datasets[0].channel_names # not checked for consistency, but should be the same for all sessions.
                    self.latency_windows : List[LatencyWindow] = self.emg_datasets[0].latency_windows.copy()
                    if not temp:
                        self.save_experiment(self.save_path)

    def apply_preferences(self):
        """
        Applies the preferences set in the config file to the dataset.
        """
        config = load_config()
        # Apply the preferences to the dataset.
        self.bin_size = config['bin_size']
        self.time_window_ms = config['time_window']
        self.butter_filter_args = config['butter_filter_args']
        self.default_method = config['default_method']
        self.m_max_args = config['m_max_args']
        self.default_channel_names = config['default_channel_names']

        self.latency_window_style = config['latency_window_style']
        self.m_color = config['m_color']
        self.h_color = config['h_color']
        self.title_font_size = config['title_font_size']
        self.axis_label_font_size = config['axis_label_font_size']
        self.tick_font_size = config['tick_font_size']
        self.subplot_adjust_args = config['subplot_adjust_args']

        # Apply preferences to the session objects.
        for dataset in self.emg_datasets:
            dataset.apply_preferences()

        # Re-create the plotter object with the new preferences.
        self.plotter = EMGExperimentPlotter(self)
        for latency_window in self.latency_windows:
            latency_window.linestyle = config['latency_window_style']

    def save_experiment(self, save_path=None):
        """
        Save the curated dataset object to disk.

        Args:
            save_path (str): The filename/save path to use for saving the dataset object.
        """
        if save_path is None:
            save_path = self.save_path
            
        with open(save_path, 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load_experiment(save_path):
        """
        Load a previously saved experiment object from disk.

        Args:
            save_path (str): The filename/save path of the saved experiment object.

        Returns:
            EMGExperiment: The loaded experiment object.
        """
        with open(save_path, 'rb') as file:
            experiment = pickle.load(file) # type: EMGExperiment
            experiment.apply_preferences()
            return experiment
    
    @staticmethod
    def getExperimentInfo(experiment_name : str) -> tuple:
        pass

    @classmethod
    def experiment_from_experiment_dict(cls, experiment_dict: dict, experiment_idx: int, emg_datasets_to_exclude: List[str] = [], temp=True) -> 'EMGExperiment':
        """
        Instantiates an EMGExperiment from an experiment dictionary for downstream analysis.

        Args:
            experiment_dict (dict): A dictionary containing experiment information (keys = experiment names, values = experiment filepaths).
            experiment_idx (int): The index of the experiment to be used.
            emg_datasets_to_exclude (list, optional): A list of EMG datasets to exclude. Defaults to an empty list.
            temp (bool, optional): Whether to create a temporary experiment object or save to the bin folder. Defaults to True.

        Returns:
            EMGExperiment: The experiment of interest for downstream analysis.
        """
        experiments = list(experiment_dict.keys())
        expt_oi = EMGExperiment(experiment_dict[experiments[experiment_idx]], emg_datasets_to_exclude=emg_datasets_to_exclude, temp=temp)
        expt_oi.apply_preferences()
        return expt_oi

    def __unpackEMGDatasets(self, emg_datasets):
        """
        Unpacks a list of EMG dataset Pickle files and outputs a list of EMGDataset instances for those pickles.
        If a list of EMGDataset instances is passed, will return that same list.

        Args:
            emg_datasets (list): a list of instances of the class EMGDataset, or a list of Pickle file locations that you want to use for the dataset.

        Returns:
            list: a list of EMGDataset instances.

        Raises:
            TypeError: if an object in the 'emg_datasets' list was not properly converted to an EMGDataset.
        """
        # Check if list dtype is EMGDataset. If it is, convert it to a new EMGDataset instance and replace the string in the list.
        pickled_datasets = []
        for dataset in emg_datasets:
            if isinstance(dataset, str): # If list object is dtype(string), then convert to an EMGDataset.
                dataset = EMGDataset(dataset) # replace the string with an actual dataset object.
                pickled_datasets.append(dataset)
            elif isinstance(dataset, EMGDataset):
                pickled_datasets.append(dataset)
            else:
                raise TypeError(f"An object in the 'emg_datasets' list was not properly converted to an EMGDataset. Object: {dataset}, {type(dataset)}")
            
            pickled_datasets = sorted(pickled_datasets, key=lambda x: x.save_path)
        
        return pickled_datasets

    def __check_dataset_consistency(self):
        """
        Checks if all datasets in the experiment have the same parameters (scan rate, num_channels, stim_start).

        Returns:
            tuple: A tuple containing a boolean value indicating whether all datasets have consistent parameters and a message indicating the result.
        """
        reference_dataset = self.emg_datasets[0]
        reference_num_channels = reference_dataset.num_channels

        for dataset in self.emg_datasets[1:]:
            if dataset.num_channels != reference_num_channels:
                return False, f"Inconsistent num_channels for {dataset.name}: {dataset.num_channels} != {reference_num_channels}."

        return True, "All datasets have consistent parameters"


if __name__ == '__main__':
    from monstim_converter import pickle_data  # noqa: F401
    from monstim_analysis.Analyze_EMG import EMGData,EMGDataset
    from monstim_utils import DATA_PATH, OUTPUT_PATH, SAVED_DATASETS_PATH  # noqa: F401

    #Process CSVs into Pickle files: 'files_to_analyze' --> 'output'
    # pickle_data(DATA_PATH, OUTPUT_PATH) # If pickles are already created, comment this line out.

    # Create dictionaries of Pickle datasets and single sessions that are in the 'output' directory.
    dataset_dict, datasets = EMGData.unpackPickleOutput(OUTPUT_PATH)
    for idx, dataset in enumerate(datasets):
        print(f'dataset index {idx}: {dataset}')

    # Define dataset of interest for downstream analysis.
    sessions_to_exclude = [] # Add any sessions to exclude from the dataset here.
    dataset_idx = int(input('Dataset of Interest (index):')) # pick the dataset index of interest from the generate list above.
    dataset_oi = EMGDataset.dataset_from_dataset_dict(dataset_dict, dataset_idx, sessions_to_exclude)
    
    # Display dataset parameters.
    dataset_oi.dataset_parameters()

    # Define session of interest for downstream analysis.
    session_idx = int(input('Session of Interest (index):')) # pick the session index of interest from the generate list above
    session_oi = dataset_oi.get_session(session_idx)

    # Display session parameters.
    session_oi.session_parameters()

    # Visualize single EMG session raw and filtered
    session_oi.plot(plot_type = 'emg', m_flags=True, h_flags=True, data_type='filtered')
    # session_oi.plot(plot_type = 'emg', m_flags=True, h_flags=True, data_type='raw')
    session_oi.plot(plot_type = 'emg', m_flags=True, h_flags=True, data_type='rectified_filtered')
    # session_oi.plot(plot_type = 'emg', m_flags=True, h_flags=True, data_type='rectified_raw')

    # Use the update_window_settings method to temporarily change the reflex window settings and then replot. Otherwise comment out.
    # session_oi.update_window_settings()
    # session_oi.plot(plot_type = 'emg', m_flags=True, h_flags=True, data_type='filtered')

    # Inspect session reflex curves and suspected H-reflex trials with these methods.
    session_oi.plot(plot_type='reflexCurves')
    session_oi.plot(plot_type='reflexCurves', relative_to_mmax=True)

    # M-max analysis and visualization.
    session_oi.m_max_report()
    session_oi.plot(plot_type = 'mmax')

    # Visualize the entire dataset's avereaged reflex curves with these methods of the dataset object.
    dataset_oi.plot(plot_type = 'reflexCurves', relative_to_mmax=True, mmax_report=True)

    # Visualize the entire dataset's avereaged reflex values at H-max with this method of the dataset object.
    dataset_oi.plot(plot_type = 'maxH', relative_to_mmax=True)

    print('Done!')