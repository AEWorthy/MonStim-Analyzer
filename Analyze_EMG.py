"""
Classes to analyze and plot EMG data from individual sessions or an entire dataset of sessions.
"""

import os
import pickle
import copy
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import yaml
import tkinter as tk
from tkinter import ttk

from Plot_EMG import EMGSessionPlotter, EMGDatasetPlotter
import EMG_Transformer


# Parent EMG data class. Mainly for loading config settings.
class EMGData:
    def __init__(self, config_file='config.yml'):
        _config = self.load_config(config_file)

        self.m_start = _config['m_start']
        self.m_end = [(time + _config['m_duration']) for time in _config['m_start']]
        self.h_start = _config['h_start']
        self.h_end = [(time + _config['h_duration']) for time in _config['h_start']]
        self.time_window_ms = _config['time_window']
        self.bin_size = _config['bin_size']

        self.flag_style = _config['flag_style']
        self.m_color = _config['m_color']
        self.h_color = _config['h_color']
        self.title_font_size = _config['title_font_size']
        self.axis_label_font_size = _config['axis_label_font_size']
        self.tick_font_size = _config['tick_font_size']
        self.subplot_adjust_args = _config['subplot_adjust_args']
        self.m_max_args = _config['m_max_args']

        self.butter_filter_args = _config['butter_filter_args']
        self.default_method = _config['default_method']
    
    def load_config(self, config_file):
        """
        Loads the config.yaml file into a YAML object that can be used to reference hard-coded configurable constants.

        Args:
            config_file (str): location of the 'config.yaml' file.
        """
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    @staticmethod
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
        - stim_delay (float): The stimulus delay from recording start (in ms) in the EMG recordings.
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
                    filtered_emg = EMG_Transformer.butter_bandpass_filter(channel_emg, self.scan_rate, **self.butter_filter_args)
                    if rectify:
                        recording['channel_data'][i] = EMG_Transformer.rectify_emg(filtered_emg)
                    else:
                        recording['channel_data'][i] = filtered_emg
                elif rectify:
                    rectified_emg = EMG_Transformer.rectify_emg(channel_emg)
                    recording['channel_data'][i] = rectified_emg
                
                #!# I decided to remove the baseline correction code for now. It's best not to transform the data more than necessary.
                
                # # Code to apply baseline correction to the processed data if a filter was applied.
                # if apply_filter:
                #     recording['channel_data'] = EMG_Transformer.correct_emg_to_baseline(recording['channel_data'], self.scan_rate, self.stim_delay)
            
            return recording
        
        # Copy recordings if deep copy is needed.
        processed_recordings = copy.deepcopy(self.recordings_raw) if apply_filter or rectify else self.recordings_raw

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            processed_recordings = list(executor.map(__process_single_recording, processed_recordings))

        return processed_recordings

    @property
    def recordings_processed (self):
        if self._recordings_processed is None:
            self._recordings_processed = self._process_emg_data(apply_filter=True, rectify=False)
        return self._recordings_processed

    @property
    def m_max(self):
        if self._m_max is None:
            m_max = []
            
            for channel_idx in range(self.num_channels):
                stimulus_voltages = [recording['stimulus_v'] for recording in self.recordings_processed]
                m_wave_amplitudes = [EMG_Transformer.calculate_emg_amplitude(recording['channel_data'][channel_idx], self.m_start[channel_idx], self.m_end[channel_idx], self.scan_rate, method=self.default_method) for recording in self.recordings_processed]                
                
                channel_mmax = EMG_Transformer.get_avg_mmax(stimulus_voltages, m_wave_amplitudes, mmax_report=False, **self.m_max_args)
                m_max.append(channel_mmax)
            
            self._m_max = m_max
        return self._m_max

    # User methods for EMGSession class
    def m_max_report(self, channel_names: Optional[List[str]] = None):
        """
        Prints the M-wave amplitudes for each channel in the session.
        """
        if channel_names is None:
            for i, channel_m_max in enumerate(self.m_max):
                print(f"Channel {i}: M-max amplitude = {channel_m_max:.2f} V")
        else:
            for i, channel_m_max in enumerate(self.m_max):
                print(f"{channel_names[i]}: M-max amplitude = {channel_m_max:.2f} V")

    def update_window_settings(self):
        """
        Opens a GUI to manually update the M-wave and H-reflex window settings for each channel.
        """
        def save_settings():
            for i, (m_start_entry, m_end_entry, h_start_entry, h_end_entry) in enumerate(entry_fields):
                try:
                    self.m_start[i] = float(m_start_entry.get())
                    self.m_end[i] = float(m_end_entry.get())
                    self.h_start[i] = float(h_start_entry.get())
                    self.h_end[i] = float(h_end_entry.get())
                except ValueError:
                    print(f"Invalid input for channel {i}. Skipping.")
            window.destroy()

        window = tk.Tk()
        window.title(f"Update Reflex Window Settings: Session {self.session_name}")

        frame = ttk.Frame(window, padding=10)
        frame.grid()

        entry_fields = []
        for i in range(self.num_channels):
            channel_label = ttk.Label(frame, text=f"Channel {i}:")
            channel_label.grid(row=i, column=0, sticky=tk.W)

            m_start_label = ttk.Label(frame, text="m_start:")
            m_start_label.grid(row=i, column=1, sticky=tk.W)
            m_start_entry = ttk.Entry(frame)
            m_start_entry.insert(0, str(self.m_start[i]))
            m_start_entry.grid(row=i, column=2)

            m_end_label = ttk.Label(frame, text="m_end:")
            m_end_label.grid(row=i, column=3, sticky=tk.W)
            m_end_entry = ttk.Entry(frame)
            m_end_entry.insert(0, str(self.m_end[i]))
            m_end_entry.grid(row=i, column=4)

            h_start_label = ttk.Label(frame, text="h_start:")
            h_start_label.grid(row=i, column=5, sticky=tk.W)
            h_start_entry = ttk.Entry(frame)
            h_start_entry.insert(0, str(self.h_start[i]))
            h_start_entry.grid(row=i, column=6)

            h_end_label = ttk.Label(frame, text="h_end:")
            h_end_label.grid(row=i, column=7, sticky=tk.W)
            h_end_entry = ttk.Entry(frame)
            h_end_entry.insert(0, str(self.h_end[i]))
            h_end_entry.grid(row=i, column=8)

            entry_fields.append((m_start_entry, m_end_entry, h_start_entry, h_end_entry))

        save_button = ttk.Button(frame, text="Confirm", command=save_settings)
        save_button.grid(row=self.num_channels, columnspan=9)

        window.mainloop()

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

    def plot(self, plot_type: str = None, channel_names: Optional[List[str]] = None, **kwargs):
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
        getattr(self.plotter, f'plot_{'emg' if not plot_type else plot_type}')(channel_names=channel_names, **kwargs)

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
        - stim_delay (float): The stimulus delay from recording start (in ms) in the EMG recordings.
    """
    
    def __init__(self, emg_sessions, date, animal_id, condition, emg_sessions_to_exclude=[]):
        """
        Initialize an EMGDataset instance from a list of EMGSession instances for multi-session analyses and plotting.

        Args:
            emg_sessions (list): A list of instances of the class EMGSession, or a list of Pickle file locations that you want to use for the dataset.
            emg_sessions_to_exclude (list, optional): A list of session names to exclude from the dataset. Defaults to an empty list.
        """
        super().__init__()
        self.plotter = EMGDatasetPlotter(self)
        
        # Unpack the EMG sessions and exclude any sessions if needed.
        self.emg_sessions: List[EMGSession] = []
        self.emg_sessions = self.__unpackEMGSessions(emg_sessions) # Convert file location strings into a list of EMGSession instances.
        if len(emg_sessions_to_exclude) > 0:
            print(f"Excluding the following sessions from the dataset: {emg_sessions_to_exclude}")
            self.emg_sessions = [session for session in self.emg_sessions if session.session_name not in emg_sessions_to_exclude]
            self._num_sessions_excluded = len(emg_sessions) - len(self.emg_sessions)
        else:
            self._num_sessions_excluded = 0

        # Set dataset parameters
        self.date = date
        self.animal_id = animal_id
        self.condition = condition

        # Check that all sessions have the same parameters and set dataset parameters.
        consistent, message = self.__check_session_consistency()
        if not consistent:
            print(f"Error: {message}")
        else:
            self.scan_rate = self.emg_sessions[0].scan_rate
            self.num_channels = self.emg_sessions[0].num_channels
            self.stim_delay = self.emg_sessions[0].stim_delay

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
                print(session)
            else:
                raise TypeError(f"An object in the 'emg_sessions' list was not properly converted to an EMGSession. Object: {session}, {type(session)}")
            
            pickled_sessions = sorted(pickled_sessions, key=lambda x: x.session_name)
        
        return pickled_sessions

    def __check_session_consistency(self):
        """
        Checks if all sessions in the dataset have the same parameters (scan rate, num_channels, stim_delay).

        Returns:
            tuple: A tuple containing a boolean value indicating whether all sessions have consistent parameters and a message indicating the result.
        """
        reference_session = self.emg_sessions[0]
        reference_scan_rate = reference_session.scan_rate
        reference_num_channels = reference_session.num_channels
        reference_stim_delay = reference_session.stim_delay

        for session in self.emg_sessions[1:]:
            if session.scan_rate != reference_scan_rate:
                return False, f"Inconsistent scan_rate for {session.session_name}: {session.scan_rate} != {reference_scan_rate}."
            if session.num_channels != reference_num_channels:
                return False, f"Inconsistent num_channels for {session.session_name}: {session.num_channels} != {reference_num_channels}."
            if session.stim_delay != reference_stim_delay:
                return False, f"Inconsistent stim_delay for {session.session_name}: {session.stim_delay} != {reference_stim_delay}."

        return True, "All sessions have consistent parameters"

    def dataset_parameters(self):
        """
        Prints EMG dataset parameters.
        """
        session_names = [session.session_name for session in self.emg_sessions]
        print(f"EMG Sessions ({len(self.emg_sessions)}): {session_names}.")
        print(f"Date: {self.date}")
        print(f"Animal ID: {self.animal_id}")
        print(f"Condition: {self.condition}")

    def plot(self, plot_type: str = None, channel_names: Optional[List[str]] = None, **kwargs):
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
        getattr(self.plotter, f'plot_{'reflexCurves' if not plot_type else plot_type}')(channel_names=channel_names, **kwargs)

    # User methods for manipulating the EMGSession instances in the dataset.
    def add_session(self, session : Union[EMGSession, str]):
        """
        Adds an EMGSession to the emg_sessions list of this EMGDataset.

        Parameters:
            session (EMGSession or str): The session to be added. It can be an instance of EMGSession or a file path to a pickled EMGSession.

        Raises:
            TypeError: If the session is neither an instance of EMGSession nor a valid file path to a pickled EMGSession.
        """
        if session in [session.session_name for session in self.emg_sessions]:
            print(f">! Error: session {session.session_name} is already in the dataset.")
        else:
            if isinstance(session, EMGSession):
                # Add the session to the dataset.
                self.emg_sessions.append(session)
                self.emg_sessions = sorted(self.emg_sessions, key=lambda x: x.session_name)
                
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
                self.stim_delay = self.emg_sessions[0].stim_delay

    def remove_session(self, session_name : str):
        """
        Removes a session from the dataset.

        Args:
            session_name (str): The name of the session to be removed.
        """
        if session_name not in [session.session_name for session in self.emg_sessions]:
            print(f">! Error: session {session_name} not found in the dataset.")
        else:
            self.emg_sessions = [session for session in self.emg_sessions if session.session_name != session_name]

    def get_session(self, session_idx: int) -> EMGSession:
        """
        Returns the EMGSession object at the specified index.
        """
        return self.emg_sessions[session_idx]

    # Save and load the dataset object.
    def save_dataset(self, filename=None):
        """
        Save the curated dataset object to disk.

        Args:
            filename (str): The filename to use for saving the dataset object.
        """
        if filename is None:
            filename = f'{self.date} {self.animal_id} {self.condition} Dataset.pickle'
            
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_dataset(filename):
        """
        Load a previously saved dataset object from disk.

        Args:
            filename (str): The filename of the saved dataset object.

        Returns:
            EMGDataset: The loaded dataset object.
        """
        with open(filename, 'rb') as file:
            dataset = pickle.load(file)
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
        pattern = r'^(\d{6})\s([A-Z]\d+\.\d)\s(.+)$'
        
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
            print(f"Error: Dataset name {dataset_name} does not match the expected format: '[YYMMDD] [AnimalID] [Condition]'.")
            return None, None, None

    @classmethod
    def dataset_from_dataset_dict(cls, dataset_dict: dict, datasets: List[str], dataset_idx: int, emg_sessions_to_exclude: List[str] = []) -> 'EMGDataset':
        """
        Instantiates an EMGDataset from a dataset dictionary for downstream analysis.

        Args:
            dataset_dict (dict): A dictionary containing dataset information (keys = dataset names, values = dataset filepaths).
            datasets (list): A list of dataset names (keys).
            dataset_idx (int): The index of the dataset to be used.
            emg_sessions_to_exclude (list, optional): A list of EMG sessions to exclude. Defaults to an empty list.

        Returns:
            EMGDataset: The session of interest for downstream analysis.
        """
        date, animal_id, condition = cls.getDatasetInfo(datasets[dataset_idx])
        # Future: Add a check to see if the dataset is already saved as a pickle file. If it is, load the pickle file instead of re-creating the dataset.
        dataset_oi = EMGDataset(dataset_dict[datasets[dataset_idx]], date, animal_id, condition, emg_sessions_to_exclude=emg_sessions_to_exclude)
        return dataset_oi

class EMGExperiment(EMGData):
    def __init__(self, emg_datasets, emg_dataset_settings, emg_sessions_to_exclude=[]):
            super.__init__()
            pass


