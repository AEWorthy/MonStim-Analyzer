
"""
Classes to analyze and plot EMG data from individual sessions or an entire dataset of sessions.
"""

import os
import pickle
import copy
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import yaml
import tkinter as tk
from tkinter import ttk

from Plot_EMG import EMGSessionPlotter, EMGDatasetPlotter
import Transform_EMG



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
    One session contains multiple recordings that will make up, for example, a single M-curve.

    This module provides functions for analyzing data stored in Pickle files from a single EMG recording session.
    It includes functions to extract session parameters, plot all EMG data, and plot EMG data from suspected H-reflex recordings.
    Class must be instantiated with the Pickled session data file.

    Attributes:
        plotter (EMGSessionPlotter): An instance of the EMGSessionPlotter class for plotting EMG data. 
            - Types of plotting commands include: plot_emg, plot_emg_susectedH, and plot_reflex_curves. 
            - See the EMGSessionPlotter class in Plot_EMG.py for more details.

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
        self.plotter = EMGSessionPlotter(self)
        self.load_session_data(pickled_data)

    def load_session_data(self, pickled_data):
        
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
                    filtered_emg = Transform_EMG.butter_bandpass_filter(channel_emg, self.scan_rate, **self.butter_filter_args)
                    if rectify:
                        recording['channel_data'][i] = Transform_EMG.rectify_emg(filtered_emg)
                    else:
                        recording['channel_data'][i] = filtered_emg
                elif rectify:
                    rectified_emg = Transform_EMG.rectify_emg(channel_emg)
                    recording['channel_data'][i] = rectified_emg
                
                # Apply baseline correction to the processed data if a filter was applied.
                if apply_filter:
                    recording['channel_data'] = Transform_EMG.correct_emg_to_baseline(recording['channel_data'], self.scan_rate, self.stim_delay)
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
    
    def __init__(self, emg_sessions, date, animal_id, condition, emg_sessions_to_exclude=[]):
        """
        Initialize an EMGDataset instance from a list of EMGSession instances for multi-session analyses and plotting.

        Args:
            emg_sessions (list): A list of instances of the class EMGSession, or a list of Pickle file locations that you want to use for the dataset.
            emg_sessions_to_exclude (list, optional): A list of session names to exclude from the dataset. Defaults to an empty list.
        """
        self.plotter = EMGDatasetPlotter(self)
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

        # Set dataset parameters
        self.date = date
        self.animal_id = animal_id
        self.condition = condition

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
            
            pickled_sessions = sorted(pickled_sessions, key=lambda x: x.session_name)
        
        return pickled_sessions

    def add_session(self, session):
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
                self.emg_sessions.append(session)
                self.emg_sessions = sorted(self.emg_sessions, key=lambda x: x.session_name)
            else:
                try:
                    self.emg_sessions.append(EMGSession(session))
                except:
                    raise TypeError("Expected an instance of EMGSession or a file path to a pickled EMGSession.")

    def remove_session(self, session_name):
        """
        Removes a session from the dataset.

        Args:
            session_name (str): The name of the session to be removed.
        """
        if session_name not in [session.session_name for session in self.emg_sessions]:
            print(f">! Error: session {session_name} not found in the dataset.")
        else:
            self.emg_sessions = [session for session in self.emg_sessions if session.session_name != session_name]

    def dataset_parameters(self):
        """
        Prints EMG dataset parameters.
        """
        session_names = [session.session_name for session in self.emg_sessions]
        print(f"EMG Sessions ({len(self.emg_sessions)}): {session_names}.")
        print(f"Date: {self.date}")
        print(f"Animal ID: {self.animal_id}")
        print(f"Condition: {self.condition}")

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

class EMGExperiment:
    def __init__(self, emg_datasets, emg_dataset_settings, emg_sessions_to_exclude=[]):
            pass


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

def getDatasetInfo(dataset_name):
    """
    Extracts information from a dataset name.

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
    date, animal_id, condition = getDatasetInfo(datasets[dataset_idx])
    dataset_oi = EMGDataset(dataset_dict[datasets[dataset_idx]], date, animal_id, condition, emg_sessions_to_exclude=emg_sessions_to_exclude)
    return dataset_oi