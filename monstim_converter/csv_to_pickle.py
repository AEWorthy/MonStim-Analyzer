import os
import re
import logging
import traceback
import pickle
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt6.QtCore import QThread, pyqtSignal

MAX_NUM_CHANNELS = 6

def read_csv(file_path):
    """Helper function to read a CSV file using pandas with optimized settings."""
    return pd.read_csv(file_path, header=None, names=range(MAX_NUM_CHANNELS), engine='c', low_memory=False, memory_map=True)

def extract_session_info_and_data(file_path):
    df = read_csv(file_path)

    if df.empty:
        raise ValueError(f"The CSV file at {file_path} is empty.")

    # Extract metadata
    data_start_marker = df[df.iloc[:, 0] == "Recorded Data (mV)"].index[0] # Find the index of the row containing "Recorded Data (mV)"
    metadata = df.iloc[:data_start_marker].set_index(0)[1].to_dict() # Select all rows above the "Recorded Data (mV)" row as metadata
    data_start_index = data_start_marker + 1 # The data start index is the row right after "Recorded Data (mV)"

    try:
        session_info = {
            'session_id': str(metadata['Session #']),
            'num_channels': int(float(metadata['# of Channels'])),
            'scan_rate': float(metadata['Scan Rate (Hz)']),
            'num_samples': float(metadata['Samples/Channel']),
            'pre_stim_acquired': float(metadata['Pre-Stim Acq. time (ms)']),
            'post_stim_acquired': float(metadata['Post-Stim Acq. time (ms)']),
            'stim_delay': float(metadata['Start Delay (ms)']),
            'stim_duration': float(metadata['Stimulus duration (ms)']),
            'stim_interval': float(metadata['Inter-Stim delay (sec)']),
            'emg_amp_gains': [int(float(metadata[f'EMG amp gain ch {i}'])) for i in range(1, int(float(metadata['# of Channels'])) + 1)]
        }
        stimulus_v = float(metadata['Stimulus Value (V)'])
    except KeyError as e:
        raise ValueError(f"Missing required metadata field: {str(e)}")
    except ValueError as e:
        raise ValueError(f"Error converting metadata value: {str(e)}")

    # Extract the recorded data and drop NaN columns
    data_df = df.iloc[data_start_index:]
    data_df = data_df.dropna(axis=1, how='all')  # Drop columns that are all NaN
    channel_data = data_df.to_numpy().T.astype(np.float32)

    # Verify number of channels and samples
    if channel_data.shape[0] != session_info['num_channels']:
        raise ValueError(f"Number of data channels ({channel_data.shape[0]}) does not match metadata ({session_info['num_channels']}) in file {file_path}.")
    
    actual_num_samples = channel_data.shape[1]
    if actual_num_samples != session_info['num_samples']:
        print(f"Warning: Number of samples in data ({actual_num_samples}) does not match metadata ({session_info['num_samples']}) in file {file_path}. Using actual number of samples.")
        session_info['num_samples'] = actual_num_samples

    return session_info, stimulus_v, channel_data

def process_recording(recording_file, expected_session_id):
    """Helper function to process a single recording file and return its data."""
    session_info, stimulus_v, channel_data = extract_session_info_and_data(recording_file)
    
    if session_info['session_id'] != expected_session_id:
        print(f'>! Error: multipleSessions_error for {session_info["session_id"]}.')
        return None

    return {'stimulus_v': stimulus_v, 'channel_data': channel_data}

def process_session(dir, session_name, csv_paths, output_path):
    """Main function to process a directory of recording CSVs into a single recording session pickle object."""
    first_csv = csv_paths[0]
    session_info, _, _ = extract_session_info_and_data(first_csv)
    session_id = session_info['session_id']

    recordings = []
    for csv_path in csv_paths:
        recording = process_recording(csv_path, session_id)
        if recording:
            recordings.append(recording)

    # Overwrite the session ID number with the actual session name.
    session_info['session_id'] = session_name

    session_data = {
        'session_info': session_info,
        'recordings': recordings
    }

    save_name = f'{dir}_{session_name}-SessionData.pickle'
    with open(os.path.join(output_path, save_name), 'wb') as pickle_file:
        pickle.dump(session_data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'> {len(recordings)} of {len(csv_paths)} CSVs processed from session "{session_name}".')

def getDatasetSessionDict(dataset_path):
    """Helper function to create an output dict containing k,v pairs of session names and their corresponding CSV file locations."""
    csv_regex = re.compile(r'.*\.csv$')
    csv_paths = [os.path.join(dataset_path, item) for item in os.listdir(dataset_path) if csv_regex.match(item)]
    csv_names = [os.path.splitext(os.path.basename(location))[0] for location in csv_paths]
    session_names = [name.split('-')[0] for name in csv_names]
    
    dataset_session_dict = {}
    for session, path in zip(session_names, csv_paths):
        dataset_session_dict.setdefault(session, []).append(path)
    
    return dataset_session_dict

def pickle_data(data_path, output_path, progress_callback=print, is_canceled_callback=lambda: False, max_workers=4):
    """Main module function to create Pickle files from EMG datasets."""
    if progress_callback is None:
        progress_callback = lambda x: None  # noqa: E731
    if is_canceled_callback is None:
        is_canceled_callback = lambda: False  # noqa: E731

    datasets = [dir for dir in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, dir))]
    total_datasets = len(datasets)
    processed_datasets = 0

    print(f'Datasets to process ({total_datasets}): {datasets}')

    def process_sessions_for_dataset(dataset_dir):
        dataset_path = os.path.join(data_path, dataset_dir)
        dataset_session_dict = getDatasetSessionDict(dataset_path)

        if len(dataset_session_dict) <= 0:
            print(f'>! Error: no CSV files detected in "{dataset_dir}." Make sure you converted STMs to CSVs.')
            return

        dataset_output_path = os.path.join(output_path, dataset_dir) if len(dataset_session_dict) > 1 else output_path
        os.makedirs(dataset_output_path, exist_ok=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_session, dataset_dir, session_name, csv_paths, dataset_output_path) 
                       for session_name, csv_paths in dataset_session_dict.items()]
            for future in as_completed(futures):
                if is_canceled_callback():
                    return
                try:
                    future.result()
                except Exception as exc:
                    logging.error(f'Error processing session: {exc}')
                    logging.error(traceback.format_exc())

    for dataset_dir in datasets:
        if is_canceled_callback():
            break
        process_sessions_for_dataset(dataset_dir)
        processed_datasets += 1
        progress_callback(int((processed_datasets / total_datasets) * 100))

    print('Processing complete.')

class GUIExptImportingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(Exception)
    canceled = pyqtSignal()

    def __init__(self, expt_name, expt_path, output_path, max_workers=None):
        super().__init__()
        self.expt_path = expt_path
        self.output_path = os.path.join(output_path, expt_name)
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
        
        self.max_workers = max_workers
        self._is_canceled = False
        self._is_finished = False

    def run(self):
        try:
            pickle_data(self.expt_path, self.output_path, self.report_progress, self.is_canceled, self.max_workers)
            if not self._is_canceled:
                self.finished.emit()
                self._is_finished = True
        except Exception as e:
            if not self._is_canceled:
                self.error.emit(e)
                logging.error(f'Error in GUIDataProcessingThread: {e}')
                logging.error(traceback.format_exc())

    def report_progress(self, value):
        if not self._is_canceled:
            self.progress.emit(value)

    def cancel(self):
        if not self._is_canceled and not self._is_finished:
            self._is_canceled = True
            self.canceled.emit()

    def is_canceled(self):
        return self._is_canceled
    
    def is_finished(self):
        return self._is_finished
    

    
if __name__ == '__main__':
    # Example usage of the pickle_data function with max_workers
    import multiprocessing
    max_workers = multiprocessing.cpu_count()  # Use the number of CPU cores
    pickle_data('files_to_analyze', 'pickles', max_workers=max_workers)