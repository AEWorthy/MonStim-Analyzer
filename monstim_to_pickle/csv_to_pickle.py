import os
import re
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


def read_csv(file_path):
    """Helper function to read a CSV file and return its lines."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def extract_session_info(lines):
    """Helper function to extract session information from CSV file lines."""
    session_id = num_channels = scan_rate = num_samples = stim_delay = stim_duration = stim_interval = None
    emg_amp_gains = []
    
    for line in lines:
        if line.startswith('Session #,'):
            session_id = float(line.split(',')[1])
        elif line.startswith('# of Channels,'):
            num_channels = int(float(line.split(',')[1]))
        elif line.startswith('Scan Rate (Hz),'):
            scan_rate = float(line.split(',')[1])
        elif line.startswith('Samples/Channel,'):
            num_samples = float(line.split(',')[1])
        elif line.startswith('Pre-Stim Acq. time (ms),'):
            stim_delay = float(line.split(',')[1])
        elif line.startswith('Stimulus duration (ms),'):
            stim_duration = float(line.split(',')[1])
        elif line.startswith('Inter-Stim delay (sec),'):
            stim_interval = float(line.split(',')[1])
        elif line.startswith('EMG amp gain ch'):
            emg_amp_gains.append(int(float(line.split(',')[1])))
        # Break loop if we reach the end of the session info portion of the CSV.
        elif line.startswith('Recorded Data (mV),'):
            break

    return session_id, num_channels, scan_rate, num_samples, stim_delay, stim_duration, stim_interval, emg_amp_gains

def extract_recording_data(lines, num_channels):
    """Helper function to extract recording data from CSV file lines."""

    data_start_index = None
    stimulus_v = None
    # Find the index of the first line of data and the stimulus voltage.
    for i, line in enumerate(lines):
        if line.startswith("Stimulus Value (V),"):
            stimulus_v = float(line.split(',')[1])
        if line.startswith("Recorded Data (mV),"):
            data_start_index = i + 1
            break
    
    data_lines = []
    # Extract the data lines.
    for i, line in enumerate(lines[data_start_index:]):
        stripped_line = line.strip()
        if stripped_line == "":
            break
        data_lines.append([float(value) for value in stripped_line.split(',')])
    
    if not data_lines:
        print('>! Error: no data lines found in CSV file.')
        return np.zeros((num_channels, 0)), stimulus_v

    data_lines_transposed = list(zip(*data_lines))  # Transpose the data lines
    channel_data = np.array(data_lines_transposed[:num_channels])
    return stimulus_v, channel_data

def process_recording(recording_file, session_info):
    """Helper function to process a single recording file and return its data."""
    lines = read_csv(recording_file)

    # Check if the session ID in the CSV matches the session ID in the session info.
    for line in lines:
        if line.startswith('Session #,'):
            test_session_id = float(line.split(',')[1])
            if test_session_id != session_info['session_id']:
                print(f'>! Error: multipleSessions_error for {test_session_id}.')
                return None
            break
    
    # Extract the stimulus voltage and channel data from the CSV.
    stimulus_v, channel_data = extract_recording_data(lines, session_info['num_channels'])
    return {'stimulus_v': stimulus_v, 'channel_data': channel_data}

def process_session(dir, session_name, csv_paths, output_path):
    """Main function to process a directory of recording CSVs into a single recording session pickle object."""
    first_csv = csv_paths[0]
    lines = read_csv(first_csv)
    session_id, num_channels, scan_rate, num_samples, stim_delay, stim_duration, stim_interval, emg_amp_gains = extract_session_info(lines)

    session_info = {
        'session_name': session_name,
        'session_id': session_id,
        'num_channels': int(num_channels),
        'scan_rate': int(scan_rate),
        'num_samples': int(num_samples),
        'stim_delay': stim_delay,
        'stim_duration': stim_duration,
        'stim_interval': stim_interval,
        'emg_amp_gains': emg_amp_gains
    }

    with ProcessPoolExecutor() as executor:
        recordings = list(filter(None, executor.map(process_recording, csv_paths, [session_info] * len(csv_paths))))

    session_data = {
        'session_info': session_info,
        'recordings': recordings
    }

    save_name = f'{dir}_{session_name}-SessionData.pickle'
    with open(os.path.join(output_path, save_name), 'wb') as pickle_file:
        pickle.dump(session_data, pickle_file)

    print(f'> {len(recordings)} of {len(csv_paths)} CSVs processed from session "{session_name}".')

def getDatasetSessionDict(dataset_path):
    """Helper function to create an output dict containing k,v pairs of session names and their corresponding CSV file locations."""
    csv_regex = re.compile(r'.*\.csv$')
    csv_filenames = [item for item in os.listdir(dataset_path) if csv_regex.match(item)]
    csv_paths = [os.path.join(dataset_path, csv_filename) for csv_filename in csv_filenames]
    csv_names = [os.path.splitext(os.path.basename(location))[0] for location in csv_paths]
    session_names = [name.split('-')[0] for name in csv_names]
    unique_session_names = list(set(session_names))

    dataset_session_dict = {session: [csv_paths[i] for i in range(len(csv_paths)) if session_names[i] == session] for session in unique_session_names}
    return dataset_session_dict

def pickle_data(data_path, output_path, progress_callback=None, is_canceled_callback=None):
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

        if len(dataset_session_dict) > 1:
            dataset_output_path = os.path.join(output_path, dataset_dir)
            os.makedirs(dataset_output_path, exist_ok=True)
        else:
            dataset_output_path = output_path

        with ThreadPoolExecutor() as executor:
            futures = []
            for session_name, csv_paths in dataset_session_dict.items():
                if is_canceled_callback():
                    return
                futures.append(executor.submit(process_session, dataset_dir, session_name, csv_paths, dataset_output_path))
            
            for future in as_completed(futures):
                if is_canceled_callback():
                    return
                try:
                    future.result()
                except Exception as exc:
                    print(f'>! Error in processing session: {exc}')

    for dataset_dir in datasets:
        if is_canceled_callback():
            break
        process_sessions_for_dataset(dataset_dir)
        processed_datasets += 1
        progress_callback(int((processed_datasets / total_datasets) * 100))

    print('Processing complete.')