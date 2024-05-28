import os
import re
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def read_csv(file_path):
    """Helper function to read a CSV file and return its lines."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def extract_session_info(lines):
    """Helper function to extract session information from CSV file lines."""
    session_id = float(next(line.split(',')[1] for line in lines if line.startswith('Session #,')))
    num_channels = int(float(next(line.split(',')[1] for line in lines if line.startswith('# of Channels,'))))
    scan_rate = float(next(line.split(',')[1] for line in lines if line.startswith('Scan Rate (Hz),')))
    num_samples = float(next(line.split(',')[1] for line in lines if line.startswith('Samples/Channel,')))
    stim_delay = float(next(line.split(',')[1] for line in lines if line.startswith('Pre-Stim Acq. time (ms),')))
    stim_duration = float(next(line.split(',')[1] for line in lines if line.startswith('Stimulus duration (ms),')))
    stim_interval = float(next(line.split(',')[1] for line in lines if line.startswith('Inter-Stim delay (sec),')))
    emg_amp_gains = []
    for line in lines:
        if line.startswith('EMG amp gain ch'):
            emg_amp_gains.append(int(float(line.split(',')[1])))

    return session_id, num_channels, scan_rate, num_samples, stim_delay, stim_duration, stim_interval, emg_amp_gains

def extract_recording_data(lines, num_channels):
    """Helper function to extract recording data from CSV file lines."""
    stimulus_v = float(next(line.split(',')[1] for line in lines if line.startswith('Stimulus Value (V),')))

    start_index = None
    data_lines = []
    for i, line in enumerate(lines):
        if line.startswith("Recorded Data (mV),"):
            start_index = i + 1
        elif start_index is not None:
            data_lines.extend([value.split(',') for value in line.strip().split('\n')])
            if line.strip() == "":
                break
    
    channel_data = np.zeros((num_channels, len(data_lines)))
    for i, row in enumerate(data_lines):
        channel_data[:, i] = np.array([float(value) for value in row])

    return stimulus_v, channel_data

def process_recording(recording_file, session_info):
    """Helper function to process a single recording file and return its data."""
    lines = read_csv(recording_file)
    test_session_id = float(next(line.split(',')[1] for line in lines if line.startswith('Session #,')))

    if test_session_id != session_info['session_id']:
        print(f'>! Error: multipleSessions_error for {test_session_id}.')
        return None

    stimulus_v, channel_data = extract_recording_data(lines, session_info['num_channels'])
    return {'stimulus_v': stimulus_v, 'channel_data': channel_data}

def process_session(dir, session_name, csv_paths, output_path):
    """Helper function to process a single session dataset."""
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

def pickle_data(data_path, output_path):
    """Main module function to create Pickle files from EMG datasets."""
    datasets = [dir for dir in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, dir))]
    print(f'Datasets to process ({len(datasets)}): {datasets}')

    for dataset_dir in datasets:
        dataset_path = os.path.join(data_path, dataset_dir)
        dataset_session_dict = getDatasetSessionDict(dataset_path)

        if len(dataset_session_dict) <= 0:
            print(f'>! Error: no CSV files detected in "{dataset_dir}." Make sure you converted STMs to CSVs.')
            continue

        if len(dataset_session_dict) == 1:
            session_name, csv_paths = next(iter(dataset_session_dict.items()))
            process_session(dataset_dir, session_name, csv_paths, output_path)
        else:
            dataset_output_path = os.path.join(output_path, dataset_dir)
            os.makedirs(dataset_output_path, exist_ok=True)
            for session_name, csv_paths in dataset_session_dict.items():
                process_session(dataset_dir, session_name, csv_paths, dataset_output_path)

    print('Processing complete.')