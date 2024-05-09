# csv_to_pickle.py

"""Module for converting EMG recording datasets in the '/files_to_analyze' directory into pickle files saved in the '/output' directory. Pickle files are generated for each single recording session, and are formatted for easy downstream analysis. For datasets containing multiple recording session CSVs, a folder will be generated in '/output' that contains Pickle files for each session in the dataset.

Data should be imported into '/files_to_analyze' as individual directories, each containing CSVs converted from STM files of MonStim-Optical-V3D EMG recordings.

Folders in the '/files_to_analyze directory' without any CSV files will be ignored by the program.
"""

import os
import re
import pickle

def getDatasetSessionDict (dataset_path):

    """Helper function to do create an output dict containing k,v pairs of session names and their corresponding CSV file locations.
    
    Args:
        dataset_path (str): Directory name of the dataset/session to be processed (folder name in /files_to_analyze).
        
    Returns:
        dict: keys = unique recording session names in the dataset, values = file locations of all CSVs taken in this recording session (e.g., same session filename).
    """
    
    csv_regex = re.compile(r'.*\.csv$') #regex to match CSV files only.
    csv_filenames = [item for item in os.listdir(dataset_path) if csv_regex.match(item)] # list of CSV filenames in dataset_path (ex.: "AA64-0023.csv").
    csv_paths = [os.path.join(dataset_path, csv_filename) for csv_filename in csv_filenames] # list of CSV file locations (ex.: "file/location/AA64-0023.csv").
    csv_names = [os.path.splitext(os.path.basename(location))[0] for location in csv_paths] # list of CSV file names w/o file type (ex.: "AA64-0023").
    session_names = [name.split('-')[0] for name in csv_names] # list of session names for each CSV file (ex.: "AA64").
    unique_session_names = list(set(session_names)) # list of unique session names

    # Create a dictionary with session names as keys and lists of corresponding file locations as values.
    dataset_session_dict = {session: [csv_paths[i] for i in range(len(csv_paths)) if session_names[i] == session] for session in unique_session_names}
    
    return dataset_session_dict

def error_check (dir, error_dict):
    
    """Helper function to check for processing errors and flag them.
    
    Args:
        dir (str): name of the dataset directory being processed.
        error_dict (dict): keys = error variables, values = (bool).
    """

    if error_dict['channel_error']:
        print(f'>! Error: mis-match in declared and recorded channels detected in {dir}.')
        print('\tData from the following recordings were not saved to the session data file:')
        for file in error_dict['unsaved_data_files']:
            print(f'\t\t{file}')
        pass

    if error_dict['multipleSessions_error']:
        print(f'>! Error: multipleSessions_error in {dir}.')
        print('\tData from the following recordings were not saved to the session data file:')
        for file in error_dict['unsaved_data_files']:
            print(f'\t\t{file}')
        pass

def pickle_session(dir, session_name, csv_paths, output_path):

    """Helper function to create a Pickle file from a single session dataset of CSV files.
    
    Args:
        dir (str): name of the dataset directory being processed.
        session_name (str): name of the session to be processed.
        csv_paths (list): Relative file paths of all CSV files in the dataset.
        output_path (str): Path to the /output folder or other location for Pickle files to be saved.
        
    Returns:
        tuple: A tuple containing the total number of CSV files in the given session/dataset and the number of CSV files that were successfully included in the Pickle file.
    """
    
    # Gather session_info from the first CSV of a dataset
    first_csv = csv_paths[0]
    with open(first_csv, 'r') as file:
        lines = file.readlines() # load .csv data lines into memory

        # load desired session parameters
        session_id = float(next(line.split(',')[1] for line in lines if line.startswith('Session #,')))
        num_channels = int(float(next(line.split(',')[1] for line in lines if line.startswith('# of Channels,'))))
        scan_rate = float(next(line.split(',')[1] for line in lines if line.startswith('Scan Rate (Hz),')))
        num_samples = float(next(line.split(',')[1] for line in lines if line.startswith('Samples/Channel,')))
        
        stim_duration = float(next(line.split(',')[1] for line in lines if line.startswith('Stimulus duration (ms),')))
        stim_interval = float(next(line.split(',')[1] for line in lines if line.startswith('Inter-Stim delay (sec),')))
        
        emg_amp_gains = []
        for line in lines:
            if line.startswith('EMG amp gain ch'):
                emg_amp_gains.append(int(float(line.split(',')[1])))

    # Create a dictionary to store the session data
    session_data = {
        'session_info': {
            'session_name' : session_name,
            'num_channels': num_channels,
            'scan_rate': int(scan_rate),
            'num_samples': int(num_samples),
            'stim_duration' : stim_duration,
            'stim_interval' : stim_interval,
            'emg_amp_gains': emg_amp_gains
        },
        'recordings': []
    }
    # Initialize a dictionary to store information about processing errors
    error_dict = {
        'channel_error' : False,
        'multipleSessions_error': False,
        'unsaved_data_files' : []
    }

    # Process each recording for stimulus and EMG data
    for recording_file in csv_paths:  # Replace with your list of recording files
        with open(recording_file, 'r') as file:
            lines = file.readlines() # load CSV lines into memory.
            test_session_id = float(next(line.split(',')[1] for line in lines if line.startswith('Session #,')))
            
            # Test if a second session's file is detected. This should never be true.
            if test_session_id != session_id: 
                error_dict['unsaved_data_files'].append(recording_file) if recording_file not in error_dict['unsaved_data_files'] else None
                error_dict['multipleSessions_error'] = True
            
        
            # Extract Stimulus Value (in volts)
            stimulus_v = float(next(line.split(',')[1] for line in lines if line.startswith('Stimulus Value (V),')))

            # Extract EMG "Recorded Data (mV)"
            start_index = None
            data_lines = []
            for i, line in enumerate(lines):
                if line.startswith("Recorded Data (mV),"):
                    start_index = i + 1
                elif start_index is not None:
                    data_lines.extend([value.split(',') for value in line.strip().split('\n')])
                    if line.strip() == "":
                        break
            # test for a mis-match in the cvs number of channels and the declared number of channels.
            if num_channels != len(data_lines[0]):
                error_dict['unsaved_data_files'].append(recording_file) if recording_file not in error_dict['unsaved_data_files'] else None
                error_dict['channel_error'] = True
                continue
            
            # Create a list to store data for each channel
            channel_data = [[] for _ in range(num_channels)]

            # Populate the channel data
            for row in data_lines:
                for i, value in enumerate(row):
                    channel_data[i].append(float(value))

            # Update the session_data with the number of channels and channel_data
            session_data['session_info']['num_channels'] = num_channels
            session_data['recordings'].append({
                'stimulus_v': stimulus_v,
                'channel_data': channel_data
            })
                    
    # Save the session data to its own pickle file only if there were not multiple sessions detected.
    save_name = f'{dir}_{session_name}-SessionData.pickle'
    with open(os.path.join(output_path, save_name), 'wb') as pickle_file:
        pickle.dump(session_data, pickle_file)
    
    # Confirm Pickled session CSVs and return whether the given file is a multi-session dataset that needs to be handled differently.
    error_check(dir, error_dict)
    num_csvs = len(csv_paths)
    num_csv_success = len(csv_paths) - len(error_dict['unsaved_data_files'])
    print(f'> {num_csv_success} of {num_csvs} CSVs processed from dataset "{session_name}".')

def pickle_data (data_path, output_path):

    """Main module function to create a Pickle files from EMG datasets.
    
    Args:
        data_path (str): filepath to location where datasets are located.
        output_path (str): filepath to location where output files will be saved.
    """

    # List all datasets in "files_to_analyze" folder
    datasets = [dir for dir in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, dir))]
    print(f'Datasets to process: {datasets}')

    # Create a pickle file for each dataset in files_to_analyze
    for dataset_dir in datasets:
        dataset_path = os.path.join(data_path, dataset_dir)#.replace('\\', '/')
        dataset_session_dict = getDatasetSessionDict(dataset_path)
        multipleSessions = False # Default this value to False for each directory until proven otherwise.
        
        # check if there are even CSVs in this dataset.
        if len(dataset_session_dict) <= 0:
            print(f'>! Error: no CSV files detected in "{dataset_dir}." Make sure you converted STMs to CSVs.')
            continue

        # check if there are multiple sessions in the dataset
        if len(dataset_session_dict) > 1:
            multipleSessions = True

        # Process the dataset if it contains only a single session.
        if not multipleSessions:
            session_name, csv_paths = next(iter(dataset_session_dict.items()))
            pickle_session(dataset_dir, session_name, csv_paths, output_path)
            continue

        # Process the dataset if it contains multiple sessions and error check.
        if multipleSessions:
            dataset_output_path = os.path.join(output_path,dataset_dir)
            if not os.path.exists(dataset_output_path):
                os.mkdir(dataset_output_path)

            for session_name, csv_paths in dataset_session_dict.items():
                pickle_session(dataset_dir, session_name, csv_paths, dataset_output_path)

    print('Processing complete.')
