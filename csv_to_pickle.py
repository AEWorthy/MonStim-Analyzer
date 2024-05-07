# csv_to_pickle.py

"""Module for converting .csv files in the /files_to_analyze directory into pickle files saved in the /output directory.

Generated pickle files are formatted for easy downstream analysis.

Data should be imported into /files_to_analyze as individual directories, each containing CSVs of recordings from a single session.

Non-CSV files in session data folders will be ignored by the program.
"""

import os
import re
import pickle

def pickle_csv(dataset, csv_paths, output_path):

    """Helper function to do create a pickle file from a CSV dataset.
    
    Args:
        dataset (str): Directory name of the dataset/session to be processed (folder name in /files_to_analyze).
        csv_paths (list): Relative file paths of all CSV files in the dataset.
        output_path (str): Path to the /output folder or other location for Pickle files to be saved.
        
    Returns:
        tuple: A tuple containing the total number of CSV files in the given session/dataset and the number of CSV files that were successfully included in the Pickle file.
    """
    
    # Gather session_info from the first CSV of a dataset
    first_csv = csv_paths[0]
    session_id = None
    channel_error = False # flag for future checkpoint
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
            'session_name' : dataset,
            'num_channels': num_channels,
            'scan_rate': int(scan_rate),
            'num_samples': int(num_samples),
            'stim_duration' : stim_duration,
            'stim_interval' : stim_interval,
            'emg_amp_gains': emg_amp_gains
        },
        'recordings': []
    }

    # Process each recording for stimulus and EMG data
    unsaved_data_files = [] # initilizing list for data files that flagged errors.
    for recording_file in csv_paths:  # Replace with your list of recording files
        with open(recording_file, 'r') as file:
            lines = file.readlines() # load CSV lines into memory.
            test_session_id = float(next(line.split(',')[1] for line in lines if line.startswith('Session #,')))
            
            if test_session_id != session_id: # Test if a second session's file is detected
                unsaved_data_files.append(recording_file)
                continue
        
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

            if num_channels != len(data_lines[0]): # test for a mis-match in the cvs number of channels and the declared number of channels.
                unsaved_data_files.append(recording_file)
                channel_error = True
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
                
    #Check for channel recording errors.
    if channel_error:
        print(f'>! Error: mis-match in declared and recorded channels detected in {dataset}.')
        print('\tData from the following recordings were not saved to the session data file:')
        for file in unsaved_data_files:
            print(f'\t\t{file}')
        pass
    # Check for unsaved files with errors.
    if len(unsaved_data_files) > 0 and not channel_error:
        print(f'>! Error: multiple recording sessions detected in {dataset}.')
        print('\tData from the following recordings were not saved to the session data file:')
        for file in unsaved_data_files:
            print(f'\t\t{file}')
        pass

    # Save the session data to its own pickle file
    save_name = dataset + '_data.pickle'
    with open(os.path.join(output_path, save_name), 'wb') as pickle_file:
        pickle.dump(session_data, pickle_file)

    # Return the number of CSV files attempted and successfully processed.
    num_csvs = len(csv_paths)
    num_csv_success = len(csv_paths) - len(unsaved_data_files)
    return num_csvs, num_csv_success

def pickle_single_sessions (data_path, output_path):

    """Converts CSV files of a single recording sessions in the specified data directory to Pickle files.
    
    Args:
        data_path (str): Path to the /files_to_analyze folder or other input folder.
        output_path (str): Path to the /output folder or other location for Pickle files to be saved.
    """

    # Process "\files_to_analyze" into Pickle files
    datasets = [dir for dir in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, dir))] # list all datasets in "files_to_analyze" folder
    print(f'Datasets to process: {datasets}')

    # Create a pickle file for each dataset in files_to_analyze
    for dataset in datasets:
        dataset_path = os.path.join(data_path, dataset)#.replace('\\', '/')
        
        csv_regex = re.compile(r'.*\.csv$') #regex to match CSV files only.
        csv_names = [item for item in os.listdir(dataset_path) if csv_regex.match(item)] #list of CSV filenames in dataset_path.
        csv_paths = [os.path.join(dataset_path, csv_name) for csv_name in csv_names]

        # check if there are CSVs in this dataset.
        if len(csv_paths) <= 0:
            print(f'>! Error: no CSV files detected in "{dataset}." Make sure you converted STMs to CSVs.')
            continue

        # Call function to extract dataset CSVs into a pickle file.
        num_csvs, num_csv_success = pickle_csv(dataset, csv_paths, output_path)
        print(f'> {num_csv_success} of {num_csvs} CSVs processed from dataset "{dataset}".')

    print('Processing complete.')

def pickle_dataset (dataset_path, output_path):
    """Converts CSV files of a dataset of recording sessions in the specified data directory into Pickle files for each individual session. Returns .
    
    Args:
        data_path (str): Path to the /files_to_analyze folder or other input folder.
        output_path (str): Path to the /output folder or other location for Pickle files to be saved.
    """

    # Process "\files_to_analyze" into Pickle files
    datasets = [dir for dir in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, dir))] # list all datasets in "files_to_analyze" folder
    print(f'Datasets to process: {datasets}')

    # Create a pickle file for each dataset in files_to_analyze
    for dataset in datasets:
        dataset_path = os.path.join(data_path, dataset)#.replace('\\', '/')
        
        csv_regex = re.compile(r'.*\.csv$') #regex to match CSV files only.
        csv_names = [item for item in os.listdir(dataset_path) if csv_regex.match(item)] #list of CSV filenames in dataset_path.
        csv_paths = [os.path.join(dataset_path, csv_name) for csv_name in csv_names]

        # check if there are CSVs in this dataset.
        if len(csv_paths) <= 0:
            print(f'>! Error: no CSV files detected in "{dataset}." Make sure you converted STMs to CSVs.')
            continue

        # Call function to extract dataset CSVs into a pickle file.
        num_csvs, num_csv_success = pickle_csv(dataset, csv_paths, output_path)
        print(f'> {num_csv_success} of {num_csvs} CSVs processed from dataset "{dataset}".')


    print('Processing complete.')