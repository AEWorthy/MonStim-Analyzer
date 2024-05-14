"""
Misc. helper functions.
"""

import os
from Analyze_EMG import EMGSession as MakeSession

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
        else:
            session_name = dataset.split('-')[0]
            dataset_pickles_dict[session_name] = os.path.join(output_path, dataset).replace('\\', '/')
    return dataset_pickles_dict

def unpackEMGSessions(emg_sessions):
    """
    Unpacks a list of EMG session Pickle files and outputs a list of EMGSession instances for those pickles. If a list of EMGSession instances is passed, will return that same list.

    Args:
        emg_sessions (list): a list of instances of the class EMGSession, or a list of Pickle file locations that you want to use for the dataset.
    """
    # Check if list dtype is EMGSession. If it is, convert it to a new EMGSession instance and replace the string in the list.
    for i, object in enumerate(emg_sessions):
        if isinstance(object, str): # If list object is dtype(string), then convert to an EMGSession
            session = MakeSession(object)
            emg_sessions[i] = session

        for item in emg_sessions:
            if not isinstance(object, str):
                raise TypeError(f"An object in the 'emg_sessions' list was not properly converted to an EMGSession. Object: {item}, {type(item)}")

    return emg_sessions